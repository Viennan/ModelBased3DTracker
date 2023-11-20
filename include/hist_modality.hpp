#ifndef MB3T_HIST_MODALITY_HPP
#define MB3T_HIST_MODALITY_HPP

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "sparse_viewpoint_model.hpp"
#include "camera.hpp"

namespace mb3t {
namespace hm {

    struct ContourPoint {
        Eigen::Vector3f orientation;
        Eigen::Vector3f center_f_body;
        Eigen::Vector3f normal_f_body;
        float foreground_distance = 0.0f;
        float background_distance = 0.0f;
        DepthOffsets depth_offsets{};
    };

    // A corresponding line in 2D
    struct Line {
        Eigen::Vector3f center_f_body{};
        Eigen::Vector3f center_f_camera{};
        Eigen::Vector2f center_uv;
        Eigen::Vector2f normal_uv;
        float delta_r = 0.0f;
        float normal_component_to_scale = 0.0f;
        std::vector<float> distribution{};
        float mean = 0.0f;
        float variance = 0.0f;
        float reciprocal_variance = 0.0f;
    };

    class Histogram {
    public:
        Histogram() = default;
        Histogram(const Histogram&) = default;
        Histogram(Histogram&&) = default;
        Histogram& operator=(const Histogram&) = default;
        Histogram& operator=(Histogram&&) = default;

        explicit Histogram(int n_bins) {
            Setup(n_bins);
        };

        void Setup(int n_bins) {
            hist_n_bins = n_bins;
            hist_n_bins_square_ = n_bins * n_bins;
            hist_n_bins_cube_ = n_bins * n_bins * n_bins;
            hist_.resize(hist_n_bins_cube_);
            n_bits_shift_ = std::log2(n_bins);
        };

        void Init() {
            std::fill(begin(hist_), end(hist_), 0.0f);
        };

        void Incr(const cv::Vec3b& color) {
            hist_[int(color[0] >> n_bits_shift_) * hist_n_bins_square_ + 
                  int(color[1] >> n_bits_shift_) * hist_n_bins + 
                  int(color[2] >> n_bits_shift_)] += 1.0f;
        }

        float operator() (const cv::Vec3b& color) const {
            return hist_[int(color[0] >> n_bits_shift_) * hist_n_bins_square_ + 
                         int(color[1] >> n_bits_shift_) * hist_n_bins + 
                         int(color[2] >> n_bits_shift_)];
        }

        void Normalize();

        void MergeUnnormalized(const Histogram& other, float learning_rate);

    private:
        int hist_n_bins; // must be power of 2
        int n_bits_shift_;
        int hist_n_bins_square_;
        int hist_n_bins_cube_;
        std::vector<float> hist_;
    };

    using PointFilter = std::function<void(const Viewpoint<ContourPoint>&, std::vector<int>&)>;

    class HistModality {
    public:
        HistModality() = default;
        HistModality(const HistModality&) = default;
        HistModality(HistModality&&) = default;
        HistModality& operator=(const HistModality&) = default;
        HistModality& operator=(HistModality&&) = default;
        ~HistModality() = default;

    private:
        // parameters and functions for camera
        Camera camera;
        // the following params about camera will update before creating correspondence lines
        // start...
        Transform3fA body2camera_;
        Eigen::Matrix3f body2camera_rotation_;
        Eigen::Matrix<float, 2, 3> body2camera_rotation_xy_;
        Intrinsics intrinsics_;
        // end
        void camera_update_params();

        // parameters and functions for smoothed step function (ssf)
        float ssf_amptitue;
        float ssf_slope;
        int ssf_length;
        std::vector<float> ssf_lookup_f_;
        std::vector<float> ssf_lookup_b_;
        void ssf_init();

        // parameters and functions for histogram
        float hist_learning_rate_f;
        float hist_learning_rate_b;
        int hist_unconsidered_line_length;
        int hist_max_considered_line_length;
        int hist_n_bins;
        Histogram hist_temp_f_;
        Histogram hist_temp_b_;
        Histogram hist_f_;
        Histogram hist_b_;
        void hist_setup(); // setup params about histogram
        void hist_update_temp(const Viewpoint<ContourPoint>&); // initialize histogram under current camera pose and captured image

        // parameters and functions for sparse viewpoint model
        SparseViewpointModel<ContourPoint> viewpoint_model;

        // parameter and functions for correspondence line
        // the following params block should be set before modality is used
        std::vector<int> line_scales;
        int line_distribution_length;

        int line_length_in_segment_;
        int line_distribution_length_minus_1_half_;
        int line_distribution_length_plus_1_half_;
        float min_expected_variance_;
        void line_init_distribution_params(); // this function will init the beyond param block before modality start

        int line_scale_;
        float line_fscale_;
        int line_length_;
        int line_length_minus_1_;
        int line_length_minus_1_half_;
        void line_init_scale_dependent_params(int scale_idx); // this function will init the beyond param block before calculating correspondence lines

        std::vector<Line> lines_; // correspondence lines
        std::vector<float> line_normalized_reciprocal_variances_; // used for weighting gradient and hessian of each correspondence line
        float line_average_variance_;
        void line_search_and_project_centers(); // only search and project center of correspondence lines, do not calculate any distribution of them
        bool line_calculate_segment_color_distribution(const cv::Mat& image, Line& line, std::vector<float>& segment_f_distribution, std::vector<float>& segment_b_distribution);
        void line_calculate_segment_distribution(Line& line, const std::vector<float>& segment_f_distribution, const std::vector<float>& segment_b_distribution);
        void line_calculate_correspondence(int scale_idx);

        // parameters and functions for point filtering, usually used for removing outliers
        std::vector<PointFilter> line_point_filters;

        // parameters and functions for non-linear optimization
        int opt_n_global_iteration;
        Eigen::Matrix<float, 6, 1> gradient_;
        Eigen::Matrix<float, 6, 6> hessian_;
        float mod_variance_in_pixel_; // for weighting between different modalities
        void optimize(int iteration);
    };

}
}

#endif