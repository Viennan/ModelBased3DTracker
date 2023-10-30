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
        float measured_depth_offset = 0.0f;
        float modeled_depth_offset = 0.0f;
        float continuous_distance = 0.0f;
        float delta_r = 0.0f;
        float normal_component_to_scale = 0.0f;
        std::vector<float> distribution{};
        float mean = 0.0f;
        float measured_variance = 0.0f;
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

        void Normalize();

        void MergeUnnormalized(const Histogram& other, float learning_rate);

    private:
        int hist_n_bins; // must be power of 2
        int n_bits_shift_;
        int hist_n_bins_square_;
        int hist_n_bins_cube_;
        std::vector<float> hist_;
    };

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
        std::vector<int> scales;
        std::vector<Line> lines_;
        void calculateCorrespondenceLines(int scale, int iteration);

        // parameters and functions for point filtering, usually used for removing outliers

    };

}
}

#endif