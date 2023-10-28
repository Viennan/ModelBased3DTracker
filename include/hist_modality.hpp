#ifndef MB3T_HIST_MODALITY_HPP
#define MB3T_HIST_MODALITY_HPP

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "sparse_viewpoint_model.hpp"

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

    // A line in 2D
    struct Line {

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
        int hist_n_bins;
        int hist_n_bins_square_;
        int hist_n_bins_cube_;
        std::vector<float> hist_temp_f_;
        std::vector<float> hist_temp_b_;
        std::vector<float> hist_f_;
        std::vector<float> hist_b_;
        void hist_init();

        // parameters and functions for sparse viewpoint model
        std::vector<ContourPoint> viewpoint_model_;
    };

}
}

#endif