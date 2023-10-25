#ifndef MB3T_HIST_MODEL_HPP
#define MB3T_HIST_MODEL_HPP

#include <opencv2/core.hpp>
#include "hist_modality.hpp"

namespace mb3t {
namespace hm {

    class ViewpointBuilder { 
    public:

        bool GeneratePointData(const cv::Mat& silhouette, const cv::Mat& depth, const Transform3fA& body2camera_pose, std::vector<ContourPoint>& points) const;

    private:
        int max_n_points;
        int min_n_points;
    };


}
}

#endif