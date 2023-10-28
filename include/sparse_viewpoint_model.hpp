#ifndef MB3T_SPARSE_VIEWPOINT_MODEL_HPP
#define MB3T_SPARSE_VIEWPOINT_MODEL_HPP

#include <vector>
#include "common.hpp"
#include <opencv2/core.hpp>

namespace mb3t {

constexpr int kMaxNDepthOffsets = 30;
using DepthOffsets = std::array<float, kMaxNDepthOffsets>;

template<typename Data>
const Data& GetClosestViewpoint(const Transform3fA& body2camera_pose, const std::vector<Data>& viewpoints) {
    // convert to camera to body orientation
    Eigen::Vector3f orientation{body2camera_pose.rotation().inverse() * body2camera_pose.translation().matrix().normalized()};
    float closest_dot = -1.0f;
    size_t index = 0;
    for (size_t i = 0; i < viewpoints.size(); ++i) {
        float dot = distance(viewpoints[i].orientation, orientation);
        if (dot > closest_dot) {
            index = i;
        }
    }
    return viewpoints_[index];
}

void GenerateGeodesicPoses(int n_divides, float radius, std::vector<Transform3fA>& camera2body_poses);

void CalculateDepthOffsets(
    const cv::Mat& depth_image, 
    const cv::Point2i &center,
    float pixel_to_meter,
    float max_radius_depth_offset,
    float stride_depth_offset,
    DepthOffsets& depth_offsets);

}

#endif