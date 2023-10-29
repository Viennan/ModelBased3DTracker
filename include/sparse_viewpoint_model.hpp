#ifndef MB3T_SPARSE_VIEWPOINT_MODEL_HPP
#define MB3T_SPARSE_VIEWPOINT_MODEL_HPP

#include <vector>
#include "common.hpp"
#include <opencv2/core.hpp>

namespace mb3t {

constexpr int kMaxNDepthOffsets = 30;
using DepthOffsets = std::array<float, kMaxNDepthOffsets>;

// calculate the relative depth feature for handling occlusion
void CalculateDepthOffsets(
    const cv::Mat& depth_image, 
    const cv::Point2i &center,
    float pixel_to_meter,
    float max_radius_depth_offset,
    float stride_depth_offset,
    DepthOffsets& depth_offsets);

void GenerateGeodesicPoses(int n_divides, float radius, std::vector<Transform3fA>& camera2body_poses);

template<typename PointData>
struct Viewpoint{
    Eigen::Vector3f oritenation;
    std::vector<PointData> data;
};

template<typename PointData>
using ViewnpointBuilder = std::function<bool(const Transform3fA& camera2pose, std::vector<PointData>&)>;

template<typename PointData>
class SparseViewpointModel {
public:
    SparseViewpointModel() = default;
    SparseViewpointModel(const SparseViewpointModel&) = default;
    SparseViewpointModel(SparseViewpointModel&&) = default;

    SparseViewpointModel(int n_divides, float radius, int n_points_per_view) : 
        n_divides_{n_divides}, radius_{radius}, n_points_per_view_{n_points_per_view} {}

    SparseViewpointModel(int n_divides, float radius, int n_points_per_view, std::vector<Viewpoint<PointData>>&& data):
        n_divides_{n_divides}, radius_{radius}, n_points_per_view_{n_points_per_view}, data_{std::move(data)} {}

    const Viewpoint<PointData>& GetClosestViewpoint(const Transform3fA& body2camera_pose) const {
         // convert to camera to body orientation
        Eigen::Vector3f orientation{body2camera_pose.rotation().inverse() * body2camera_pose.translation().matrix().normalized()};
        float closest_dot = -1.0f;
        size_t index = 0;
        for (size_t i = 0; i < data_.size(); ++i) {
            float dot = data_[i].orientation.dot(orientation);
            if (dot > closest_dot) {
                index = i;
            }
        }
        return data_[index];
    }

    bool calculateViewpoints(ViewnpointBuilder<PointData> viewpoint_builder) {
        data_.clear();
        std::vector<Transform3fA> camera2body_poses;
        GenerateGeodesicPoses(n_divides_, radius_, camera2body_poses);
        for (const auto& pose : camera2body_poses) {
            std::vector<PointData> data(n_points_per_view_);
            if (!viewpoint_builder_(pose, data)) {
                return false
            }
            data_.emplace_back(Viewpoint<PointData>{pose.translation().matrix().normalized(), std::move(data)});
        }
        return true;
    }
    
    void AddViewpoint(const Viewpoint<PointData>& viewpoint) {
        data_.emplace_back(viewpoint);
    }

    void AddViewpoint(Viewpoint<PointData>&& viewpoint) {
        data_.emplace_back(std::move(viewpoint));
    }

    void ReplaceViewpoints(const std::vector<Viewpoint<PointData>>& data) {
        data_ = data
    }

    void ReplaceViewpoints(std::vector<Viewpoint<PointData>>&& data) {
        data_ = std::move(data)
    }
    
    const std::vector<Viewpoint<PointData>>& Data() const {
        return data_;
    }

private:
    std::vector<Viewpoint<PointData>> data_;
    int n_divides_;
    float radius_;
    int n_points_per_view_;
};

}

#endif