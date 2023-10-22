#include <vector>
#include <Eigen/Dense>

namespace mb3t {

template <typename T>
struct Viewpoint{
    std::vector<T> data;
    Eigen::Vector3f orientation;
}

template <typename T>
class SparseViewpointModel {
public:
    SparseViewpointModel() = default;
    SparseViewpointModel(const SparseViewpointModel&) = default;
    SparseViewpointModel(SparseViewpointModel&&) = default;
    SparseViewpointModel& operator=(const SparseViewpointModel&) = default;
    SparseViewpointModel& operator=(SparseViewpointModel&&) = default;
    ~SparseViewpointModel() = default;

    const Viewpoint<T>& GetClosestViewpoint(const Eigen::Transform3fA& body2camera_pose) const {
        // convert to camera to body orientation
        Eigen::Vector3f orientation{body2camera_pose.rotation().inverse() * body2camera_pose.translation().matrix().normalized()};
        float closest_dot = -1.0f;
        size_t index = 0;
        for (size_t i = 0; i < viewpoints_.size(); ++i) {
            float dot = orientation.dot(view.orientation);
            if (dot > closest_dot) {
                index = i;
            }
        }
        return viewpoints_[index];
    }

private:
    std::vector<Viewpoint<T>> viewpoints_;
};

}