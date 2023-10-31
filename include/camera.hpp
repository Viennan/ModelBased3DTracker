#ifndef MB3T_RENDER_HPP
#define MB3T_RENDER_HPP

#include <functional>
#include <opencv2/core.hpp>
#include "common.hpp"

namespace mb3t {

struct Intrinsics {
    float fu, fv;
    float ppu, ppv;
    int width, height;
};

inline Eigen::Vector2f PinHoleProject(const Intrinsics &intrinsics, const Eigen::Vector3f &point) {
    return Eigen::Vector2f{intrinsics.fu * point.x() / point.z() + intrinsics.ppu, intrinsics.fv * point.y() / point.z() + intrinsics.ppv};
}

inline Eigen::Vector3f ReversePinHoleProject(const Intrinsics &intrinsics, float depth, float x, float y) {
    return Eigen::Vector3f{depth * (x - intrinsics.ppu) / intrinsics.fu, depth * (y - intrinsics.ppv) / intrinsics.fv, depth};
}

struct Camera {
    std::function<void(const Transform3fA& body2camera)> Capture;
    std::function<cv::Mat()> Depth;
    std::function<cv::Mat()> Normal;
    std::function<cv::Mat()> Color;
    std::function<const Intrinsics&()> Intrinsics;
    std::function<std::tuple<Transform3fA, Transform3fA>()> ModelViewMatrixes;
};

}

#endif