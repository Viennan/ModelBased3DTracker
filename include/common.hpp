#ifndef MB3T_COMMON_HPP
#define MB3T_COMMON_HPP

#include <Eigen/Dense>

namespace mb3t {

    // Commonly used types
    typedef Eigen::Transform<float, 3, Eigen::Affine> Transform3fA;

    // Commonly used constants
    constexpr float kPi = 3.1415926535897f;

    // Commonly used structs
    struct Intrinsics {
        float fu, fv;
        float ppu, ppv;
        int width, height;
    };
}

#endif