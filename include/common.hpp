#ifndef MB3T_COMMON_HPP
#define MB3T_COMMON_HPP

#include <Eigen/Dense>

namespace mb3t {

    // Commonly used types
    typedef Eigen::Transform<float, 3, Eigen::Affine> Transform3fA;

    // Commonly used constants
    constexpr float kPi = 3.1415926535897f;

    // Commonly used structs
    

    // Commonly used mathematical functions
    template <typename T>
    inline int sgn(T value) {
        return (value > T(0)) - (value < T(0));
    }

    template <typename T>
    inline T square(T value) {
        return value * value;
    }
}

#endif