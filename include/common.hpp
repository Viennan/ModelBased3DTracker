#ifndef MB3T_COMMON_HPP
#define MB3T_COMMON_HPP

#include <Eigen/Dense>

namespace mb3t {

    // Commonly used types
    typedef Eigen::Transform<float, 3, Eigen::Affine> Transform3fA;

    // Commonly used constants
    constexpr float kPi = 3.1415926535897f;

    // Commonly used structs
    
    // Commonly used helper functions
    template <typename T>
    inline auto last_valid(const std::vector<T>& l, size_t idx) {
        if (idx >= l.size()) {
            return l.back();
        }
        return l[idx];
    }

    // Commonly used mathematical functions
    template <typename T>
    inline int sgn(T value) {
        return (value > T(0)) - (value < T(0));
    }

    template <typename T>
    inline T square(T value) {
        return value * value;
    }

    inline Eigen::Matrix3f Vector2Skewsymmetric(const Eigen::Vector3f &vector) {
        Eigen::Matrix3f skew_symmetric;
        skew_symmetric << 0.0f, -vector(2), vector(1), vector(2), 0.0f, -vector(0),
            -vector(1), vector(0), 0.0f;
        return skew_symmetric;
    }
}

#endif