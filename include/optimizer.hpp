#ifndef MB3T_OPTIMIZER_HPP
#define MB3T_OPTIMIZER_HPP

#include <string>
#include <functional>
#include <Eigen/Dense>
#include "camera.hpp"

namespace mb3t {

    struct OptimizeResult {
        Eigen::Matrix<float, 6, 1> gradient;
        Eigen::Matrix<float, 6, 6> hessian;
        float variance;
        bool success;
    };

    struct Modality {
        std::string Name;
        std::function<bool(int scale_idx)> CalculateCorrespondence;
        std::function<OptimizeResult(int iteration)> Optimize;
    };

    class Optimizer {
    public:

        bool operator()(const Transform3fA& src, Transform3fA& dst, bool& convergent);

        void Reset();

    private:
        std::vector<Modality> modalities;
        int scale_num;
        std::vector<int> iteration_num;
        float tikhonov_parameter_rotation;
        float tikhonov_parameter_translation;
        int curr_scale_;
        int curr_iteration_;
    };


}

#endif