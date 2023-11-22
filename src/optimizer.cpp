#include "optimizer.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <numeric>

namespace mb3t {

bool Optimizer::operator()(const Transform3fA& body2world, Transform3fA& body2world_next, bool& convergent) {
    // check if should calculate correspondence
    if (curr_iteration_ == 0) {
        for (const auto& modality : modalities) {
            if (!modality.CalculateCorrespondence(curr_scale_)) {
                return false;
            }
        }
    }

    std::vector<OptimizeResult> results;
    for (const auto& modality : modalities) {
        auto res = modality.Optimize(curr_iteration_);
        if (!res.success) {
            CV_LOG_WARNING(nullptr, "Optimize " + modality.Name + " failed");
            continue;
        }
        results.push_back(modality.Optimize(curr_iteration_));
    }
    if (results.empty()) {
        CV_LOG_ERROR(nullptr, "No modality optimization succeeded");
        return false;
    }

    std::vector<float> weights;
    for (const auto& result : results) {
        weights.push_back(1.0f / result.variance);
    }
    float sum_recip = 1.0f / std::accumulate(weights.begin(), weights.end(), 0.0f);
    for (auto& weight : weights) {
        weight *= sum_recip;
    }

    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 6> a = Eigen::Matrix<float, 6, 6>::Zero();
    for (int i = 0; i < results.size(); ++i) {
        b += weights[i] * results[i].gradient;
        a -= weights[i] * results[i].hessian;
    }
    a.diagonal().head<3>().array() += tikhonov_parameter_rotation;
    a.diagonal().tail<3>().array() += tikhonov_parameter_translation;
    Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{a};
    if (lu.isInvertible()) {
        Eigen::Matrix<float, 6, 1> delta = lu.solve(b);
        Transform3fA pose_variation{Transform3fA::Identity()};
        pose_variation.rotate(Vector2Skewsymmetric(delta.head<3>()).exp());
        pose_variation.translate(delta.tail<3>());
        body2world_next = body2world * pose_variation;
    } else {
        CV_LOG_WARNING(nullptr, "Hessian matrix is not invertible");
        body2world_next = body2world;
    }

    ++curr_iteration_;
    convergent = false;
    if (curr_iteration_ >= last_valid(iteration_num, curr_scale_)) {
        curr_iteration_ = 0;
        ++curr_scale_;
        if (curr_scale_ >= scale_num) {
            curr_scale_ = 0;
            convergent = true;
        }
    }
    return true;
}

void Optimizer::Reset() {
    curr_scale_ = 0;
    curr_iteration_ = 0;
}

}