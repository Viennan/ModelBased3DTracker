#include <numeric>
#include "hist_modality.hpp"

namespace mb3t {
namespace hm{

    void Histogram::Normalize() {
        float sum = 0.0;
#ifndef _DEBUG
#pragma omp simd
#endif
        for(int i = 0; i < hist_n_bins_cube_; ++i) {
            sum += hist_[i];
        }

        if (sum <= 0.0f) {
            return;
        }

        float recip = 1.0f / sum;
#ifndef _DEBUG
#pragma omp simd
#endif
        for(int i=0; i < hist_n_bins_cube_; ++i) {
            hist_[i] *= recip;
        }
    }

    void Histogram::MergeUnnormalized(const Histogram& temp, float learning_rate) {
        if (learning_rate == 1.0f) {
            *this = temp;
            this->Normalize();
            return;
        }

        float sum = 0.0f;
#ifndef _DEBUG
#pragma omp simd
#endif
        for(int i = 0; i < hist_n_bins_cube_; ++i) {
            sum += hist_[i];
        }

        if (sum <= 0.0f) {
            return;
        }

        float complement_learning_rate = 1.0f - learning_rate;
        float learning_rate_divide_sum = learning_rate / sum;
#ifndef _DEBUG
#pragma omp simd
#endif
        for (int i = 0; i < hist_n_bins_cube_; ++i) {
            hist_[i] = complement_learning_rate * hist_[i] + learning_rate_divide_sum * temp.hist_[i];
        }
    }
    
    void HistModality::ssf_init() {
        ssf_lookup_f_.resize(ssf_length);
        ssf_lookup_b_.resize(ssf_length);
        for (int i = 0; i < ssf_length; ++i) {
            float x = static_cast<float>(i) - static_cast<float>(ssf_length - 1) / 2.0f;
            if (ssf_slope == 0.0f) {
                // why not (x < 0.0f? -1 : 1) ? answer: we want x==0.0f to be jsut 0.5f
                ssf_lookup_f_[i] = 0.5f - ssf_amptitue * ((0.0f < x) - (x < 0.0f)); 
            } else {
                ssf_lookup_f_[i] = 0.5f - ssf_amptitue * std::tanh(x / (2.0f * ssf_slope));
            }
            ssf_lookup_b_[i] = 1.0f - ssf_lookup_f_[i];
        }
    }

    void HistModality::camera_update_params() {
        auto[m, v] = camera.ModelViewMatrixes();
        body2camera_ = v * m;
        body2camera_rotation_ = body2camera_.rotation();
        body2camera_rotation_xy_ = body2camera_rotation_.block<2, 3>(0, 0);
        intrinsics_ = camera.Intrinsics();
    }

    void HistModality::hist_setup() {
        hist_temp_f_.Setup(hist_n_bins);
        hist_temp_b_.Setup(hist_n_bins);
        hist_f_.Setup(hist_n_bins);
        hist_b_.Setup(hist_n_bins);
    }

    void HistModality::hist_update_temp(const Viewpoint<ContourPoint>& points) {
        auto image = camera.Color(); // the width and height of image will be (width_, height_) which is the same as camera intrinsics
        // Iterate over n_lines
        hist_temp_f_.Init();
        hist_temp_b_.Init();
        size_t n_points = points.data.size();
        int width_minus_1 = image.cols - 1;
        int height_minus_1 = image.rows - 1;
        for (size_t i=0;i<n_points;++i) {
            const auto& point = points.data[i];
            const auto& line = lines_[i];

            // Calculate considered line lengths
            float length_f = std::fmax(point.foreground_distance * intrinsics_.fu / line.center_f_camera.z() - hist_unconsidered_line_length, 0.0f);
            length_f = std::fmin(length_f, hist_max_considered_line_length);
            float length_b = std::fmax(point.background_distance * intrinsics_.fu / line.center_f_camera.z() - hist_unconsidered_line_length, 0.0f);
            length_b = std::fmin(length_b, hist_max_considered_line_length);

            // Define steps and projected considered line lengths
            float u_step, v_step;
            int projected_length_f, projected_length_b;
            float abs_normal_u = std::fabs(line.normal_uv.x());
            float abs_normal_v = std::fabs(line.normal_uv.y());
            if (abs_normal_u > abs_normal_v) {
                u_step = float(sgn(line.normal_uv.x()));
                v_step = line.normal_uv.y() / abs_normal_u;
                projected_length_f = int(length_f * abs_normal_u + 0.5f);
                projected_length_b = int(length_b * abs_normal_u + 0.5f);
            } else {
                u_step = line.normal_uv.x() / abs_normal_v;
                v_step = float(line.normal_uv.y());
                projected_length_f = int(length_f * abs_normal_v + 0.5f);
                projected_length_b = int(length_b * abs_normal_v + 0.5f);
            }

            // Iterate over foreground pixels
            float u = line.center_uv.x() - line.normal_uv.x() * hist_unconsidered_line_length + 0.5f;
            float v = line.center_uv.y() - line.normal_uv.y() * hist_unconsidered_line_length + 0.5f;
            int i_u, i_v;
            for (int i = 0; i < projected_length_f; ++i) {
                i_u = int(u);
                i_v = int(v);
                if (i_u < 0 || i_u > width_minus_1 || i_v < 0 || i_v > height_minus_1)
                    break;
                hist_temp_f_.Incr(image.at<cv::Vec3b>(i_v, i_u));
                u -= u_step;
                v -= v_step;
            }

            // Iterate over background pixels
            u = line.center_uv.x() + line.normal_uv.x() * hist_unconsidered_line_length + 0.5f;
            v = line.center_uv.y() + line.normal_uv.y() * hist_unconsidered_line_length + 0.5f;
            for (int i = 0; i < projected_length_b; ++i) {
                i_u = int(u);
                i_v = int(v);
                if (i_u < 0 || i_u > width_minus_1 || i_v < 0 || i_v > height_minus_1)
                    break;
                hist_temp_b_.Incr(image.at<cv::Vec3b>(i_v, i_u));
                u += u_step;
                v += v_step;
            }
        }
    }

    void HistModality::line_init_distribution_params() {
        line_distribution_length_minus_1_half_ = float(line_distribution_length - 1) * 0.5f;
        line_length_in_segment_ = ssf_length + line_distribution_length - 1;
        line_distribution_length_plus_1_half_ = float(line_distribution_length + 1) * 0.5f;
        auto min_expected_variance_laplace = 1.0f / (2.0f * powf(std::atanhf(2.0f * ssf_amptitue), 2.0f));
        auto min_expected_variance_gaussian = ssf_slope;
        min_expected_variance_ = std::max(min_expected_variance_laplace, min_expected_variance_gaussian);
    }

    void HistModality::line_init_scale_dependent_params(int scale_idx) {  
        line_scale_ = last_valid(line_scales, scale_idx);
        line_fscale_ = float(line_scale_);
        line_length_ = line_scale_ * line_length_in_segment_;
        line_length_minus_1_ = line_length_ - 1;
        line_length_minus_1_half_ = float(line_length_ - 1) * 0.5f;
    }

    void HistModality::line_search_and_project_centers() {
        // get closest viewpoint from sparse viewpoint model
        const auto& vp = viewpoint_model.GetClosestViewpoint(body2camera_);

        // filter out outliers, usually used for occlusion handling
        int valid_num = vp.data.size();
        std::vector<int> mask(vp.data.size(), 1);
        if (line_point_filters.size() > 0) {
            for (const auto& filter : line_point_filters) {
                filter(vp, mask);
            }
            valid_num = std::accumulate(mask.cbegin(), mask.cend(), 0);
        }

        // project centers and 2D normals of correspondence lines
        lines_.resize(valid_num);
        for (int i = 0, j = 0; i < vp.data.size(); ++i) {
            if (mask[i] == 0) {
                continue;
            }
            const auto& point = vp.data[i];
            auto& line = lines_[j];
            auto center_f_camera = body2camera_ * point.center_f_body;
            auto normal_f_camera = (body2camera_rotation_xy_ * point.normal_f_body).normalized();
            line.center_f_camera = center_f_camera;
            line.center_f_body = point.center_f_body;
            line.center_uv = PinHoleProject(intrinsics_, center_f_camera);
            line.normal_uv = Eigen::Vector2f(normal_f_camera.x(), normal_f_camera.y());
            ++j;
        }
    }

    bool HistModality::line_calculate_segment_color_distribution(const cv::Mat& image, Line& line, std::vector<float>& segment_f_distribution, std::vector<float>& segment_b_distribution) {
        // init some iteration params
        Eigen::Vector2f uv_start, uv_step;
        float *seg_f_ptr, *seg_b_ptr;
        // we init the beyond params for always iterating along the v axis's positive direction, which means v coordinate is none-decreasing when iterating.
        if (std::fabs(line.normal_uv.x()) > std::fabs(line.normal_uv.y())) {
            // this branch has included the case that normal_uv.y() == 0.0f
            auto v_step = line.normal_uv.y() / line.normal_uv.x();
            auto v_step_sgn = sgn(v_step);
            if (v_step_sgn == 0.0f) {
                v_step_sgn = 1.0f;
            }
            uv_step = Eigen::Vector2f(1.0f, v_step) * v_step_sgn;
            uv_start = line.center_uv - uv_step * line_length_minus_1_half_;
            line.normal_component_to_scale = std::fabs(line.normal_uv.x()) / line_fscale_;
            line.delta_r = (std::round(line.center_uv.x() - line_length_minus_1_half_) + line_length_minus_1_half_ - line.center_uv.x()) / line.normal_uv.x();
        } else {
            // if normal_uv is normalized, then normal_uv.y() is always non-zero in this branch.
            uv_step = Eigen::Vector2f(line.normal_uv.x() / line.normal_uv.y(), 1.0f);
            uv_start = line.center_uv - uv_step * line_length_minus_1_half_;
            line.normal_component_to_scale = std::fabs(line.normal_uv.y()) / line_fscale_;
            line.delta_r = (std::round(line.center_uv.y() - line_length_minus_1_half_) + line_length_minus_1_half_ - line.center_uv.y()) / line.normal_uv.y();
        }
        float v_sgn = sgn(line.normal_uv.y());
        if (v_sgn > 0.0f || (v_sgn == 0.0f && uv_step.x() > 0.0f)) {
            seg_f_ptr = &segment_f_distribution.front();
            seg_b_ptr = &segment_b_distribution.front(); 
        } else {
            seg_f_ptr = &segment_f_distribution.back();
            seg_b_ptr = &segment_b_distribution.back();
        }

        Eigen::Vector2f uv_start_for_round = uv_start + Eigen::Vector2f(0.5f, 0.5f); // add 0.5f for rounding when forced to int type
        // check start point
        if (uv_start_for_round.x() < 0 || int(uv_start_for_round.x()) > image.cols - 1 || uv_start_for_round.y() < 0) {
            return false;
        }
        // check end point
        Eigen::Vector2f uv_end_for_round = uv_start_for_round + uv_step * line_length_minus_1_;
        if (uv_end_for_round.x() < 0 || int(uv_end_for_round.x()) > image.cols - 1 || uv_end_for_round.y() > image.rows - 1) {
            return false;
        }

        // iterate over pixels along the line
        auto uv_for_round = uv_start_for_round;
        for (int i = 0; i < line_length_; i += line_scale_) {
            float f_p = 1.0f, b_p = 1.0f;
            for (int j = 0; j < line_scale_; j++, uv_for_round += uv_step) {
                int u = int(uv_for_round.x());
                int v = int(uv_for_round.y());
                const auto& color = image.at<cv::Vec3b>(v, u);
                float f_p_j = hist_f_(color);
                float b_p_j = hist_b_(color);
                float p_j = f_p_j + b_p_j;
                if (p_j > 0.0f) {
                    f_p_j /= p_j;
                    b_p_j /= p_j;
                } else {
                    f_p_j = 0.5f;
                    b_p_j = 0.5f;
                }
                f_p *= f_p_j;
                b_p *= b_p_j;
            }
            *seg_f_ptr++ = f_p;
            *seg_b_ptr++ = b_p;
        }

        // normalize distribution between foreground and background when scale_ > 1
        if (line_scale_ > 1) {
            for (auto it_f = segment_f_distribution.begin(), it_b = segment_b_distribution.begin(); it_f != segment_f_distribution.end(); ++it_f, ++it_b) {
                float sum = *it_f + *it_b;
                if (sum > 0.0f) {
                    *it_f /= sum;
                    *it_b /= sum;
                } else {
                    *it_f = 0.5f;
                    *it_b = 0.5f;
                }
            }
        }

        return true;
    }

    void HistModality::line_calculate_segment_distribution(Line& line, const std::vector<float>& seg_f_distribution, const std::vector<float>& seg_b_distribution) {
        // Loop over entire distribution and start values of segment probabilities
        float sum = 0.0f;
        for (int i=0; i<line_distribution_length; ++i) {
            float p = 1.0f;
            for (int j=0; j<ssf_length; ++j) {
                p *= (seg_f_distribution[i+j] * ssf_lookup_f_[j] + seg_b_distribution[i+j] * ssf_lookup_b_[j]);
            }
            line.distribution[i] = p;
            sum += p;
        }
        // normalize distribution
        float recip = 1.0f / sum;
        for (auto& p : line.distribution) {
            p *= recip;
        }
        // calculate mean and variance
        float mean_from_begin = 0.0f;
        for (int i=0; i<line_distribution_length; ++i) {
            mean_from_begin += line.distribution[i] * float(i);
        }
        float variance_from_begin = 0.0f;
        for (int i=0; i<line_distribution_length; ++i) {
            float diff = float(i) - mean_from_begin;
            variance_from_begin += line.distribution[i] * diff * diff;
        }
        line.mean = mean_from_begin - line_distribution_length_minus_1_half_;
        line.variance = std::max(variance_from_begin, min_expected_variance_);
        line.reciprocal_variance = 1.0f / line.variance;
    }

    void HistModality::line_calculate_correspondence(int scale_idx) {
        camera_update_params();
        line_init_scale_dependent_params(scale_idx);
        line_search_and_project_centers();
        std::vector<float> segment_f_distribution(line_length_in_segment_);
        std::vector<float> segment_b_distribution(line_length_in_segment_);
        auto image = camera.Color();
        std::vector<Line> lines_temp;
        std::vector<int> mask;
        for (size_t i=0; i<lines_.size(); ++i) {
            auto& line = lines_[i];
            if (!line_calculate_segment_color_distribution(image, line, segment_f_distribution, segment_b_distribution)) {
                mask.push_back(i);
                continue;
            }
            line_calculate_segment_distribution(line, segment_f_distribution, segment_b_distribution);
        }
        // remove invalid lines. This situation may not happen very often, so we explicitly check and handle it out of the beyond loop.
        if (mask.size() > 0) {
            lines_temp.reserve(lines_.size() - mask.size());
            for (size_t i=0, j=0; i<lines_.size(); ++i) {
                if (i == mask[j]) {
                    ++j;
                    continue;
                }
                lines_temp[i-j] = std::move(lines_[i]);
            }
            lines_ = std::move(lines_temp);
        }
        // calculate normalized reciprocal variance for weighting
        float sum_reciprocal_variance = 0.0f;
        for (const auto& line : lines_) {
            sum_reciprocal_variance += line.reciprocal_variance;
        }
        float recip_sum_reciprocal_variance = 1.0f / sum_reciprocal_variance;
        line_normalized_reciprocal_variances_.resize(lines_.size());
        for (size_t i=0; i<lines_.size(); ++i) {
            line_normalized_reciprocal_variances_[i] = lines_[i].reciprocal_variance * recip_sum_reciprocal_variance;
        }
        // calculate average variance
        line_average_variance_ = std::accumulate(lines_.cbegin(), lines_.cend(), 0.0f, [](const Line& line) {
            return line.variance;
        }) / lines_.size();
    }

    void HistModality::optimize(int iteration) {
        camera_update_params();

        gradient_.setZero();
        hessian_.setZero();
        for (size_t i=0;i<lines_.size();++i) {
            const auto& line = lines_[i];
            auto center_f_camera = body2camera_ * line.center_f_body;
            float fu_z = intrinsics_.fu / center_f_camera.z();
            float fv_z = intrinsics_.fv / center_f_camera.z();
            float xfu_z = center_f_camera.x() * fu_z;
            float yfv_z = center_f_camera.y() * fv_z;
            auto center_uv = Eigen::Vector2f{xfu_z + intrinsics_.ppu, yfv_z + intrinsics_.ppv};
            // calculate delta_cs           
            float delta_cs = (line.normal_uv.x() * (xfu_z + intrinsics_.ppu - line.center_uv.x()) +
                              line.normal_uv.y() * (yfv_z + intrinsics_.ppv - line.center_uv.y()) - 
                              line.delta_r) * line.normal_component_to_scale;
            
            // calculate gradient of loglikihood with respect to delta_cs
            float dloglikelihood_ddelta_cs;
            if (iteration < opt_n_global_iteration) {
                // treat the distribution as a perfect gaussian distribution in global optimization
                dloglikelihood_ddelta_cs = (line.mean -delta_cs) * line.reciprocal_variance;
            } else {
                // local optimization
                int dist_idx_upper = int(delta_cs + line_distribution_length_plus_1_half_);
                int dist_idx_lower = dist_idx_upper - 1;
                if (dist_idx_lower < 0 || dist_idx_upper >= line_distribution_length) {
                    dloglikelihood_ddelta_cs = (line.mean - delta_cs) * line.reciprocal_variance;
                } else {
                    dloglikelihood_ddelta_cs = std::log(line.distribution[dist_idx_upper]) - std::log(line.distribution[dist_idx_lower]);
                }
            }

            // calculate jacobian matrix of delta_cs with respect to body pose variations in body frame.
            Eigen::RowVector3f ddelta_cs_dcenter{
                line.normal_component_to_scale * fu_z * line.normal_uv.x(),
                line.normal_component_to_scale * fv_z * line.normal_uv.y(),
                line.normal_component_to_scale * (-line.normal_uv.x() * xfu_z - line.normal_uv.y() * yfv_z) / center_f_camera.z()
            };
            Eigen::RowVector3f ddelta_cs_translation = ddelta_cs_dcenter * body2camera_rotation_;
            Eigen::Matrix<float, 1, 6> ddelta_cs_dtheta;
            ddelta_cs_dtheta << line.center_f_body.transpose().cross(ddelta_cs_translation), ddelta_cs_translation;

            // calculate gradient and hessian of loglikelihood with body pose variations in body frame.
            gradient_ += line_normalized_reciprocal_variances_[i] * dloglikelihood_ddelta_cs * ddelta_cs_dtheta.transpose();
            hessian_ += line_normalized_reciprocal_variances_[i] * line.reciprocal_variance * ddelta_cs_dtheta.transpose() * ddelta_cs_dtheta;
        }
        // caculate weight of the modality
        mod_variance_in_pixel_ = line_average_variance_ * line_scale_ * line_scale_;
    }

}
}
