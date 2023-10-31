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

    void HistModality::line_search_and_project_centers() {
        // get closest viewpoint from sparse viewpoint model
        const auto& vp = viewpoint_model.GetClosestViewpoint(body2camera_);

        // filter out outliers, usually used for occlusion handling
        int valid_num = vp.data.size();
        std::vector<int> mask(vp.data.size(), 1);
        if (point_filters.size() > 0) {
            for (const auto& filter : point_filters) {
                filter(vp, mask);
            }
            valid_num = std::count(mask.begin(), mask.end(), 1);
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

}
}
