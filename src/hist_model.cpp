#include <random>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include "hist_model.hpp"

namespace mb3t {
namespace hm {

// Fixed parameters
static constexpr int kContourNormalApproxRadius = 3;
static constexpr int kMinContourLength = 15;

static bool generateValidContours(const cv::Mat &silhouette, std::vector<std::vector<cv::Point2i>> &contours, int& total_contour_length_in_pixel) {
    // test if outer border is empty
    int width = silhouette.cols;
    int height = silhouette.rows;
    for (int i=0;i<width;++i) {
        if (silhouette.at<uchar>(0, i) || silhouette.at<uchar>(height-1, i)) {
            CV_LOG_ERROR(nullptr, "Silhouette has non-empty outer border");
            return false;
        }
    }
    for (int i=1;i<height-1;++i) {
        if (silhouette.at<uchar>(i, 0) || silhouette.at<uchar>(i, width-1)) {
            CV_LOG_ERROR(nullptr, "Silhouette has non-empty outer border");
            return false;
        }
    }

    // Compute contours
    cv::findContours(silhouette, contours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

    // Filter contours that are too short
    contours.erase(std::remove_if(begin(contours), end(contours),
                    [](const std::vector<cv::Point2i> &contour) {
                    return contour.size() < kMinContourLength;
                    }),
                    end(contours));

    // Test if contours are closed
    for (auto &contour : contours) {
        if (abs(contour.front().x - contour.back().x) > 1 ||
            abs(contour.front().y - contour.back().y) > 1) {
            CV_LOG_ERROR(nullptr, "Contours are not closed");
            return false;
        }
    }

    // Calculate total pixel length of contour
    total_contour_length_in_pixel = 0;
    for (auto &contour : contours) {
        total_contour_length_in_pixel += int(contour.size());
    }

    // Check if pixel length is greater zero
    if (total_contour_length_in_pixel == 0) {
        CV_LOG_ERROR(nullptr, "No valid contour in image");
        return false;
    }
    return true;
}

static cv::Point2i sampleContourPointCoordinate(
    const std::vector<std::vector<cv::Point2i>> &contours, 
    int total_contour_length_in_pixel, 
    std::mt19937 &generator) 
{
    int idx = int(generator() % total_contour_length_in_pixel);
    for (auto &contour : contours) {
    if (idx < contour.size())
        return contour[idx];
    else
        idx -= int(contour.size());
    }
    return cv::Point2i();  // Never reached
}

static bool calculateContourSegment(
    const std::vector<std::vector<cv::Point2i>> &contours, cv::Point2i &center,
    std::vector<cv::Point2i> &contour_segment) {
    for (auto &contour : contours) {
        if (contour.size() < kContourNormalApproxRadius + 1) {
            continue;
        }
        for (int idx = 0; idx < contour.size(); ++idx) {
            if (contour.at(idx) == center) {
                int start_idx = idx - kContourNormalApproxRadius;
                int end_idx = idx + kContourNormalApproxRadius;
                if (start_idx < 0) {
                    contour_segment.insert(end(contour_segment), end(contour) + start_idx, end(contour));
                    start_idx = 0;
                }
                if (end_idx >= int(contour.size())) {
                    contour_segment.insert(end(contour_segment), begin(contour) + start_idx, end(contour));
                    start_idx = 0;
                    end_idx = end_idx - int(contour.size());
                }
                contour_segment.insert(end(contour_segment), begin(contour) + start_idx, begin(contour) + end_idx + 1);

                // Check quality of contour segment
                float segment_distance = std::hypotf(
                    float(contour_segment.back().x - contour_segment.front().x),
                    float(contour_segment.back().y - contour_segment.front().y));
                return segment_distance > float(kContourNormalApproxRadius);
            }
        }
    }
    CV_LOG_ERROR(nullptr, "Could not find point on contour");
    return false;
}

static inline Eigen::Vector2f approximateNormalVector(const std::vector<cv::Point2i> &contour_segment) {
    return Eigen::Vector2f{
        -float(contour_segment.back().y - contour_segment.front().y),
        float(contour_segment.back().x - contour_segment.front().x)
    }.normalized();
}

void findClosestContourPoint(
    const std::vector<std::vector<cv::Point2i>> &contours, float u, float v,
    int& u_contour, int& v_contour) 
{
    float min_distance = std::numeric_limits<float>::max();
    for (auto &contour : contours) {
        for (auto &point : contour) {
            float distance = hypotf(float(point.x) - u, float(point.y) - v);
            if (distance < min_distance) {
                u_contour = point.x;
                v_contour = point.y;
                min_distance = distance;
            }
        }
    }
}

static void calculateLineDistances(
    const cv::Mat &silhouette,
    const std::vector<std::vector<cv::Point2i>> &contours,
    const cv::Point2i &center, const Eigen::Vector2f &normal,
    float pixel_to_meter, float &foreground_distance,
    float &background_distance) {
    // Calculate starting positions and steps for both sides of the line
    float u_out = float(center.x) + 0.5f;
    float v_out = float(center.y) + 0.5f;
    float u_in = float(center.x) + 0.5f;
    float v_in = float(center.y) + 0.5f;
    float u_step, v_step;
    if (std::fabs(normal.y()) < std::fabs(normal.x())) {
        u_step = float(sgn(normal.x()));
        v_step = normal.y() / abs(normal.x());
    } else {
        u_step = normal.x() / abs(normal.y());
        v_step = float(sgn(normal.y()));
    }

    // Search for first inwards intersection with contour
    int u_in_endpoint, v_in_endpoint;
    while (true) {
        u_in -= u_step;
        v_in -= v_step;
        if (!silhouette.at<uchar>(int(v_in), int(u_in))) {
            findClosestContourPoint(contours, u_in + u_step - 0.5f, v_in + v_step - 0.5f, u_in_endpoint, v_in_endpoint);
            foreground_distance = pixel_to_meter * hypotf(float(u_in_endpoint - center.x), float(v_in_endpoint - center.y));
            break;
        }
    }

    // Search for first outwards intersection with contour
    int w = silhouette.cols;
    int h = silhouette.rows;
    int u_out_endpoint, v_out_endpoint;
    while (true) {
        u_out += u_step;
        v_out += v_step;
        if (int(u_out) < 0 || int(u_out) >= w || int(v_out) < 0 || int(v_out) >= h) {
            background_distance = std::numeric_limits<float>::max();
            break;
        }
        if (silhouette.at<uchar>(int(v_out), int(u_out))) {
            findClosestContourPoint(contours, u_out - 0.5f, v_out - 0.5f, u_out_endpoint, v_out_endpoint);
            background_distance =  pixel_to_meter * hypotf(float(u_out_endpoint - center.x), float(v_out_endpoint - center.y));
            break;
        }
    }
}

bool ViewpointBuilder::GeneratePointData(const Transform3fA& body2camera, std::vector<ContourPoint>& points) const {
    // Capture image
    camera.Capture(body2camera);

    // Generate contour
    int total_contour_length_in_pixel;
    std::vector<std::vector<cv::Point2i>> contours;
    cv::Mat silhouette = camera.Color();
    if (!generateValidContours(silhouette, contours, total_contour_length_in_pixel))
        return false;

    // Calculate data for contour points
    std::mt19937 generator{this->seed};
    auto intrinsics = camera.Intrinsics();
    auto depth = camera.Depth();
    auto camera2body = body2camera.inverse();
    // Calculate data for contour points
    for (auto data_point{begin(points)}; data_point != end(points);) {
        // Randomly sample point on contour and calculate 3D center
        cv::Point2i center{sampleContourPointCoordinate(contours, total_contour_length_in_pixel, generator)};
        auto center_f_camera = ReversePinHoleProject(intrinsics, depth.at<float>(center.y, center.x), float(center.x), float(center.y));
        data_point->center_f_body = camera2body * center_f_camera;

        // Calculate contour segment and approximate normal vector
        std::vector<cv::Point2i> contour_segment;
        if (!calculateContourSegment(contours, center, contour_segment)) 
            continue;
        Eigen::Vector2f normal{approximateNormalVector(contour_segment)};
        Eigen::Vector3f normal_f_camera{normal.x(), normal.y(), 0.0f};
        data_point->normal_f_body = camera2body.rotation() * normal_f_camera;

        // Calculate foreground and background distance
        float pixel_to_meter = center_f_camera(2) / intrinsics.fu;
        calculateLineDistances(silhouette, contours, center, normal, pixel_to_meter,
                                data_point->foreground_distance,
                                data_point->background_distance);
        
        // Calculate depth offsets
        CalculateDepthOffsets(depth, center, pixel_to_meter, max_radius_depth_offset, stride_depth_offset, data_point->depth_offsets);

        data_point++;
    }
    return true;
}

}
}
