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

}
}
