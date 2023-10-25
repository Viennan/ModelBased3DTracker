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
    for (int i=0;i<height;++i) {
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



}
}