#include <functional>
#include <opencv2/opencv.hpp>

namespace mb3t {

    struct Camera {
        std::function<cv::Mat()> image;
    };

    
}