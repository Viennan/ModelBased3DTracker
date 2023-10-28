#ifndef MB3T_HIST_MODEL_HPP
#define MB3T_HIST_MODEL_HPP

#include <opencv2/core.hpp>
#include "camera.hpp"
#include "hist_modality.hpp"

namespace mb3t {
namespace hm {

    class ViewpointBuilder { 
    public:

        bool GeneratePointData(const Transform3fA& body2camera, std::vector<ContourPoint>& points) const;

    private:
        int seed;
        float max_radius_depth_offset;
        float stride_depth_offset;
        Camera camera;
    };


}
}

#endif