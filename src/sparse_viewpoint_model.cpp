#include "sparse_viewpoint_model.hpp"

#include <set>

namespace mb3t {

struct CompareSmallerVector3f {
    bool operator()(const Eigen::Vector3f &v1,
                    const Eigen::Vector3f &v2) const {
    return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]) ||
            (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
    }
};

static void subdivideTriangle(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, 
    const Eigen::Vector3f &v3, int n_divides, std::set<Eigen::Vector3f, CompareSmallerVector3f>& geodesic_points) 
{
    if (n_divides == 0) {
        geodesic_points.insert(v1);
        geodesic_points.insert(v2);
        geodesic_points.insert(v3);
    } else {
        Eigen::Vector3f v12 = (v1 + v2).normalized();
        Eigen::Vector3f v13 = (v1 + v3).normalized();
        Eigen::Vector3f v23 = (v2 + v3).normalized();
        subdivideTriangle(v1, v12, v13, n_divides - 1, geodesic_points);
        subdivideTriangle(v2, v12, v23, n_divides - 1, geodesic_points);
        subdivideTriangle(v3, v13, v23, n_divides - 1, geodesic_points);
        subdivideTriangle(v12, v13, v23, n_divides - 1, geodesic_points);
    }
}

static void generateGeodesicPoints(std::set<Eigen::Vector3f, CompareSmallerVector3f>& geodesic_points, int n_divides) {
    // Define icosahedron
    constexpr float x = 0.525731112119133606f;
    constexpr float z = 0.850650808352039932f;
    std::vector<Eigen::Vector3f> icosahedron_points{
        {-x, 0.0f, z}, {x, 0.0f, z},  {-x, 0.0f, -z}, {x, 0.0f, -z},
        {0.0f, z, x},  {0.0f, z, -x}, {0.0f, -z, x},  {0.0f, -z, -x},
        {z, x, 0.0f},  {-z, x, 0.0f}, {z, -x, 0.0f},  {-z, -x, 0.0f}};
    std::vector<std::array<int, 3>> icosahedron_ids{
        {0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
        {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
        {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
        {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}};

    // Create points
    geodesic_points.clear();
    for (const auto &icosahedron_id : icosahedron_ids) {
        subdivideTriangle(icosahedron_points[icosahedron_id[0]],
                        icosahedron_points[icosahedron_id[1]],
                        icosahedron_points[icosahedron_id[2]], n_divides,
                        geodesic_points);
    }
}

void GenerateGeodesicPoses(int n_divides, float radius, std::vector<Transform3fA>& camera2body_poses) {
    // Generate geodesic points
    std::set<Eigen::Vector3f, CompareSmallerVector3f> geodesic_points;
    generateGeodesicPoints(geodesic_points, n_divides);

    // Generate geodesic poses from points
    Eigen::Vector3f downwards{0.0f, 1.0f, 0.0f};  // direction in body frame, alse the polar axis
    camera2body_poses.clear();
    for (const auto &geodesic_point : geodesic_points) {
        Transform3fA pose;
        pose = Eigen::Translation<float, 3>{geodesic_point * radius};

        Eigen::Matrix3f Rotation;
        Rotation.col(2) = -geodesic_point;
        if (geodesic_point[0] == 0.0f && geodesic_point[2] == 0.0f) {
            Rotation.col(0) = Eigen::Vector3f(1, 0, 0);
        } else {
            Rotation.col(0) = downwards.cross(-geodesic_point).normalized();
        }
        Rotation.col(1) = Rotation.col(2).cross(Rotation.col(0));
        pose.rotate(Rotation);
        camera2body_poses.push_back(pose);
    }
}

void CalculateDepthOffsets(
    const cv::Mat& depth_image, 
    const cv::Point2i &center,
    float pixel_to_meter,
    float max_radius_depth_offset,
    float stride_depth_offset,
    DepthOffsets& depth_offsets) 
{
    // Precalculate variables in pixel coordinates
    int n_values = int(max_radius_depth_offset / stride_depth_offset + 1.0f);
    float stride = stride_depth_offset / pixel_to_meter;
    float max_diameter = 2.0f * n_values * stride;

    // Precalculate rounded variables to iterate over image
    int image_stride = int(stride + 1.0f);
    int n_image_strides = int(max_diameter / image_stride + 1.0f);
    int image_diameter = n_image_strides * image_stride;
    int image_radius_minus = image_diameter / 2;
    int image_radius_plus = image_diameter - image_radius_minus;

    // Calculate limits for iteration
    int v_min = std::max(center.y - image_radius_minus, 0);
    int v_max = std::min(center.y + image_radius_plus, depth_image.rows - 1);
    int u_min = std::max(center.x - image_radius_minus, 0);
    int u_max = std::min(center.x + image_radius_plus, depth_image.cols - 1);

    // Iterate image to find minimum values corresponding to a certain radius
    int v, u, i;
    float distance;
    const float *ptr_image;
    DepthOffsets min_abs_depths;
    min_abs_depths.fill(std::numeric_limits<float>::max());
    min_abs_depths[0] = depth_image.at<float>(center);
    for (v = v_min; v <= v_max; v += image_stride) {
        ptr_image = depth_image.ptr<float>(v);
        for (u = u_min; u <= u_max; u += image_stride) {
            distance = std::sqrt(square(u - center.x) + square(v - center.y));
            i = int(distance / stride);
            if (i < n_values) min_abs_depths[i] = std::min(min_abs_depths[i], depth_image.at<float>(v, u));
        }
    }

    // Accumulate minimum values for circular regions and calculate depth offset
    float depth_center = depth_image.at<float>(center);
    depth_offsets[0] = depth_center - min_abs_depths[0];
    for (size_t i = 1; i < kMaxNDepthOffsets; ++i) {
        min_abs_depths[i] = std::min(min_abs_depths[i], min_abs_depths[i - 1]);
        depth_offsets[i] = depth_center - min_abs_depths[i];
    }
}

}