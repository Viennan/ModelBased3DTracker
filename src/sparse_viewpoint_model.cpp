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

}