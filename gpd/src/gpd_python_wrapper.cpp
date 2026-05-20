// Python ctypes용 C API 래퍼.
// camera_source 배열을 직접 받아 멀티뷰 normal 계산을 정확하게 수행한다.
//
// 반환 포맷 (grasp 1개당 14개 double):
//   [score, px, py, pz, app_x, app_y, app_z, bin_x, bin_y, bin_z, ax_x, ax_y, ax_z, width]

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gpd/candidate/hand.h>
#include <gpd/grasp_detector.h>
#include <gpd/util/cloud.h>

using PointCloudRGB = pcl::PointCloud<pcl::PointXYZRGBA>;

static Eigen::Matrix3Xd to_view_points(const float* arr, int n) {
    Eigen::Matrix3Xd mat(3, n);
    for (int i = 0; i < n; i++)
        mat.col(i) << arr[3*i], arr[3*i+1], arr[3*i+2];
    return mat;
}

// camera_index[i] = 카메라 번호(0 or 1) → (num_cameras x num_points) 이진 행렬
static Eigen::MatrixXi to_camera_source(const int* camera_index, int num_cameras, int num_points) {
    Eigen::MatrixXi mat = Eigen::MatrixXi::Zero(num_cameras, num_points);
    for (int i = 0; i < num_points; i++) {
        int cam = camera_index[i];
        if (cam >= 0 && cam < num_cameras)
            mat(cam, i) = 1;
    }
    return mat;
}

extern "C" {

// 멀티뷰 grasp 검출.
// 반환값: 힙에 할당된 double 배열 (grasp 1개당 14 double), free_grasp_data()로 해제 필요.
// 반환값이 nullptr이면 grasp 없음.
double* detect_grasps_multi_view(
    const char* config_path,
    const float* points,       // [x0,y0,z0, x1,y1,z1, ...], 길이 = num_points * 3
    const int*   camera_index, // [0,0,...,1,1,...], 길이 = num_points
    const float* view_points,  // [cx0,cy0,cz0, cx1,cy1,cz1, ...], 길이 = num_cameras * 3
    int num_points,
    int num_cameras,
    int* num_grasps_out
) {
    *num_grasps_out = 0;

    PointCloudRGB::Ptr cloud(new PointCloudRGB);
    cloud->resize(num_points);
    for (int i = 0; i < num_points; i++) {
        cloud->at(i).x = points[3*i];
        cloud->at(i).y = points[3*i+1];
        cloud->at(i).z = points[3*i+2];
    }

    Eigen::Matrix3Xd vp = to_view_points(view_points, num_cameras);
    Eigen::MatrixXi  cs = to_camera_source(camera_index, num_cameras, num_points);

    gpd::util::Cloud gpd_cloud(cloud, cs, vp);

    gpd::GraspDetector detector(config_path);
    detector.preprocessPointCloud(gpd_cloud);

    auto grasps = detector.detectGrasps(gpd_cloud);

    int n = static_cast<int>(grasps.size());
    *num_grasps_out = n;
    if (n == 0) return nullptr;

    double* result = new double[n * 14];
    for (int i = 0; i < n; i++) {
        double* g    = result + i * 14;
        Eigen::Vector3d pos = grasps[i]->getPosition();
        Eigen::Vector3d app = grasps[i]->getApproach();
        Eigen::Vector3d bin = grasps[i]->getBinormal();
        Eigen::Vector3d ax  = grasps[i]->getAxis();
        g[0]  = grasps[i]->getScore();
        g[1]  = pos(0); g[2]  = pos(1); g[3]  = pos(2);
        g[4]  = app(0); g[5]  = app(1); g[6]  = app(2);
        g[7]  = bin(0); g[8]  = bin(1); g[9]  = bin(2);
        g[10] = ax(0);  g[11] = ax(1);  g[12] = ax(2);
        g[13] = grasps[i]->getGraspWidth();
    }
    return result;
}

void free_grasp_data(double* ptr) {
    delete[] ptr;
}

}  // extern "C"
