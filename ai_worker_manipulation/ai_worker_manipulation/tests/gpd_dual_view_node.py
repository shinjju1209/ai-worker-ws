#!/usr/bin/env python3
"""
Dual-view GPD grasp detection node.

팀원이 TF 변환 완료한 두 PointCloud2 토픽을 구독해서:
  1. 두 클라우드 합성 + 카메라 인덱스 추적
  2. libgpd_python_wrapper.so C API 호출 (멀티뷰 normal 계산)
  3. geometry_msgs/PoseArray 로 grasp pose 퍼블리시

Subscriptions:
  left_topic  (PointCloud2) : 왼팔 카메라, base_link 기준으로 변환 완료된 것
  right_topic (PointCloud2) : 오른팔 카메라, base_link 기준으로 변환 완료된 것

Publish:
  /gpd/grasp_poses (PoseArray, frame_id = base_link)
"""

import ctypes
import os

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import message_filters

_FIELDS_PER_GRASP = 14  # [score, px,py,pz, app_x,app_y,app_z, bin_x,bin_y,bin_z, ax_x,ax_y,ax_z, width]


# ---------------------------------------------------------------------------
# ctypes 라이브러리 로드
# ---------------------------------------------------------------------------

def _load_gpd_lib(gpd_dir: str) -> ctypes.CDLL:
    so_path = os.path.join(gpd_dir, 'build', 'libgpd_python_wrapper.so')
    lib = ctypes.CDLL(so_path)

    lib.detect_grasps_multi_view.restype  = ctypes.POINTER(ctypes.c_double)
    lib.detect_grasps_multi_view.argtypes = [
        ctypes.c_char_p,                        # config_path
        ctypes.POINTER(ctypes.c_float),          # points
        ctypes.POINTER(ctypes.c_int),            # camera_index
        ctypes.POINTER(ctypes.c_float),          # view_points
        ctypes.c_int,                            # num_points
        ctypes.c_int,                            # num_cameras
        ctypes.POINTER(ctypes.c_int),            # num_grasps_out
    ]

    lib.free_grasp_data.restype  = None
    lib.free_grasp_data.argtypes = [ctypes.POINTER(ctypes.c_double)]

    return lib


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float64, NaN 제거."""
    pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def merge_with_index(pts_l: np.ndarray, pts_r: np.ndarray):
    """두 클라우드를 합치고 카메라 인덱스 배열을 함께 반환.

    outlier 제거는 eigen_params.cfg 의 remove_outliers=1 로 GPD 내부에서 처리.

    Returns:
        all_pts   : (N, 3) float32
        cam_index : (N,)   int32   — 왼쪽=0, 오른쪽=1
    """
    parts = [p for p in (pts_l, pts_r) if len(p) > 0]
    if not parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    all_pts = np.vstack(parts).astype(np.float32)
    cam_idx = np.array(
        [0] * len(pts_l) + [1] * len(pts_r), dtype=np.int32)
    return all_pts, cam_idx


def parse_grasp_array(raw: np.ndarray, n: int) -> list[dict]:
    """C API 반환값(flat double 배열) → grasp dict 리스트."""
    grasps = []
    for i in range(n):
        g = raw[i * _FIELDS_PER_GRASP: (i + 1) * _FIELDS_PER_GRASP]
        grasps.append({
            'score':    float(g[0]),
            'position': g[1:4].copy(),
            'approach': g[4:7].copy(),
            'binormal': g[7:10].copy(),
            'axis':     g[10:13].copy(),
            'width':    float(g[13]),
        })
    return grasps


def grasp_to_pose(grasp: dict) -> Pose:
    """GPD grasp dict → geometry_msgs/Pose."""
    pose = Pose()

    pos = grasp['position']
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])

    R = np.column_stack([grasp['approach'], grasp['binormal'], grasp['axis']])
    R, _ = np.linalg.qr(R)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class GpdDualViewNode(Node):
    def __init__(self):
        super().__init__('gpd_dual_view')

        self.declare_parameter('gpd_dir',     '/root/ros2_ws/src/ai_worker/gpd')
        self.declare_parameter('gpd_config',  'cfg/eigen_params.cfg')
        self.declare_parameter('left_topic',  '/camera_left/points_base')
        self.declare_parameter('right_topic', '/camera_right/points_base')
        self.declare_parameter('left_frame',  'camera_l_depth_optical_frame')
        self.declare_parameter('right_frame', 'camera_r_depth_optical_frame')
        self.declare_parameter('base_frame',  'base_link')
        self.declare_parameter('sync_slop',   0.1)

        gpd_dir = self.get_parameter('gpd_dir').value
        try:
            self._lib = _load_gpd_lib(gpd_dir)
            self.get_logger().info('libgpd_python_wrapper.so loaded.')
        except OSError as e:
            self.get_logger().error(f'Failed to load GPD library: {e}')
            self.get_logger().error(
                'Run: cd gpd/build && cmake .. && make gpd_python_wrapper')
            raise

        self._config_path = os.path.join(
            gpd_dir, self.get_parameter('gpd_config').value).encode()

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        left_sub  = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('left_topic').value)
        right_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter('right_topic').value)

        slop = self.get_parameter('sync_slop').value
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=5, slop=slop)
        self.sync.registerCallback(self._on_clouds)

        self.grasp_pub = self.create_publisher(PoseArray, '/gpd/grasp_poses', 10)
        self.get_logger().info('GPD dual-view node ready.')

    # ------------------------------------------------------------------
    def _get_camera_position(self, camera_frame: str) -> np.ndarray | None:
        base_frame = self.get_parameter('base_frame').value
        try:
            t = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'TF lookup failed ({camera_frame}→{base_frame}): {e}')
            return None

    # ------------------------------------------------------------------
    def _on_clouds(self, left_msg: PointCloud2, right_msg: PointCloud2):
        pts_l = pointcloud2_to_xyz(left_msg)
        pts_r = pointcloud2_to_xyz(right_msg)

        self.get_logger().info(
            f'Clouds received — left: {len(pts_l)}, right: {len(pts_r)} pts')

        if len(pts_l) == 0 and len(pts_r) == 0:
            self.get_logger().warn('Both clouds empty, skipping.')
            return

        cam_l = self._get_camera_position(self.get_parameter('left_frame').value)
        cam_r = self._get_camera_position(self.get_parameter('right_frame').value)
        if cam_l is None or cam_r is None:
            self.get_logger().warn('TF not ready, skipping.')
            return

        grasps = self._run_gpd(pts_l, pts_r, cam_l, cam_r)

        self.get_logger().info(f'GPD detected {len(grasps)} grasps.')
        for i, g in enumerate(grasps):
            pos = g['position']
            self.get_logger().info(
                f'  [{i}] score={g["score"]:.3f}  '
                f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')

        self._publish(grasps)

    # ------------------------------------------------------------------
    def _run_gpd(self, pts_l: np.ndarray, pts_r: np.ndarray,
                 cam_l: np.ndarray, cam_r: np.ndarray) -> list[dict]:
        all_pts, cam_idx = merge_with_index(pts_l, pts_r)
        n = len(all_pts)

        if n == 0:
            self.get_logger().warn('Merged PCD is empty.')
            return []

        self.get_logger().info(f'Merged PCD: {n} pts (left={int((cam_idx==0).sum())}, right={int((cam_idx==1).sum())})')

        # view_points: [cam_l_x, cam_l_y, cam_l_z, cam_r_x, cam_r_y, cam_r_z]
        view_pts = np.concatenate([cam_l, cam_r]).astype(np.float32)

        num_grasps_out = ctypes.c_int(0)
        result_ptr = self._lib.detect_grasps_multi_view(
            self._config_path,
            all_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cam_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            view_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n),
            ctypes.c_int(2),
            ctypes.byref(num_grasps_out),
        )

        n_grasps = num_grasps_out.value
        if n_grasps == 0 or not result_ptr:
            return []

        raw = np.ctypeslib.as_array(result_ptr, shape=(n_grasps * _FIELDS_PER_GRASP,)).copy()
        self._lib.free_grasp_data(result_ptr)

        return parse_grasp_array(raw, n_grasps)

    # ------------------------------------------------------------------
    def _publish(self, grasps: list[dict]):
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('base_frame').value
        msg.poses = [grasp_to_pose(g) for g in grasps]
        self.grasp_pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = GpdDualViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
