#!/usr/bin/env python3
"""
Dual-view GPD grasp detection node.

팀원이 TF 변환 완료한 두 PointCloud2 토픽을 구독해서:
  1. 두 클라우드 합성 (filter/downsample)
  2. GPD CLI 실행
  3. geometry_msgs/PoseArray 로 grasp pose 퍼블리시

Subscriptions:
  left_topic  (PointCloud2) : 왼팔 카메라, base_link 기준으로 변환 완료된 것
  right_topic (PointCloud2) : 오른팔 카메라, base_link 기준으로 변환 완료된 것

Publish:
  /gpd/grasp_poses (PoseArray, frame_id = base_link)
"""

import os
import re
import subprocess
import tempfile

import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import message_filters


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float64, NaN 제거."""
    pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def merge_and_filter(pts_left: np.ndarray, pts_right: np.ndarray,
                     voxel_size: float) -> o3d.geometry.PointCloud:
    """두 배열 합치고 outlier 제거 + voxel downsample."""
    parts = [p for p in (pts_left, pts_right) if len(p) > 0]
    if not parts:
        return o3d.geometry.PointCloud()
    all_pts = np.vstack(parts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def make_temp_config(base_cfg_path: str, view_point: np.ndarray) -> str:
    """GPD cfg를 복사하고 camera_position을 view_point 로 교체해 임시 파일 반환."""
    with open(base_cfg_path, 'r') as f:
        content = f.read()

    new_pos = f'{view_point[0]:.6f} {view_point[1]:.6f} {view_point[2]:.6f}'
    content = re.sub(r'camera_position\s*=.*', f'camera_position = {new_pos}', content)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False, prefix='gpd_')
    tmp.write(content)
    tmp.close()
    return tmp.name


def parse_gpd_output(stdout: str) -> list[dict]:
    """GPD stdout 파싱 → grasp dict 리스트.

    각 dict: score, position, approach, binormal, axis (모두 np.ndarray)
    """
    grasps: list[dict] = []
    cur: dict = {}

    def _parse_xyz(line: str) -> np.ndarray:
        vals = line.split('x=')[1].split(', y=')
        x = float(vals[0])
        y_str, z_str = vals[1].split(', z=')
        return np.array([x, float(y_str), float(z_str)])

    for line in stdout.splitlines():
        if 'Grasp' in line and 'score:' in line:
            if cur:
                grasps.append(cur)
            score = float(line.split('score:')[1].replace(')', '').strip())
            cur = {'score': score}
        elif cur:
            if 'position:' in line:
                cur['position'] = _parse_xyz(line)
            elif 'approach:' in line:
                cur['approach'] = _parse_xyz(line)
            elif 'binormal:' in line:
                cur['binormal'] = _parse_xyz(line)
            elif 'axis:' in line:
                cur['axis'] = _parse_xyz(line)

    if cur:
        grasps.append(cur)
    return grasps


def grasp_to_pose(grasp: dict) -> Pose:
    """GPD grasp dict → geometry_msgs/Pose (quaternion 변환 포함)."""
    pose = Pose()

    pos = grasp.get('position', np.zeros(3))
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])

    approach = grasp.get('approach', np.array([1.0, 0.0, 0.0]))
    binormal = grasp.get('binormal', np.array([0.0, 1.0, 0.0]))
    axis     = grasp.get('axis',     np.array([0.0, 0.0, 1.0]))

    # GPD hand frame: approach=x, binormal=y, axis=z
    R = np.column_stack([approach, binormal, axis])
    R, _ = np.linalg.qr(R)  # orthonormalize
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

        self.declare_parameter('gpd_dir',      '/home/jiwoo/ai-worker-ws/gpd')
        self.declare_parameter('gpd_config',   'cfg/eigen_params.cfg')
        self.declare_parameter('left_topic',   '/camera_left/points_base')
        self.declare_parameter('right_topic',  '/camera_right/points_base')
        self.declare_parameter('left_frame',   'camera_l_depth_optical_frame')
        self.declare_parameter('right_frame',  'camera_r_depth_optical_frame')
        self.declare_parameter('base_frame',   'base_link')
        self.declare_parameter('voxel_size',   0.003)
        self.declare_parameter('sync_slop',    0.1)
        self.declare_parameter('gpd_timeout',  60.0)

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

        self.get_logger().info(
            f"Subscribing: {self.get_parameter('left_topic').value}, "
            f"{self.get_parameter('right_topic').value}")
        self.get_logger().info("GPD dual-view node ready.")

    # ------------------------------------------------------------------
    def _get_camera_position(self, camera_frame: str) -> np.ndarray | None:
        base_frame = self.get_parameter('base_frame').value
        try:
            t = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'TF lookup failed ({camera_frame} → {base_frame}): {e}')
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

        # 두 카메라 중간점을 GPD camera_position 으로 사용
        view_point = (cam_l + cam_r) / 2.0

        grasps = self._run_gpd(pts_l, pts_r, view_point)

        self.get_logger().info(f'GPD detected {len(grasps)} grasps.')
        for i, g in enumerate(grasps):
            pos = g.get('position', np.zeros(3))
            self.get_logger().info(
                f'  [{i}] score={g["score"]:.3f}  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')

        self._publish(grasps)

    # ------------------------------------------------------------------
    def _run_gpd(self, pts_l: np.ndarray, pts_r: np.ndarray,
                 view_point: np.ndarray) -> list[dict]:
        gpd_dir    = self.get_parameter('gpd_dir').value
        config_rel = self.get_parameter('gpd_config').value
        config_abs = os.path.join(gpd_dir, config_rel)
        voxel_size = self.get_parameter('voxel_size').value
        timeout    = self.get_parameter('gpd_timeout').value

        pcd = merge_and_filter(pts_l, pts_r, voxel_size)
        self.get_logger().info(f'Merged PCD: {len(pcd.points)} pts after filter/downsample')

        if len(pcd.points) == 0:
            self.get_logger().warn('Merged PCD is empty.')
            return []

        tmp_pcd = tempfile.NamedTemporaryFile(suffix='.pcd', delete=False, prefix='gpd_in_')
        tmp_pcd.close()
        tmp_cfg = None

        try:
            o3d.io.write_point_cloud(tmp_pcd.name, pcd)
            tmp_cfg = make_temp_config(config_abs, view_point)

            result = subprocess.run(
                ['./build/detect_grasps', tmp_cfg, tmp_pcd.name],
                cwd=gpd_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'LIBGL_ALWAYS_SOFTWARE': '1'},
            )

            if result.returncode != 0:
                self.get_logger().error(f'GPD exited with code {result.returncode}')
                self.get_logger().error(result.stderr[-500:])
                return []

            return parse_gpd_output(result.stdout)

        except subprocess.TimeoutExpired:
            self.get_logger().error(f'GPD timed out after {timeout}s.')
            return []
        except FileNotFoundError:
            self.get_logger().error(
                f"GPD binary not found: {gpd_dir}/build/detect_grasps")
            return []
        finally:
            os.unlink(tmp_pcd.name)
            if tmp_cfg:
                os.unlink(tmp_cfg)

    # ------------------------------------------------------------------
    def _publish(self, grasps: list[dict]):
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('base_frame').value
        msg.poses = [grasp_to_pose(g) for g in grasps if 'position' in g]
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
