#!/usr/bin/env python3
"""
GPD grasp publisher node.

TF 변환이 완료된 mask cloud와 물체 pose 토픽을 받아 GPD를 실행하고
MoveIt에 사용할 수 있는 grasp pose를 발행합니다.

Subscribe:
  <cloud_topic>  (PointCloud2,  base_link 기준, TF 변환 완료)
  <pose_topic>   (PoseStamped,  base_link 기준, 물체 중심 위치)

Publish:
  /gpd/grasp_poses  (PoseArray,   base_link 기준) — 필터링된 전체
  /gpd/best_grasp   (PoseStamped, base_link 기준) — score 1위

Usage:
  ros2 run ai_worker_manipulation gpd_grasp_publisher
  ros2 run ai_worker_manipulation gpd_grasp_publisher \\
    --ros-args -p cloud_topic:=/perception/wrist/target_pcd/bottle \\
               -p pose_topic:=/perception/wrist/target_pose/bottle
"""

import os
import re
import subprocess
import tempfile
import threading

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


# ---------------------------------------------------------------------------
# 필터 상수  (test_gpd_wrist150.py와 동일)
# ---------------------------------------------------------------------------
MIN_POINTS         = 50
APPROACH_THRESHOLD = 0.3   # camera-approach cosine similarity
POSITION_RADIUS    = 1.0  # grasp position이 물체 중심에서 허용 거리 (m)
MAX_GRASPS         = 8

GPD_DIR = '/root/ros2_ws/src/ai_worker/gpd'
GPD_CFG = 'cfg/eigen_params.cfg'


# ---------------------------------------------------------------------------
# Filters  (test_gpd_wrist150.py와 동일한 로직)
# ---------------------------------------------------------------------------

def filter_by_approach(grasps: list[dict],
                       camera_pos: np.ndarray,
                       object_center: np.ndarray,
                       threshold: float = APPROACH_THRESHOLD) -> list[dict]:
    cam_dir = object_center - camera_pos
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-9)
    valid = []
    for g in grasps:
        if 'approach' not in g:
            continue
        approach_norm = g['approach'] / (np.linalg.norm(g['approach']) + 1e-9)
        cos_sim = float(np.dot(cam_dir, approach_norm))
        if cos_sim >= threshold:
            valid.append(g)
    return valid


def filter_by_approach_x(grasps: list[dict]) -> list[dict]:
    return [g for g in grasps if g.get('approach', np.zeros(3))[0] >= 0]


def filter_by_position(grasps: list[dict],
                       object_center: np.ndarray,
                       radius: float = POSITION_RADIUS) -> list[dict]:
    valid = []
    for g in grasps:
        if 'position' not in g:
            continue
        if float(np.linalg.norm(g['position'] - object_center)) <= radius:
            valid.append(g)
    return valid


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def write_pcd(path: str, points: np.ndarray):
    pts = points.astype(np.float32)
    n = len(pts)
    header = (
        f"# .PCD v0.7 - Point Cloud Data file\n"
        f"VERSION 0.7\n"
        f"FIELDS x y z\n"
        f"SIZE 4 4 4\n"
        f"TYPE F F F\n"
        f"COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(pts.tobytes())


def make_temp_config(base_cfg_path: str, view_point: np.ndarray) -> str:
    with open(base_cfg_path, 'r') as f:
        content = f.read()
    new_pos = f'{view_point[0]:.6f} {view_point[1]:.6f} {view_point[2]:.6f}'
    content = re.sub(r'camera_position\s*=.*', f'camera_position = {new_pos}', content)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False, prefix='gpd_cfg_')
    tmp.write(content)
    tmp.close()
    return tmp.name


def parse_gpd_output(stdout: str) -> list[dict]:
    grasps: list[dict] = []
    cur: dict = {}

    def _xyz(line: str, key: str) -> np.ndarray:
        try:
            rest = line.split(key + ':')[1]
            parts = rest.strip().replace('x=', '').replace('y=', '').replace('z=', '')
            vals = [v.strip().rstrip(',') for v in parts.split()]
            return np.array([float(vals[0]), float(vals[1]), float(vals[2])])
        except Exception:
            return np.zeros(3)

    for line in stdout.splitlines():
        if 'Grasp' in line and 'score:' in line:
            if cur:
                grasps.append(cur)
            cur = {'score': float(line.split('score:')[1].replace(')', '').strip())}
        elif cur:
            if 'position:' in line:
                cur['position'] = _xyz(line, 'position')
            elif 'approach:' in line:
                cur['approach'] = _xyz(line, 'approach')
            elif 'binormal:' in line:
                cur['binormal'] = _xyz(line, 'binormal')
            elif 'axis:' in line:
                cur['axis'] = _xyz(line, 'axis')
    if cur:
        grasps.append(cur)
    return grasps


def grasp_to_pose(grasp: dict) -> Pose:
    """grasp dict → geometry_msgs/Pose (test_gpd_wrist150.py와 동일한 quaternion 변환)."""
    approach = grasp.get('approach', np.array([1.0, 0.0, 0.0]))
    axis     = grasp.get('axis',     np.array([0.0, 0.0, 1.0]))

    approach = approach / (np.linalg.norm(approach) + 1e-9)
    axis     = axis     / (np.linalg.norm(axis)     + 1e-9)

    if axis[0] < 0:
        axis = -axis

    binormal = np.cross(axis, approach)
    binormal = binormal / (np.linalg.norm(binormal) + 1e-9)

    if binormal[1] < 0:
        binormal = -binormal

    axis = np.cross(approach, binormal)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    R    = np.column_stack([axis, binormal, -approach])
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

    pos  = grasp.get('position', np.zeros(3))
    pose = Pose()
    pose.position.x    = float(pos[0])
    pose.position.y    = float(pos[1])
    pose.position.z    = float(pos[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class GpdGraspPublisher(Node):

    def __init__(self):
        super().__init__('gpd_grasp_publisher')

        self.declare_parameter('cloud_topic',  '/perception/wrist/target_pcd/target')
        self.declare_parameter('pose_topic',   '/perception/wrist/target_pose/target')
        self.declare_parameter('base_frame',   'base_link')
        self.declare_parameter('gpd_dir',      GPD_DIR)
        self.declare_parameter('gpd_config',   GPD_CFG)
        self.declare_parameter('gpd_timeout',  60.0)

        cloud_topic = self.get_parameter('cloud_topic').value
        pose_topic  = self.get_parameter('pose_topic').value

        best_effort_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_cloud = self.create_subscription(
            PointCloud2, cloud_topic, self._on_cloud, best_effort_qos)
        self.sub_pose  = self.create_subscription(
            PoseStamped, pose_topic, self._on_pose, 10)

        self.pub_poses = self.create_publisher(PoseArray,   '/gpd/grasp_poses', 10)
        self.pub_best  = self.create_publisher(PoseStamped, '/gpd/best_grasp',  10)

        self._latest_object_center: np.ndarray | None = None
        self._running = False

        self.get_logger().info(f'cloud_topic : {cloud_topic}')
        self.get_logger().info(f'pose_topic  : {pose_topic}')
        self.get_logger().info('GPD grasp publisher ready.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_pose(self, msg: PoseStamped):
        self._latest_object_center = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def _on_cloud(self, msg: PointCloud2):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._process, args=(msg,), daemon=True).start()

    # ── Processing pipeline ───────────────────────────────────────────────────

    def _process(self, msg: PointCloud2):
        try:
            pts = pointcloud2_to_xyz(msg)
            self.get_logger().info(f'Cloud received: {len(pts)} pts')

            if len(pts) < MIN_POINTS:
                self.get_logger().warn(f'포인트 수 부족 ({len(pts)}개), GPD 스킵')
                return

            centroid      = pts.mean(axis=0)
            camera_pos    = self._get_camera_position(centroid)
            object_center = self._latest_object_center if self._latest_object_center is not None else centroid

            grasps = self._run_gpd(pts, camera_pos)
            if not grasps:
                return

            # test_gpd_wrist150.py와 동일한 필터 순서
            grasps = filter_by_approach(grasps, camera_pos, object_center)
            grasps = filter_by_approach_x(grasps)
            grasps = filter_by_position(grasps, object_center)
            grasps = sorted(grasps, key=lambda g: g['score'], reverse=True)[:MAX_GRASPS]

            self.get_logger().info(f'필터 후 유효 grasp: {len(grasps)}개')
            if not grasps:
                self.get_logger().warn('유효한 grasp 없음')
                return

            for i, g in enumerate(grasps):
                pos = g.get('position', np.zeros(3))
                self.get_logger().info(
                    f'  [{i}] score={g["score"]:.3f}  '
                    f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')

            self._publish(grasps, msg.header.stamp)

        finally:
            self._running = False

    # ── Camera position ───────────────────────────────────────────────────────

    def _get_camera_position(self, centroid: np.ndarray) -> np.ndarray:
        """test_gpd_wrist150.py와 동일: centroid 기준 오른쪽 손목 카메라 근사 위치."""
        return centroid + np.array([0.0, -0.3, 0.4])

    # ── GPD ───────────────────────────────────────────────────────────────────

    def _run_gpd(self, points: np.ndarray, camera_pos: np.ndarray) -> list[dict]:
        gpd_dir    = self.get_parameter('gpd_dir').value
        config_abs = os.path.join(gpd_dir, self.get_parameter('gpd_config').value)
        timeout    = self.get_parameter('gpd_timeout').value

        tmp_pcd = tempfile.NamedTemporaryFile(suffix='.pcd', delete=False, prefix='gpd_in_')
        tmp_pcd.close()
        tmp_cfg = None

        try:
            write_pcd(tmp_pcd.name, points)
            tmp_cfg = make_temp_config(config_abs, camera_pos)

            self.get_logger().info(
                f'GPD 실행 중 — {len(points)}pts, camera_pos={np.round(camera_pos, 3)}')

            result = subprocess.run(
                ['./build/detect_grasps', tmp_cfg, tmp_pcd.name],
                cwd=gpd_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'LIBGL_ALWAYS_SOFTWARE': '1'},
            )

            if result.returncode != 0:
                self.get_logger().error(
                    f'GPD 실패 (code {result.returncode}): {result.stderr[-500:]}')
                return []

            return parse_gpd_output(result.stdout)

        except subprocess.TimeoutExpired:
            self.get_logger().error(f'GPD timeout ({timeout}s)')
            return []
        except FileNotFoundError:
            self.get_logger().error(f'GPD binary not found: {gpd_dir}/build/detect_grasps')
            return []
        finally:
            os.unlink(tmp_pcd.name)
            if tmp_cfg:
                os.unlink(tmp_cfg)

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish(self, grasps: list[dict], stamp):
        frame_id = self.get_parameter('base_frame').value

        arr = PoseArray()
        arr.header.stamp    = stamp
        arr.header.frame_id = frame_id
        arr.poses = [grasp_to_pose(g) for g in grasps if 'position' in g]
        self.pub_poses.publish(arr)

        best = PoseStamped()
        best.header.stamp    = stamp
        best.header.frame_id = frame_id
        best.pose = arr.poses[0]
        self.pub_best.publish(best)

        pos = grasps[0]['position']
        self.get_logger().info(
            f'발행 — best: score={grasps[0]["score"]:.3f}  '
            f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = GpdGraspPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
