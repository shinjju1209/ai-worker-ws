#!/usr/bin/env python3
"""
grasp_please 데이터셋 → GPD grasp detection + Open3D 시각화

Usage:
  # 단일 프레임 (기본 0번)
  DISPLAY=:0 python3 test_gpd_grasp_please.py --frame 0

  # run 폴더 지정
  DISPLAY=:0 python3 test_gpd_grasp_please.py --run run_20260603_135710 --frame 0

  # sequential 모드 (N키 = 다음 프레임)
  DISPLAY=:0 python3 test_gpd_grasp_please.py --mode sequential

  # GPD 없이 포인트 클라우드만 보기
  DISPLAY=:0 python3 test_gpd_grasp_please.py --frame 0 --no-gpd
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

GRASP_PLEASE_DIR = "/root/ros2_ws/src/ai_worker/grasp_please"
DEFAULT_RUN      = "run_20260603_133129"   # 기본 run 폴더명 (변경 가능)
GPD_DIR          = "/root/ros2_ws/src/ai_worker/gpd"
GPD_CFG          = "cfg/eigen_params.cfg"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def sorted_pcd_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "sample_*.pcd")))


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(path)


def load_pose_json(pcd_path: str) -> dict | None:
    """pcd 경로에서 대응하는 sample_XXX_pose.json 로드."""
    pose_path = pcd_path.replace('.pcd', '_pose.json')
    if not os.path.exists(pose_path):
        return None
    with open(pose_path) as f:
        data = json.load(f)
    pos  = data['pose']['position']
    quat = data['pose']['orientation']
    return {
        'position': np.array([pos['x'],  pos['y'],  pos['z']]),
        'quat':     np.array([quat['x'], quat['y'], quat['z'], quat['w']]),
    }


# ---------------------------------------------------------------------------
# Filters  (gpd_grasp_publisher.py / test_gpd_wrist150.py와 동일한 로직)
# ---------------------------------------------------------------------------

def filter_by_approach(grasps: list[dict],
                       camera_pos: np.ndarray,
                       object_center: np.ndarray,
                       threshold: float = 0.3) -> list[dict]:
    cam_dir = object_center - camera_pos
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-9)
    valid = []
    for g in grasps:
        if 'approach' not in g:
            continue
        approach_norm = g['approach'] / (np.linalg.norm(g['approach']) + 1e-9)
        cos_sim = float(np.dot(cam_dir, approach_norm))
        g['cos_sim'] = cos_sim
        if cos_sim >= threshold:
            valid.append(g)
    print(f"[필터] approach 유사도 >= {threshold}: {len(valid)}/{len(grasps)}개 통과")
    return valid


def filter_by_approach_x(grasps: list[dict]) -> list[dict]:
    valid = [g for g in grasps if g.get('approach', np.zeros(3))[0] >= 0]
    print(f"[필터] approach X >= 0: {len(valid)}/{len(grasps)}개 통과")
    return valid


def filter_by_position(grasps: list[dict],
                       object_center: np.ndarray,
                       radius: float = 0.07) -> list[dict]:
    valid = []
    for g in grasps:
        if 'position' not in g:
            continue
        if float(np.linalg.norm(g['position'] - object_center)) <= radius:
            valid.append(g)
    print(f"[필터] grasp position 거리 <= {radius}m: {len(valid)}/{len(grasps)}개 통과")
    return valid


# ---------------------------------------------------------------------------
# GPD
# ---------------------------------------------------------------------------

def run_gpd(pcd: o3d.geometry.PointCloud) -> tuple[list[dict], np.ndarray]:
    """GPD 실행 → (grasps, camera_pos) 반환."""
    centroid   = np.asarray(pcd.points).mean(axis=0)
    camera_pos = centroid + np.array([0.0, -0.3, 0.4])

    tmp_pcd = tempfile.NamedTemporaryFile(suffix='.pcd', delete=False, prefix='gpd_gp_')
    tmp_pcd.close()
    tmp_cfg = None

    try:
        o3d.io.write_point_cloud(tmp_pcd.name, pcd)
        tmp_cfg = _make_temp_config(os.path.join(GPD_DIR, GPD_CFG), camera_pos)

        print(f"[GPD] 포인트 수: {len(pcd.points)}, camera_pos={np.round(camera_pos, 3)}")
        print("[GPD] 실행 중...")

        result = subprocess.run(
            ['./build/detect_grasps', tmp_cfg, tmp_pcd.name],
            cwd=GPD_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'LIBGL_ALWAYS_SOFTWARE': '1'},
        )
    finally:
        os.unlink(tmp_pcd.name)
        if tmp_cfg:
            os.unlink(tmp_cfg)

    if result.returncode != 0:
        print(f"[GPD ERROR] returncode={result.returncode}")
        print(result.stderr[-500:])
        return [], camera_pos

    printing = False
    for line in result.stdout.split('\n'):
        if 'Selected grasps' in line:
            printing = True
        if printing:
            print(line)

    return _parse_gpd_output(result.stdout), camera_pos


def _make_temp_config(base_cfg: str, view_point: np.ndarray) -> str:
    import re
    with open(base_cfg) as f:
        content = f.read()
    new_pos = f'{view_point[0]:.6f} {view_point[1]:.6f} {view_point[2]:.6f}'
    content = re.sub(r'camera_position\s*=.*', f'camera_position = {new_pos}', content)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False, prefix='gpd_cfg_')
    tmp.write(content)
    tmp.close()
    return tmp.name


def _parse_gpd_output(stdout: str) -> list[dict]:
    grasps, cur = [], {}

    def _xyz(line, key):
        try:
            rest  = line.split(key + ':')[1]
            parts = rest.strip().replace('x=', '').replace('y=', '').replace('z=', '')
            vals  = [v.strip().rstrip(',') for v in parts.split()]
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def grasp_to_quaternion(grasp: dict) -> np.ndarray:
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

    R = np.column_stack([axis, binormal, -approach])
    return Rotation.from_matrix(R).as_quat()  # [x, y, z, w]


def make_grasp_frame(grasp: dict, scale=0.04) -> o3d.geometry.TriangleMesh:
    pos      = grasp.get('position', np.zeros(3))
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
    R = np.column_stack([approach, binormal, axis])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.rotate(R, center=np.zeros(3))
    frame.translate(pos)
    return frame


def make_gt_sphere(position: np.ndarray, radius=0.015) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    sphere.paint_uniform_color([1.0, 0.9, 0.0])
    sphere.compute_vertex_normals()
    return sphere


def show_frame(pcd, grasps, gt_pose, title):
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geoms = [pcd, base_frame]

    for i, g in enumerate(grasps):
        if 'position' not in g:
            continue
        geoms.append(make_grasp_frame(g, scale=0.04))
        quat = grasp_to_quaternion(g)
        pos  = g['position']
        app  = g.get('approach', np.zeros(3))
        ax   = g.get('axis',     np.zeros(3))
        bn   = g.get('binormal', np.zeros(3))
        print(f"  Grasp {i}: score={g['score']:.3f}")
        print(f"    position:    x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
        print(f"    approach:    x={app[0]:.4f}  y={app[1]:.4f}  z={app[2]:.4f}")
        print(f"    axis:        x={ax[0]:.4f}  y={ax[1]:.4f}  z={ax[2]:.4f}")
        print(f"    binormal:    x={bn[0]:.4f}  y={bn[1]:.4f}  z={bn[2]:.4f}")
        print(f"    orientation: x={quat[0]:.4f}  y={quat[1]:.4f}  z={quat[2]:.4f}  w={quat[3]:.4f}")

    if gt_pose is not None:
        geoms.append(make_gt_sphere(gt_pose['position']))
        print(f"  GT position: {np.round(gt_pose['position'], 3)}")

    print()
    print("  [좌표축] GPD grasp pose (RGB = XYZ)")
    if gt_pose is not None:
        print("  [노란 구] Ground truth target position")
    print("  마우스 드래그: 회전 / 스크롤: 줌 / Q: 종료")

    centroid = np.asarray(pcd.points).mean(axis=0)
    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280, height=720,
        lookat=centroid,
        up=[0.0, 0.0, 1.0],
        front=[0.0, -1.0, 0.3],
        zoom=0.5,
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single(files, frame_idx, use_gpd):
    idx   = frame_idx % len(files)
    path  = files[idx]
    fname = os.path.basename(path)
    print(f"\n[프레임 {idx:03d}] {fname}")

    pcd     = load_pcd(path)
    gt_pose = load_pose_json(path)
    pts     = np.asarray(pcd.points)
    print(f"  포인트 수: {len(pts)}")

    grasps = []
    if use_gpd:
        if len(pts) < 50:
            print("  [경고] 포인트가 너무 적어 GPD 스킵")
        else:
            centroid = pts.mean(axis=0)
            object_center = gt_pose['position'] if gt_pose else centroid
            grasps, camera_pos = run_gpd(pcd)
            grasps = filter_by_approach(grasps, camera_pos, object_center)
            grasps = filter_by_approach_x(grasps)
            grasps = filter_by_position(grasps, object_center, radius=0.10)
            grasps = sorted(grasps, key=lambda g: g['score'], reverse=True)[:8]
            print(f"\n[결과] 필터링 후 유효 grasp: {len(grasps)}개 (최대 8개, score 순)")

    title = f"grasp_please — {fname}  |  GPD {len(grasps)} grasps"
    show_frame(pcd, grasps, gt_pose, title)


def run_sequential(files, use_gpd):
    print(f"[sequential] {len(files)}프레임  |  N=다음 프레임 / Q=종료")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="grasp_please — Sequential", width=1280, height=720)

    state = {"idx": 0, "geoms": []}

    def _load_geoms(idx):
        path    = files[idx]
        pcd     = load_pcd(path)
        gt_pose = load_pose_json(path)

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geoms = [pcd, base_frame]

        if use_gpd and len(pcd.points) >= 50:
            centroid      = np.asarray(pcd.points).mean(axis=0)
            object_center = gt_pose['position'] if gt_pose else centroid
            grasps, camera_pos = run_gpd(pcd)
            grasps = filter_by_approach(grasps, camera_pos, object_center)
            grasps = filter_by_approach_x(grasps)
            grasps = filter_by_position(grasps, object_center, radius=0.10)
            grasps = sorted(grasps, key=lambda g: g['score'], reverse=True)[:8]
            print(f"  frame {idx:03d}: {len(grasps)} grasps  ({len(pcd.points)} pts)")
            for g in grasps:
                if 'position' in g:
                    geoms.append(make_grasp_frame(g))

        if gt_pose is not None:
            geoms.append(make_gt_sphere(gt_pose['position']))
        return geoms

    def next_frame(vis):
        for g in state["geoms"]:
            vis.remove_geometry(g, reset_bounding_box=False)
        state["idx"] = (state["idx"] + 1) % len(files)
        state["geoms"] = _load_geoms(state["idx"])
        for g in state["geoms"]:
            vis.add_geometry(g, reset_bounding_box=False)
        vis.update_renderer()
        return False

    state["geoms"] = _load_geoms(0)
    for g in state["geoms"]:
        vis.add_geometry(g)

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("n"), next_frame)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',    type=str, default=DEFAULT_RUN,
                        help=f'grasp_please 안의 run 폴더명 (기본: {DEFAULT_RUN})')
    parser.add_argument('--frame',  type=int, default=0)
    parser.add_argument('--mode',   choices=['single', 'sequential'], default='single')
    parser.add_argument('--no-gpd', action='store_true', help='GPD 없이 시각화만')
    args = parser.parse_args()

    run_dir = os.path.join(GRASP_PLEASE_DIR, args.run)
    if not os.path.isdir(run_dir):
        print(f"[ERROR] run 폴더 없음: {run_dir}")
        return

    files = sorted_pcd_files(run_dir)
    if not files:
        print(f"[ERROR] PCD 파일 없음: {run_dir}")
        return
    print(f"run: {args.run}  |  PCD {len(files)}개 발견")

    use_gpd = not args.no_gpd

    if args.mode == 'sequential':
        run_sequential(files, use_gpd)
    else:
        run_single(files, args.frame, use_gpd)


if __name__ == '__main__':
    main()
