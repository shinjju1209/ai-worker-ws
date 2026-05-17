import open3d as o3d
import numpy as np
import subprocess
import os

def capture_from_viewpoint(pcd_full, camera_location):
    _, pt_map = pcd_full.hidden_point_removal(camera_location, radius=100)
    return pcd_full.select_by_index(pt_map)

def parse_grasp_poses(output):
    grasps = []
    lines = output.split('\n')
    current_grasp = {}
    for line in lines:
        if 'Grasp' in line and 'score:' in line:
            if current_grasp:
                grasps.append(current_grasp)
            score_str = line.split('score:')[1].replace(')', '').replace(':', '').strip()
            current_grasp = {'score': float(score_str)}
        elif 'position:' in line:
            vals = line.split('x=')[1].split(', y=')
            x = float(vals[0])
            y, z = vals[1].split(', z=')
            current_grasp['position'] = np.array([x, float(y), float(z)])
        elif 'approach:' in line:
            vals = line.split('x=')[1].split(', y=')
            x = float(vals[0])
            y, z = vals[1].split(', z=')
            current_grasp['approach'] = np.array([x, float(y), float(z)])
    if current_grasp:
        grasps.append(current_grasp)
    return grasps

def filter_grasps_by_approach(grasps, camera_positions, object_center=np.array([0,0,0])):
    valid_grasps = []
    for grasp in grasps:
        if 'approach' not in grasp:
            continue
        approach_norm = grasp['approach'] / np.linalg.norm(grasp['approach'])
        is_valid = False
        for cam_pos in camera_positions:
            cam_dir = object_center - cam_pos
            cam_dir = cam_dir / np.linalg.norm(cam_dir)
            if np.dot(approach_norm, cam_dir) > 0.3:
                is_valid = True
                break
        if is_valid:
            valid_grasps.append(grasp)
    return valid_grasps

def run_gpd(pcd_path, scene_name, camera_positions):
    print(f"\n{'='*50}")
    print(f"Scene: {scene_name}")
    print(f"{'='*50}")
    gpd_dir = "/root/ros2_ws/src/ai_worker/gpd"
    result = subprocess.run(
        ["./build/detect_grasps", "cfg/eigen_params.cfg", pcd_path],
        cwd=gpd_dir,
        capture_output=True,
        text=True,
        env={**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1"}
    )
    lines = result.stdout.split('\n')
    printing = False
    for line in lines:
        if 'GRASP POSES' in line or 'Selected grasps' in line:
            printing = True
        if printing:
            print(line)

    grasps = parse_grasp_poses(result.stdout)
    valid = filter_grasps_by_approach(grasps, camera_positions)
    print(f"\n필터링 후 유효한 grasp: {len(valid)}/{len(grasps)}개")
    for i, g in enumerate(valid):
        print(f"  Valid Grasp {i} (score: {g['score']:.2f})")
        print(f"    position: {g['position']}")
        print(f"    approach: {g['approach']}")

# AI Worker 양팔 카메라 시점
camera_positions = [
    np.array([0.3, -0.2, 0.3]),  # 오른팔 카메라
    np.array([0.3,  0.2, 0.3]),  # 왼팔 카메라
]

# =============================================
# 여러 물체가 있는 Scene 생성
# 테이블 위에 물체들이 놓여있는 상황
# =============================================

# 1. 실린더 (음료캔) - 왼쪽
cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.12)
cyl.translate(np.array([-0.1, 0.0, 0.06]))  # 왼쪽에 배치

# 2. 박스 - 가운데
box = o3d.geometry.TriangleMesh.create_box(width=0.06, height=0.06, depth=0.08)
box.translate(np.array([0.0, 0.0, 0.04]))   # 가운데 배치

# 3. 구 - 오른쪽
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
sphere.translate(np.array([0.1, 0.0, 0.03]))  # 오른쪽에 배치

# 4. 바닥 (테이블)
table = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.01)
table.translate(np.array([-0.25, -0.25, -0.01]))

# 각 mesh를 포인트 클라우드로 변환 후 합성
all_points = []
for mesh in [cyl, box, sphere, table]:
    pcd_full = mesh.sample_points_uniformly(number_of_points=5000)
    for cam_pos in camera_positions:
        pcd_view = capture_from_viewpoint(pcd_full, cam_pos)
        all_points.append(np.asarray(pcd_view.points))

# 합성 + 다운샘플링
scene_pcd = o3d.geometry.PointCloud()
scene_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.003)
print(f"Scene 포인트 수: {len(scene_pcd.points)}")

o3d.io.write_point_cloud("/tmp/test_scene.pcd", scene_pcd)
run_gpd("/tmp/test_scene.pcd", "멀티 오브젝트 Scene (실린더+박스+구+테이블)", camera_positions)