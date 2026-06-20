# gpd_grasp_publisher 노드 설명

## 개요

손목 카메라로 촬영한 물체의 포인트 클라우드를 받아 GPD(Grasp Pose Detection)를 실행하고,
MoveIt에서 바로 사용할 수 있는 grasp pose를 토픽으로 발행하는 노드입니다.

---

## 실행 방법

```bash
# 기본 실행 (기본 토픽명 사용)
ros2 run ai_worker_manipulation gpd_grasp_publisher

# 토픽명 직접 지정
ros2 run ai_worker_manipulation gpd_grasp_publisher \
  --ros-args \
  -p cloud_topic:=/perception/wrist/target_pcd/bottle \
  -p pose_topic:=/perception/wrist/target_pose/bottle
```

---

## 구독 토픽 (Subscribe)

### 1. `/perception/wrist/target_pcd/target` — PointCloud2

물체 영역만 마스킹된 포인트 클라우드입니다.
**base_link 기준으로 TF 변환이 완료된 상태**여야 합니다.

```
- 메시지 타입 : sensor_msgs/PointCloud2
- 좌표계      : base_link
- QoS         : BEST_EFFORT
- 클라우드가 도착할 때마다 GPD 실행이 트리거됨
- 포인트 수가 50개 미만이면 GPD 스킵
```

### 2. `/perception/wrist/target_pose/target` — PoseStamped

물체 중심 위치입니다. GPD 결과 필터링 시 물체 중심 반경 7cm 이내 grasp만 통과시키는 데 사용합니다.

```
- 메시지 타입 : geometry_msgs/PoseStamped
- 좌표계      : base_link
- 가장 최근에 수신한 값을 캐시해서 사용
- 수신 전까지는 포인트 클라우드의 centroid로 대체
```

> **토픽명 변경 시** `--ros-args -p cloud_topic:=<이름> -p pose_topic:=<이름>` 으로 지정

---

## 발행 토픽 (Publish)

### 1. `/gpd/best_grasp` — PoseStamped ★ MoveIt 입력용

필터링된 grasp 중 **score 1위** pose 하나를 발행합니다.
MoveIt `move_to_pose()`에 바로 넘길 수 있습니다.

```
- 메시지 타입 : geometry_msgs/PoseStamped
- 좌표계      : base_link (header.frame_id = "base_link")
```

**메시지 구조 예시:**
```
header:
  frame_id: "base_link"
  stamp: ...
pose:
  position:
    x: 0.3014
    y: -0.2542
    z: 0.8930
  orientation:     # 쿼터니언 [x, y, z, w]
    x: 0.3846
    y: -0.0636
    z: 0.0507
    w: 0.9195
```

**토픽 확인:**
```bash
ros2 topic echo /gpd/best_grasp
```

---

### 2. `/gpd/grasp_poses` — PoseArray

필터링된 grasp 전체 (최대 8개, score 내림차순) 를 발행합니다.

```
- 메시지 타입 : geometry_msgs/PoseArray
- 좌표계      : base_link (header.frame_id = "base_link")
- poses[0]    : score 1위 (best_grasp와 동일)
- poses[1..N] : score 2위 이하
```

**메시지 구조 예시:**
```
header:
  frame_id: "base_link"
poses:
  - position: {x: 0.3014, y: -0.2542, z: 0.8930}
    orientation: {x: 0.3846, y: -0.0636, z: 0.0507, w: 0.9195}
  - position: {x: 0.3021, y: -0.2518, z: 0.8945}
    orientation: {x: 0.3712, y: -0.0541, z: 0.0489, w: 0.9268}
  ...
```

**토픽 확인:**
```bash
ros2 topic echo /gpd/grasp_poses
```

---

## 처리 흐름

```
[cloud_topic] ─────┐
                   ▼
            포인트 수 체크 (< 50개 → 스킵)
                   │
            GPD 실행 (detect_grasps 바이너리)
                   │
            ┌──────▼──────────────────────────────┐
            │  필터 1: approach-카메라 방향 유사도 ≥ 0.3  │
            │  필터 2: approach X 방향 ≥ 0 (로봇 전방)  │
            │  필터 3: 물체 중심에서 7cm 이내           │
            │  정렬: score 내림차순, 상위 8개           │
            └──────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  /gpd/best_grasp       /gpd/grasp_poses
  (PoseStamped, 1위)    (PoseArray, 전체)

[pose_topic] ──→ 물체 중심 캐시 → 필터 3에 사용
```

---

## 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `cloud_topic` | `/perception/wrist/target_pcd/target` | 구독할 PointCloud2 토픽 |
| `pose_topic` | `/perception/wrist/target_pose/target` | 구독할 물체 pose 토픽 |
| `base_frame` | `base_link` | 발행 토픽의 좌표계 |
| `gpd_dir` | `/root/ros2_ws/src/ai_worker/gpd` | GPD 바이너리 경로 |
| `gpd_config` | `cfg/eigen_params.cfg` | GPD 설정 파일 경로 |
| `gpd_timeout` | `60.0` | GPD 실행 타임아웃 (초) |

---

## 필터 상수 (코드 상단에서 수정)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `MIN_POINTS` | `50` | GPD 실행 최소 포인트 수 |
| `APPROACH_THRESHOLD` | `0.3` | 카메라-approach 방향 cosine similarity 하한 |
| `POSITION_RADIUS` | `0.07` | 물체 중심에서 허용 거리 (m) |
| `MAX_GRASPS` | `8` | 발행할 최대 grasp 수 |

---

## MoveIt 연동 예시

`/gpd/best_grasp`를 구독해서 `move_to_pose`에 직접 사용:

```python
from geometry_msgs.msg import PoseStamped

def grasp_cb(self, msg: PoseStamped):
    result = self.moveit_client.move_to_pose(msg.pose, arm=Arm.RIGHT)
```

또는 `demo_gpd_grasp.py`처럼 상수로 박아서 테스트:

```python
GRASP_POSITION    = [0.3014, -0.2542, 0.8930]
GRASP_ORIENTATION = [0.3846, -0.0636, 0.0507, 0.9195]  # [x, y, z, w]
```

---

## 주의사항

- GPD 실행 중 새 cloud 메시지가 들어오면 **무시**됩니다 (GPD 한 번에 하나씩 실행).
- `pose_topic`을 수신하기 전에 cloud가 들어오면 **포인트 클라우드의 centroid**를 물체 중심으로 대체합니다.
- 카메라 위치는 `centroid + [0, -0.3, 0.4]` (base_link 기준) 로 근사합니다.
- GPD 바이너리(`detect_grasps`)가 빌드되어 있어야 합니다:
  ```bash
  cd /root/ros2_ws/src/ai_worker/gpd && mkdir -p build && cd build
  cmake .. && make -j$(nproc)
  ```
