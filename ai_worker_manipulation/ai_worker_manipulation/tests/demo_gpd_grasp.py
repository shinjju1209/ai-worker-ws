"""
GPD grasp pose → move_to_pose + gripper close test.
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2

from ai_worker_manipulation.robot_interface.moveit_client import MoveItClient
from ai_worker_manipulation.robot_interface.gripper_controller import GripperInterface
from ai_worker_manipulation.skill_primitives.environment import setup_environment


GRASP_POSITION    = [0.4318, -0.0381, 0.8211]
GRASP_ORIENTATION = [0.2241, -0.2344, -0.5705, 0.7546]
#GRASP_ORIENTATION = [0.0, 0.0, 0.0, 1.0]

LIFT_POSITION     = -0.0   # 15cm 내리기 (0.0 = 최상단, -0.5 = 최하단)
APPROACH_HEIGHT   = 0.20    # pre-grasp: 목표 위 10cm에서 접근 후 lift로 하강


def move_lift(node: Node, position: float, cb_group) -> None:
    moveit_lift = MoveIt2(
        node=node,
        joint_names=['lift_joint'],
        base_link_name='base_link',
        end_effector_name='lift_link',
        group_name='lift',
        callback_group=cb_group,
        use_move_group_action=True,
    )
    moveit_lift.max_velocity    = 0.2
    moveit_lift.max_acceleration = 0.2
    moveit_lift.move_to_configuration([position])
    timeout = time.time() + 10.0
    from pymoveit2.moveit2 import MoveIt2State
    while moveit_lift.query_state() == MoveIt2State.IDLE:
        if time.time() > timeout:
            node.get_logger().warn('lift: goal never left IDLE')
            return
        time.sleep(0.01)
    while moveit_lift.query_state() != MoveIt2State.IDLE:
        if time.time() > timeout:
            node.get_logger().warn('lift: timeout')
            return
        time.sleep(0.05)
    ok = 'SUCCEEDED' if moveit_lift.motion_suceeded else 'FAILED'
    node.get_logger().info(f'lift → {position:.3f}m: {ok}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-home', action='store_true',
                        help='skip move_to_home at the start (use when already at capture pose)')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    rclpy.init()
    node    = Node('demo_gpd_grasp')
    client  = MoveItClient(node)
    log     = node.get_logger()
    gripper = GripperInterface(node=node)
    setup_environment(client)

    pre_grasp_pose = Pose()
    pre_grasp_pose.position.x = GRASP_POSITION[0]
    pre_grasp_pose.position.y = GRASP_POSITION[1]
    pre_grasp_pose.position.z = GRASP_POSITION[2] + APPROACH_HEIGHT
    pre_grasp_pose.orientation.x = GRASP_ORIENTATION[0]
    pre_grasp_pose.orientation.y = GRASP_ORIENTATION[1]
    pre_grasp_pose.orientation.z = GRASP_ORIENTATION[2]
    pre_grasp_pose.orientation.w = GRASP_ORIENTATION[3]

    gripper.open('right')
    if not args.skip_home:
        client.move_to_home()

    log.info(f'lift 내리기: {LIFT_POSITION}m')
    move_lift(node, LIFT_POSITION, client._cb_group)

    log.info(f'pre-grasp 이동 (z = {pre_grasp_pose.position.z:.4f}m)')
    result = client.move_to_pose(pre_grasp_pose)
    log.info(f'pre-grasp result: {result.value}')
    if result.value != 'succeeded':
        return

    log.info(f'lift 추가 하강 ({APPROACH_HEIGHT*100:.0f}cm): {LIFT_POSITION - APPROACH_HEIGHT:.3f}m')
    move_lift(node, LIFT_POSITION - APPROACH_HEIGHT, client._cb_group)

    log.info('그리퍼 닫기')
    gripper.close('right')
    time.sleep(1.5)

    log.info('lift 복귀')
    move_lift(node, 0.0, client._cb_group)
    log.info(f"move_to_home: {client.move_to_home().value}")

    client.destroy()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
