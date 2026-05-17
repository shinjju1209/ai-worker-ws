#!/usr/bin/env python3
"""
Reusable gripper controller library.

Usage from another script:
    from ai_worker_manipulation.robot_interface.gripper_controller import GripperController

    gc = GripperController()
    gc.control('left', 0.5)   # set left gripper to 0.5
    gc.control('right', 0.0)  # open right gripper
    gc.control('both', 1.0)   # close both grippers
    gc.open('both')
    gc.close('left')
    gc.shutdown()

Joint range:
    0.0 = open
    1.0 = closed
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GripperController:
    OPEN   = 0.0
    CLOSED = 1.0

    def __init__(self, node: Node = None):  # node를 외부에서 받음
        if node is None:
            # 단독 실행 시 직접 init
            rclpy.init()
            self._node = Node('gripper_controller')
            self._own_node = True
        else:
            # 다른 코드에서 호출 시 node 재사용
            self._node = node
            self._own_node = False

            
        self._left_pub = self._node.create_publisher(
            JointTrajectory,
            '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory',
            10,
        )
        self._right_pub = self._node.create_publisher(
            JointTrajectory,
            '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory',
            10,
        )
        self._sub = self._node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
        )

        self._left_joints = [
            'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4',
            'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1',
        ]
        self._right_joints = [
            'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
            'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1',
        ]

        self._current_positions = {}

        # Wait for joint states
        self._node.get_logger().info('Waiting for /joint_states...')
        self._wait_for_joint_states()
        self._node.get_logger().info('GripperController ready!')

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._current_positions[name] = pos

    def _wait_for_joint_states(self):
    # executor가 이미 spin 중이므로 그냥 기다리면 됨
        timeout = 5.0
        start = time.time()
        while time.time() - start < timeout:
            if (
                'gripper_l_joint1' in self._current_positions
                and 'gripper_r_joint1' in self._current_positions
            ):
                return
            time.sleep(0.1)

    def _send(self, side: str, gripper_position: float):
        gripper_position = max(self.OPEN, min(self.CLOSED, gripper_position))

        if side == 'left':
            joint_names = self._left_joints
            publisher = self._left_pub
        elif side == 'right':
            joint_names = self._right_joints
            publisher = self._right_pub
        else:
            self._node.get_logger().error("side must be 'left' or 'right'")
            return

        positions = []
        for joint in joint_names:
            if 'gripper' in joint:
                positions.append(float(gripper_position))
            else:
                positions.append(float(self._current_positions.get(joint, 0.0)))

        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        msg.points.append(point)

        for _ in range(10):
            publisher.publish(msg)
            time.sleep(0.05)

        self._node.get_logger().info(f'gripper [{side}] → {gripper_position:.2f}')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def control(self, side: str, position: float):
        """Set gripper position. side: 'left' | 'right' | 'both', position: 0.0~1.0"""
        if side == 'both':
            self._send('left', position)
            self._send('right', position)
        else:
            self._send(side, position)

    def open(self, side: str = 'both'):
        """Open gripper. side: 'left' | 'right' | 'both'"""
        self.control(side, self.OPEN)

    def close(self, side: str = 'both'):
        """Close gripper. side: 'left' | 'right' | 'both'"""
        self.control(side, self.CLOSED)

    def shutdown(self):
        self._node.destroy_node()
        if self._own_node:
            rclpy.shutdown()