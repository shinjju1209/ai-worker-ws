import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GripperControl(Node): # inherit from Node
    def __init__(self):
        super().__init__("gripper_control")

        self.left_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            10,
        )
        
        self.right_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            10,
        )
        
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        
        self.left_joints = [
            "arm_l_joint1",
            "arm_l_joint2",
            "arm_l_joint3",
            "arm_l_joint4",
            "arm_l_joint5",
            "arm_l_joint6",
            "arm_l_joint7",
            "gripper_l_joint1",
        ]

        self.right_joints = [
            "arm_r_joint1",
            "arm_r_joint2",
            "arm_r_joint3",
            "arm_r_joint4",
            "arm_r_joint5",
            "arm_r_joint6",
            "arm_r_joint7",
            "gripper_r_joint1",
        ]

        self.current_positions = {}

        # 0.0 = open
        # 1.0 = close
        self.open_position = 0.0
        self.close_position = 1.0

        self.get_logger().info("Gripper control node started")

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_positions[name] = pos

    def wait_for_joint_states(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if (
                "gripper_l_joint1" in self.current_positions
                and "gripper_r_joint1" in self.current_positions
            ):
                return

    def send_arm_with_gripper(self, side, gripper_position):
        if side == "left":
            joint_names = self.left_joints
            publisher = self.left_pub

        elif side == "right":
            joint_names = self.right_joints
            publisher = self.right_pub
        # control each arms with publisher
        else:
            self.get_logger().error("side must be left or right")
            return

        positions = []

        for joint in joint_names:
            if "gripper" in joint:
                positions.append(float(gripper_position))
            else:
                positions.append(
                    float(self.current_positions.get(joint, 0.0))
                )
            # maintain current position for arm joints, only change gripper joint
        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points.append(point)

        for _ in range(10):
            publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        self.get_logger().info(
            f"{side} gripper command sent: {gripper_position}"
        )

    def open(self, side):
        if side == "both":
            self.send_arm_with_gripper("left", self.open_position)
            self.send_arm_with_gripper("right", self.open_position)

        else:
            self.send_arm_with_gripper(side, self.open_position)

    def close(self, side):
        if side == "both":
            self.send_arm_with_gripper("left", self.close_position)
            self.send_arm_with_gripper("right", self.close_position)

        else:
            self.send_arm_with_gripper(side, self.close_position)


def main():
    rclpy.init() # initialize ROS 2

    node = GripperControl() # create node instance

    print("Waiting for /joint_states...")
    node.wait_for_joint_states()

    print("Ready.")
    print("Commands:")
    print(" open left")
    print(" close left")
    print(" open right")
    print(" close right")
    print(" open both")
    print(" close both")
    print(" set left 0.5")
    print(" set right 0.3")
    print(" set both 0.7")
    print(" q")

    try:
        while rclpy.ok():
            cmd = input("gripper> ").strip().split()

            if not cmd:
                continue

            if cmd[0] == "q":
                break

            # set left 0.5
            if cmd[0] == "set":
                if len(cmd) != 3:
                    print("Use: set left/right/both position")
                    continue

                side = cmd[1]

                try:
                    position = float(cmd[2])

                except ValueError:
                    print("Position must be a number")
                    continue

                if position < 0.0:
                    position = 0.0

                if position > 1.0:
                    position = 1.0

                if side == "both":
                    node.send_arm_with_gripper("left", position)
                    node.send_arm_with_gripper("right", position)

                else:
                    node.send_arm_with_gripper(side, position)

                continue

            if len(cmd) != 2:
                print("Use:")
                print(" open left")
                print(" close left")
                print(" set left 0.5")
                continue

            action = cmd[0]
            side = cmd[1]

            if action == "open":
                node.open(side)

            elif action == "close":
                node.close(side)

            else:
                print("action must be open or close")

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()