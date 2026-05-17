# Talks to MoveIt2's move_group node.
# Receives a clean Pose in base_link frame and handles all motion planning and execution.
# All other files feed into this one — nothing else touches MoveIt2 directly.
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread 
from pymoveit2 import MoveIt2

class MoveItClient:
    def __init__(self):
        rclpy.init()
        self.node = Node('moveit_client')
        self.callback_group = ReentrantCallbackGroup() #allows simultaneous programs to run
        self.executor = MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True) #creates a background thread looping forever. daemon =true means it dies when program exits
        self.executor_thread.start()

        #creates a moveit2 object
        self.moveit2 = MoveIt2(
            node=self.node, 
            joint_names=['arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3',
                'arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7'],
            base_link_name='base_link',
            end_effector_name='end_effector_r_link',
            group_name='arm_r',
            callback_group=self.callback_group,
        )

        #startup sleep - pausing the program for 1 second 
        time.sleep(1.0)

    def move_to_home(self, velocity_scaling=0.3, acceleration_scaling=0.3):
        self.moveit2.max_velocity = velocity_scaling
        self.moveit2.max_acceleration = acceleration_scaling
        self.moveit2.move_to_configuration([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = self.moveit2.wait_until_executed()
        time.sleep(0.2)  # 200ms
        return result

    def move_to_pose(self, pose, velocity_scaling=0.5, acceleration_scaling=0.5):
        self.moveit2.max_velocity = velocity_scaling
        self.moveit2.max_acceleration = acceleration_scaling
        self.moveit2.move_to_pose(pose=pose)
        result = self.moveit2.wait_until_executed()
        # Wait for joint states to reflect the completed trajectory before the next plan.
        # The controller reports done before the next /joint_states tick arrives (~10ms at
        # 100Hz), so planning for the next move would see a stale start state and get
        # STATUS_ABORTED with "deviates from current robot state more than 0.01".
        time.sleep(0.2)  # 200ms
        return result

    def shutdown(self):
        rclpy.shutdown()
