"""Node that moves the GP7 to end-effector poses published on a topic.

    ros2 run gp7_control pose_goal_node
    ros2 topic pub --once /gp7/tcp_goal geometry_msgs/msg/PoseStamped \\
      "{header: {frame_id: base_link}, pose: {position: {x: 0.55, y: -0.3, z: 0.2},
        orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}}}"

The pose is the gripper TCP in base_link (= world, the robot stands at the origin).
Orientation (x=1, y=0, z=0, w=0) is the top-down gripper (180 deg about X).
Each goal: IK closest to the current joints -> arm_controller joint-space move.
Results are logged and published on /gp7/tcp_goal/result (std_msgs/String).
Goals that arrive while a move is running are rejected (logged).
"""
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from gp7_kinematics import make_pose

from .commander import GP7Commander, quat_to_matrix

ACCEPTED_FRAMES = ('', 'base_link', 'world')


class PoseGoalNode(GP7Commander):
    def __init__(self):
        super().__init__('pose_goal_node')
        self.speed_scale = float(self.declare_parameter('speed_scale', 0.25).value)
        self.result_pub = self.create_publisher(String, '/gp7/tcp_goal/result', 10)
        self.create_subscription(PoseStamped, '/gp7/tcp_goal', self._on_goal, 1)
        self._busy = threading.Lock()
        self._pending = None
        self.get_logger().info('waiting for TCP pose goals on /gp7/tcp_goal')

    def _on_goal(self, msg):
        if msg.header.frame_id not in ACCEPTED_FRAMES:
            self._report(False, f"frame '{msg.header.frame_id}' not supported (use base_link)")
            return
        if self._busy.locked():
            self._report(False, 'busy with the previous goal - goal ignored')
            return
        p, o = msg.pose.position, msg.pose.orientation
        self._pending = make_pose([p.x, p.y, p.z], quat_to_matrix(o.x, o.y, o.z, o.w))

    def step(self):
        """Run a pending goal (outside the subscription callback, so we can spin inside)."""
        if self._pending is None:
            return
        T, self._pending = self._pending, None
        with self._busy:
            self.get_logger().info(f'goal: TCP at {np.round(T[:3, 3], 4)}')
            ok, msg, sol = self.move_to_pose(T, speed_scale=self.speed_scale)
            if sol is not None:
                msg = f'q = {np.round(np.degrees(sol.q), 2)} deg; ' + msg
            self._report(ok, msg)

    def _report(self, ok, msg):
        text = ('OK   ' if ok else 'FAIL ') + msg
        # rclpy fixes the severity per call site, so use two separate calls
        if ok:
            self.get_logger().info(text)
        else:
            self.get_logger().error(text)
        self.result_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = PoseGoalNode()
    try:
        # The commander spins the same (global) executor while a move runs.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.step()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
