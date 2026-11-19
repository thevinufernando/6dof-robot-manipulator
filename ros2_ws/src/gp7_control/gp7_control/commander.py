"""GP7Commander: send the arm to an end-effector pose (IK + arm_controller) and drive the gripper.

    pose (4x4, base frame) --IK (closest to current joints)--> q_goal
         --FollowJointTrajectory--> arm_controller (velocity interface) --> Gazebo

This is a JOINT-SPACE move: the controller interpolates from the current joints to
q_goal with a cubic spline (zero velocity at both ends), so the TCP reaches the goal
pose exactly but its path and orientation in between are not controlled. Cartesian
(straight-line, fixed-orientation) motion is the trajectory planner's job.

Move duration (when not given): a cubic spline with zero end velocities peaks at
1.5 * dq / T, so T = max_i 1.5 * |dq_i| / (speed_scale * v_max_i), at least min_duration.
"""
import math
import threading
import time

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gp7_kinematics import (JOINT_NAMES, GP7Kinematics, GripperModel, IKError, find_config_dir,
                            pose_error)


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _duration_msg(t):
    sec = int(t)
    return Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))


class GP7Commander(Node):
    """Thin ROS layer around gp7_kinematics and the two trajectory controllers."""

    def __init__(self, node_name='gp7_commander', tool='tcp'):
        super().__init__(node_name)
        self.kin = GP7Kinematics.from_config(tool=tool)
        cfg = find_config_dir()
        with open(f'{cfg}/joint_limits.yaml') as f:
            lim = yaml.safe_load(f)
        self.v_max = np.radians([lim[n]['velocity_deg_s'] for n in JOINT_NAMES])
        # Robotiq 2F-85: one knuckle angle -> six joint references (software mimic)
        self.grp = GripperModel.from_config(cfg)

        self._js = None
        self._js_event = threading.Event()
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.arm = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.grip = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

    # ------------------------------------------------------------------ state
    def _on_js(self, msg):
        self._js = dict(zip(msg.name, msg.position))
        self._js_event.set()

    def wait_for_state(self, timeout=10.0):
        """Block (spinning) until /joint_states has arrived."""
        # wall-clock timeout: with use_sim_time the ROS clock reads 0 until /clock arrives
        end = time.monotonic() + timeout
        while not self._js_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > end:
                raise TimeoutError('no /joint_states - is the simulation running?')

    def current_q(self, fresh=True):
        if fresh:
            self._js_event.clear()
        self.wait_for_state()
        return np.array([self._js[n] for n in JOINT_NAMES])

    def current_gripper(self):
        """Measured positions of the six gripper joints (GripperModel.joint_names order)."""
        self.wait_for_state()
        return [self._js[n] for n in self.grp.joint_names]

    def current_knuckle(self):
        """Measured knuckle angle theta [rad] (mean of both knuckles)."""
        q = dict(zip(self.grp.joint_names, self.current_gripper()))
        return 0.5 * (q['robotiq_85_left_knuckle_joint'] - q['robotiq_85_right_knuckle_joint'])

    def current_width(self):
        """Pad gap [m] from the measured knuckle angle."""
        return self.grp.width_for_theta(self.current_knuckle())

    # ------------------------------------------------------------------ arm
    def plan_pose(self, T_goal, q_start=None):
        """IK for the goal pose, closest to q_start (default: current joints)."""
        q_start = self.current_q() if q_start is None else q_start
        return self.kin.ik_closest(T_goal, q_ref=q_start)          # raises IKError

    def auto_duration(self, q0, q1, speed_scale=0.25, min_duration=2.0):
        peak = 1.5 * np.abs(np.asarray(q1) - np.asarray(q0)) / (speed_scale * self.v_max)
        return float(max(min_duration, peak.max()))

    def move_joints(self, q_goal, duration=None, speed_scale=0.25, timeout_margin=10.0):
        """Send one joint-space goal to arm_controller and wait. Returns (ok, message)."""
        q0 = self.current_q()
        duration = duration or self.auto_duration(q0, q_goal, speed_scale)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(JOINT_NAMES))
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(v) for v in q_goal], velocities=[0.0] * 6,
            time_from_start=_duration_msg(duration))]
        return self._run(self.arm, goal, duration + timeout_margin, 'arm_controller')

    def follow_trajectory(self, t, q, qd=None, qdd=None, timeout_margin=10.0):
        """Send a sampled joint trajectory (from the planner) to arm_controller and wait.

        t: (N,) times from the start [s]; q, qd, qdd: (N,6). With positions, velocities
        and accelerations the controller interpolates with quintic splines between the
        samples and feeds the velocity forward. A first sample at t = 0 (= the current
        state) is dropped: the controller always starts from where the arm actually is.
        """
        t = np.asarray(t, float)
        keep = t > 1e-9
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(JOINT_NAMES))
        for k in np.where(keep)[0]:
            p = JointTrajectoryPoint(positions=[float(v) for v in q[k]],
                                     time_from_start=_duration_msg(float(t[k])))
            if qd is not None:
                p.velocities = [float(v) for v in qd[k]]
            if qdd is not None:
                p.accelerations = [float(v) for v in qdd[k]]
            goal.trajectory.points.append(p)
        if not goal.trajectory.points:
            return True, 'arm_controller: nothing to do'
        return self._run(self.arm, goal, float(t[-1]) + timeout_margin, 'arm_controller')

    def move_to_pose(self, T_goal, duration=None, speed_scale=0.25):
        """IK + joint-space move to a TCP pose. Returns (ok, message, solution)."""
        q0 = self.current_q()
        try:
            sol = self.plan_pose(T_goal, q0)
        except IKError as e:
            return False, f'IK failed: {e}', None
        ok, msg = self.move_joints(sol.q, duration, speed_scale)
        if ok:
            dp, dr = pose_error(self.kin.fk(self.current_q()), T_goal)
            msg += f' | reached pose error: {dp * 1000:.2f} mm, {math.degrees(dr):.3f} deg'
        return ok, msg, sol

    # ------------------------------------------------------------------ gripper
    def gripper(self, theta, duration=None, theta_linkage=None):
        """Knuckle angle `theta` [rad]: 0 = open (84 mm) .. 0.8 = closed.

        Sends all six 2F-85 joints q_j = multiplier_j * theta (Gazebo cannot enforce the URDF
        mimic constraints, so every joint is driven). For a grasp the inner knuckles and
        finger tips get `theta_linkage` (the contact angle) instead - see GripperModel.
        """
        theta = self.grp.clamp(theta)
        if duration is None:
            duration = self.grp.move_time(self.current_knuckle(), theta)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(self.grp.joint_names))
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=self.grp.references(theta, theta_linkage), velocities=[0.0] * len(self.grp.joint_names),
            time_from_start=_duration_msg(duration))]
        ok, msg = self._run(self.grip, goal, duration + 5.0, 'gripper_controller')
        if ok:
            msg += (f' | knuckle {self.current_knuckle():.4f} rad (ref {theta:.4f}), '
                    f'pad gap {self.current_width() * 1000:.1f} mm')
        return ok, msg

    def open_gripper(self, duration=None):
        return self.gripper(self.grp.theta_open, duration)

    def close_gripper(self, duration=None):
        return self.gripper(self.grp.theta_closed, duration)

    def gripper_width(self, width, duration=None):
        """Pads `width` [m] apart (no object: position command)."""
        return self.gripper(self.grp.theta_for_width(width), duration)

    def grasp(self, width, force=None, duration=None):
        """Close on an object of `width` [m] with about `force` [N] per pad.

        The knuckle PID pushes with tau = Kp (theta_ref - theta) and the pad force is
        tau / lever, so the knuckle reference is set force * lever / Kp beyond the contact
        angle; the finger tips are held at the contact angle (pads parallel).
        """
        theta_ref, theta_contact = self.grp.grasp_theta(width, force)
        return self.gripper(theta_ref, duration, theta_linkage=theta_contact)

    # ------------------------------------------------------------------ internals
    def _run(self, client, goal, timeout, name):
        if not client.wait_for_server(timeout_sec=10.0):
            return False, f'{name} action server not available'
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            return False, f'{name} rejected the goal'
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout)
        if res_fut.result() is None:
            return False, f'{name} did not finish in time'
        res = res_fut.result().result
        if res.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, f'{name} failed: error_code {res.error_code} {res.error_string}'
        return True, f'{name}: done'
