"""Pick Box B and place it on the centre of Box A's top face (Gazebo, ros2_control).

    ros2 launch gp7_control pick_place.launch.py                   # plan + execute
    ros2 launch gp7_control pick_place.launch.py dry_run:=true     # plan + RViz preview only
    ros2 launch gp7_control pick_place.launch.py profile:=trapezoidal log_dir:=runs

1. Read Box B's pose (/box_b/pose, bridged from Gazebo) and the joints (/joint_states).
2. Plan the whole task with gp7_motion (joint-space free moves, constant-orientation
   Cartesian moves while carrying, S-curve or trapezoidal velocity profiles).
   Nothing moves if any segment is infeasible.
3. Publish the planned TCP path and the scene for RViz (/pick_place/planned_path,
   /pick_place/markers).
4. Execute the steps in order: arm segments -> arm_controller (FollowJointTrajectory with
   positions, velocities, accelerations), gripper steps -> gripper_controller.
   Before each arm segment the measured joints are compared with the planned start.
   After the grasp, the 2F-85 knuckle angle is checked: the pads must stop on the box
   (pad gap = box width +-4 mm), else nothing was grasped.
5. Monitor Box B the whole time and report its maximum tilt while carried and the final
   placement error; optionally save CSV logs (log_dir) of the planned and measured motion.
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from gp7_kinematics import JOINT_NAMES
from gp7_motion.planner import PlanningError
from gp7_motion.sequence import load_config, make_planner, plan_pick_place

from .commander import GP7Commander

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _default_config():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('gp7_motion'), 'config', 'pick_place.yaml')


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _tilt_deg(q):
    """Angle between the box's z axis and the world's z axis."""
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))


class PickPlaceNode(GP7Commander):
    def __init__(self):
        super().__init__('pick_place_node')
        self.cfg_path = self.declare_parameter('config', _default_config()).value
        self.profile = self.declare_parameter('profile', '').value or None
        self.dry_run = bool(self.declare_parameter('dry_run', False).value)
        self.log_dir = self.declare_parameter('log_dir', '').value
        self.box_a_xy = [float(self.declare_parameter('box_a_x', 0.5).value),
                         float(self.declare_parameter('box_a_y', 0.5).value)]

        self.cfg = load_config(self.cfg_path)
        self.cfg['task']['box_a']['x'], self.cfg['task']['box_a']['y'] = map(float, self.box_a_xy)
        self.planner = make_planner(self.cfg, self.kin)

        # Logs use SIMULATION time (message stamps), the clock the controllers run on.
        # Wall-clock stamps would show a fake tracking lag whenever Gazebo runs slower
        # than real time.
        self.sim_t = None                     # latest /joint_states stamp [s]
        self.t0 = None                        # common time origin of all logs (sim time)
        self.step_log = []                    # (name, kind, start, end) of executed steps
        self.box = None                       # latest Box B pose message
        self.carrying = False
        self.box_log = []                     # (sim time, x, y, z, tilt, carrying)
        self.joint_log = []                   # (sim time, q..., qd...)
        self.create_subscription(PoseStamped, '/box_b/pose', self._on_box, 10)
        self.create_subscription(JointState, '/joint_states', self._on_js_log, 10)
        self.path_pub = self.create_publisher(Path, '/pick_place/planned_path', LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, '/pick_place/markers', LATCHED)

    # ------------------------------------------------------------------ callbacks
    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def now(self):
        """Simulation time [s] (latest /joint_states stamp), wall clock until one arrived."""
        return self.sim_t if self.sim_t is not None else time.monotonic()

    def _on_box(self, msg):
        self.box = msg
        p, o = msg.pose.position, msg.pose.orientation
        t = self._stamp(msg) or self.now()
        self.box_log.append((t, p.x, p.y, p.z, _tilt_deg(o), self.carrying))

    def _on_js_log(self, msg):
        if msg.header.stamp.sec or msg.header.stamp.nanosec:
            self.sim_t = self._stamp(msg)
        if not self.log_dir:
            return
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if all(n in pos for n in JOINT_NAMES):
            self.joint_log.append([self.now()] + [pos[n] for n in JOINT_NAMES]
                                  + [vel.get(n, float('nan')) for n in JOINT_NAMES])

    def wait_for_box(self, timeout=15.0):
        end = time.monotonic() + timeout
        while self.box is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.box is None:
            raise TimeoutError('no /box_b/pose - is the simulation (and its bridge) running?')
        return self.box

    def settle(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ------------------------------------------------------------------ visualisation
    def publish_plan(self, plan, box_b):
        path = Path()
        path.header.frame_id = 'base_link'
        for st in plan.arm_steps:
            for p in st.trajectory.tcp[::5]:
                ps = PoseStamped()
                ps.header.frame_id = 'base_link'
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        self.path_pub.publish(path)

        task = self.cfg['task']
        a = task['box_a']
        markers = MarkerArray()

        def cube(mid, name, x, y, size, h, yaw, rgba):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns, m.id, m.type, m.action = name, mid, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, h / 2
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x = m.scale.y = size
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            return m

        markers.markers.append(cube(0, 'box_a', a['x'], a['y'], 0.30, a['height'], 0.0, (0.60, 0.45, 0.25, 0.5)))
        markers.markers.append(cube(1, 'box_b', box_b[0], box_b[1], task['box_b_size'], task['box_b_size'],
                                    box_b[3], (0.85, 0.10, 0.10, 0.6)))
        for mid, (name, p) in enumerate([('grasp', plan.grasp_tcp), ('place', plan.place_tcp)], start=2):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns, m.id, m.type, m.action = name, mid, Marker.SPHERE, Marker.ADD
            m.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.02
            m.color.g, m.color.a = 0.8, 1.0
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    # ------------------------------------------------------------------ main sequence
    def run(self):
        q0 = self.current_q()
        end = time.monotonic() + 5.0                 # wait for a stamped /joint_states (sim clock)
        while self.sim_t is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.t0 = self.now()
        msg = self.wait_for_box()
        p, o = msg.pose.position, msg.pose.orientation
        box_b = (p.x, p.y, p.z, _yaw(o))
        self.get_logger().info(f'Box B at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) yaw '
                               f'{math.degrees(box_b[3]):.1f} deg; Box A at {self.box_a_xy}')
        try:
            plan = plan_pick_place(self.planner, self.cfg, box_b, q0, profile=self.profile,
                                   gripper=self.grp)
        except PlanningError as e:
            self.get_logger().error(f'planning failed, nothing moved: {e}')
            return False
        prof = self.profile or self.cfg['planner']['profile']
        self.get_logger().info(f'plan ({prof}): {len(plan.steps)} steps, {plan.duration:.1f} s, '
                               f'grasp yaw {math.degrees(plan.psi):.1f} deg')
        for st in plan.steps:
            extra = (f'{st.trajectory.kind:9s} {st.duration:5.2f} s' if st.kind == 'arm'
                     else f'gripper   {st.duration:5.2f} s -> pad gap {st.width * 1000:.0f} mm '
                          f'(knuckle {st.theta:.3f} rad)')
            self.get_logger().info(f'  {st.name:13s} {extra}{"  [carrying]" if st.carrying else ""}')
        self.publish_plan(plan, box_b)
        if self.log_dir:
            self.save_plan(plan)
        if self.dry_run:
            # latched topics live only as long as this node: keep it up for RViz
            self.get_logger().info('dry run: plan published to RViz, not executing (Ctrl-C to exit)')
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.2)
            return True

        try:
            ok = self.execute(plan)
        finally:
            self.carrying = False
            if self.log_dir:                 # also after an abort: that is when logs matter most
                self.save_logs()
        if ok:
            self.settle(1.0)
            self.report(plan)
        return ok

    def execute(self, plan):
        t_start = time.monotonic()
        for st in plan.steps:
            self.carrying = st.carrying
            t_step = self.now()
            if st.kind == 'gripper':
                # the plan budgets a full stroke for the first 'open' (start unknown); here the
                # opening is measured, so time it from the actual travel
                dur = None if st.name == 'open' else st.duration
                ok, text = self.gripper(st.theta, dur, theta_linkage=st.theta_linkage)
                self.settle(0.3)
                if ok and st.name == 'grasp':
                    size = self.cfg['task']['box_b_size']
                    if not self.grp.contact_ok(self.current_knuckle(), size):
                        ok, text = False, (f'grasp failed: pads stopped {self.current_width() * 1000:.1f} mm '
                                           f'apart, expected ~{size * 1000:.0f} mm (box not between the pads)')
            else:
                tr = st.trajectory
                dev = float(np.max(np.abs(self.current_q() - tr.q_start)))
                if dev > 0.05:
                    self.get_logger().error(f'{st.name}: arm is {dev:.3f} rad away from the planned start - abort')
                    return False
                ok, text = self.follow_trajectory(tr.t, tr.q, tr.qd, tr.qdd)
                if ok:
                    err = float(np.max(np.abs(self.current_q() - tr.q_end)))
                    text += f' | final joint error {math.degrees(err):.3f} deg'
            self.step_log.append((st.name, st.kind, t_step - self.t0, self.now() - self.t0))
            if ok:
                self.get_logger().info(f'[{time.monotonic() - t_start:6.2f} s] {st.name}: {text}')
            else:
                self.get_logger().error(f'{st.name}: {text} - abort')
                return False
        return True

    # ------------------------------------------------------------------ results
    def report(self, plan):
        task = self.cfg['task']
        a = task['box_a']
        carried = [r[4] for r in self.box_log if r[5]]
        p, o = self.box.pose.position, self.box.pose.orientation
        target = np.array([a['x'], a['y'], a['height']])
        err = np.array([p.x, p.y, p.z]) - target
        self.get_logger().info(
            f'RESULT: Box B max tilt while carried {max(carried) if carried else float("nan"):.3f} deg '
            f'({len(carried)} samples); final position ({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) -> '
            f'error {np.linalg.norm(err[:2]) * 1000:.1f} mm horizontal, {err[2] * 1000:+.1f} mm vertical '
            f'(target: centre of Box A top face {np.round(target, 3).tolist()}); final tilt {_tilt_deg(o):.3f} deg')

    def save_plan(self, plan):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'planned.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'segment', 'carrying'] + [f'q{i}' for i in range(1, 7)]
                       + [f'qd{i}' for i in range(1, 7)] + [f'qdd{i}' for i in range(1, 7)]
                       + ['tcp_x', 'tcp_y', 'tcp_z'])
            t0 = 0.0
            for st in plan.steps:
                if st.kind == 'arm':
                    tr = st.trajectory
                    for k in range(len(tr.t)):
                        w.writerow([f'{t0 + tr.t[k]:.4f}', tr.name, int(st.carrying)]
                                   + [f'{v:.6f}' for v in np.r_[tr.q[k], tr.qd[k], tr.qdd[k], tr.tcp[k]]])
                t0 += st.duration

    def save_logs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'measured_joints.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t'] + [f'q{i}' for i in range(1, 7)] + [f'qd{i}' for i in range(1, 7)])
            t0 = self.t0
            for r in self.joint_log:
                w.writerow([f'{r[0] - t0:.4f}'] + [f'{v:.6f}' for v in r[1:]])
        with open(os.path.join(self.log_dir, 'box_b.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y', 'z', 'tilt_deg', 'carrying'])
            for r in self.box_log:
                w.writerow([f'{r[0] - self.t0:.4f}'] + [f'{v:.6f}' for v in r[1:5]] + [int(r[5])])
        with open(os.path.join(self.log_dir, 'steps.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'kind', 't_start', 't_end'])
            for name, kind, a, b in self.step_log:
                w.writerow([name, kind, f'{a:.4f}', f'{b:.4f}'])
        self.get_logger().info(f'logs written to {os.path.abspath(self.log_dir)}')


def main():
    rclpy.init()
    node = PickPlaceNode()
    ok = False
    try:
        ok = node.run()
    except (TimeoutError, KeyboardInterrupt) as e:
        node.get_logger().error(str(e) or 'interrupted')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
