"""The pick & place task as a list of planned steps.

All TCP poses are top-down (z_tcp = -Z0) with ONE fixed yaw psi, so from the grasp
to the release the gripper - and therefore Box B - never changes orientation.

  #   step              kind        motion                                  carrying
  1   open              gripper     2F-85 fully open
  2   to_ready          joint PTP   current -> ready pose
  3   to_above_b        joint PTP   ready -> above Box B at z_safe
  4   lower             Cartesian   straight down to the pre-grasp height (fast)
  5   approach          Cartesian   straight down to the grasp pose (slow)
  6   grasp             gripper     close on Box B with grip_force
  7   lift              Cartesian   straight up to z_safe                   yes
  8   transfer          Cartesian   arc around the base to above Box A      yes
  9   descend           Cartesian   straight down to the place pose         yes
  10  release           gripper     open
  11  retreat           Cartesian   straight up (+approach_height)
  12  to_ready_end      joint PTP   back to the ready pose

Free (joint-space) moves only constrain their END points: in between, the TCP follows
whatever curve the joint interpolation gives. So they start and end at z_safe, and every
segment except the deliberate descend/retreat at Box A is checked against Box A: the
gripper's top-view footprint (a rectangle, 2F-85 outer size, turned with the tool's yaw)
plus a margin must not overlap Box A unless it is high enough above it. A plan that would
hit Box A is rejected.

Grasp yaw psi: a cube looks the same every 90 deg, so psi = yaw_B + k*90 deg. The four
candidates are tried in order of |q6| at the grasp and place poses (joint 6 has to turn
by the transfer's azimuth change to keep psi fixed in the world); the first one whose
whole plan is feasible wins (e.g. a yaw whose open fingers would reach over Box A next
to Box B is skipped).

Grasp height (Robotiq 2F-85): the pads move on an arc, so the pad centre's distance from
the flange depends on the opening. The TCP (gripper_tcp) is the pad centre at the
reference width; for Box B's width the grasp TCP is shifted so the pad centre is at the
box's mid-height, and raised if the fingertips would come closer than FLOOR_CLEARANCE to
the ground.
"""
import math
from dataclasses import dataclass

import numpy as np
import yaml

from gp7_kinematics import GP7Kinematics, GripperModel, top_down_pose

from .paths import CylindricalPath, LinePath
from .planner import JointLimits, MotionPlanner, PlanningError, Trajectory


@dataclass
class Step:
    name: str
    kind: str                         # 'arm' or 'gripper'
    trajectory: Trajectory = None     # for 'arm'
    theta: float = None               # for 'gripper': knuckle reference [rad]
    theta_linkage: float = None       # for 'gripper': inner knuckle / finger-tip reference [rad]
    width: float = None               # for 'gripper': pad gap the step aims for [m]
    duration: float = 0.0
    carrying: bool = False


@dataclass
class PickPlacePlan:
    steps: list
    psi: float                        # grasp/carry yaw [rad]
    grasp_tcp: np.ndarray
    place_tcp: np.ndarray

    @property
    def arm_steps(self):
        return [s for s in self.steps if s.kind == 'arm']

    @property
    def duration(self):
        return sum(s.duration for s in self.steps)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_planner(cfg, kin=None):
    kin = kin or GP7Kinematics.from_config(tool='tcp')
    jl = cfg['joint_limits']
    limits = JointLimits.from_config(jl['speed_scale'], jl['accel_time'], jl['jerk_time'])
    return MotionPlanner(kin, limits, dt=cfg['planner']['dt'], profile=cfg['planner']['profile'])


CLEARANCE = 0.03              # m, required gap above / beside Box A
FLOOR_CLEARANCE = 0.003       # m, fingertips above the ground at the grasp


def _overlap(c1, axes1, half1, c2, axes2, half2):
    """Separating-axis test for two rectangles in the plane (centre, 2 unit axes, half sizes)."""
    d = c2 - c1
    for ax in (*axes1, *axes2):
        r1 = sum(h * abs(ax @ u) for h, u in zip(half1, axes1))
        r2 = sum(h * abs(ax @ u) for h, u in zip(half2, axes2))
        if abs(ax @ d) > r1 + r2:
            return False
    return True


def check_box_a_clearance(traj, task, kin, gripper, carrying_size=0.0):
    """Raise PlanningError if the gripper (or the carried box) comes near Box A.

    Top view: the 2F-85 is a rectangle (outer size, fingers open) around the TCP axis,
    turned with the tool's yaw at every sample; if it overlaps Box A grown by CLEARANCE,
    its lowest point (fingertips, or the carried box's bottom) must be CLEARANCE above Box A.
    """
    a = task['box_a']
    c_a, axes_a = np.array([a['x'], a['y']]), (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    half_a = (0.15 + CLEARANCE, 0.15 + CLEARANCE)
    half_g = (gripper.outer_half_side, gripper.outer_half_closing)    # along tool X, tool Y
    below = max(gripper.tip_below_tcp(gripper.open_width), gripper.tip_below_tcp(task['box_b_size']),
                carrying_size)
    worst = None
    for k in range(0, len(traj.t), 2):
        p = traj.tcp[k]
        if p[2] - below - a['height'] >= CLEARANCE:
            continue                                    # high enough whatever the footprint
        R = kin.fk(traj.q[k])[:3, :3]
        axes_g = []
        for col in (0, 1):
            u = R[:2, col]
            n = np.linalg.norm(u)
            axes_g.append(u / n if n > 1e-9 else np.array([1.0, 0.0]))
        if _overlap(p[:2], axes_g, half_g, c_a, axes_a, half_a):
            gap = p[2] - below - a['height']
            if worst is None or gap < worst[0]:
                worst = (gap, p)
    if worst is not None:
        raise PlanningError(f'{traj.name}: gripper passes {worst[0] * 1000:.0f} mm above Box A '
                            f'(needs {CLEARANCE * 1000:.0f} mm) at TCP {np.round(worst[1], 3)}')


def yaw_candidates(kin, box_yaw, grasp_xyz, place_xyz, q_ref):
    """psi in {box_yaw + k*90deg}, reachable at grasp and place, sorted by max |q6| there."""
    out = []
    for k in range(4):
        psi = math.atan2(math.sin(box_yaw + k * math.pi / 2), math.cos(box_yaw + k * math.pi / 2))
        try:
            qg = kin.ik_closest(top_down_pose(*grasp_xyz, yaw=psi), q_ref).q
            qp = kin.ik_closest(top_down_pose(*place_xyz, yaw=psi), qg).q
        except Exception:
            continue
        out.append((max(abs(qg[5]), abs(qp[5])), psi))
    if not out:
        raise PlanningError('no yaw gives a reachable grasp and place pose')
    return [psi for _, psi in sorted(out)]


def grasp_height(gripper, box_bottom_z, size):
    """TCP height for grasping a cube of edge `size` standing at box_bottom_z: pad centre at
    the box's mid-height, fingertips at least FLOOR_CLEARANCE above the ground (z = 0)."""
    z = box_bottom_z + size / 2.0 + gripper.pad_center_below_tcp(size)
    return max(z, gripper.tip_below_tcp(size) + FLOOR_CLEARANCE)


def plan_pick_place(planner, cfg, box_b, q_start, profile=None, gripper=None):
    """box_b = (x, y, z_bottom, yaw) of Box B; q_start = current joints.

    Tries the grasp yaws in order of preference (yaw_candidates); returns the first
    feasible plan, else raises the first yaw's PlanningError.
    """
    task = cfg['task']
    kin = planner.kin
    gripper = gripper or GripperModel.from_config()
    bx, by, bz, byaw = box_b
    size = task['box_b_size']
    ax, ay, ah = task['box_a']['x'], task['box_a']['y'], task['box_a']['height']
    grasp = np.array([bx, by, grasp_height(gripper, bz, size)])
    # Box B is carried with its mid-height at the pad centre: place it place_clearance above Box A
    place = np.array([ax, ay, ah + size / 2.0 + gripper.pad_center_below_tcp(size) + task['place_clearance']])
    if task['z_safe'] - size <= ah + 0.03:
        raise PlanningError('z_safe too low: Box B would not clear Box A')
    q_ready = kin.ik_closest(top_down_pose(*task['ready_tcp']), q_start).q
    errors = []
    for psi in yaw_candidates(kin, byaw, grasp, place, q_ready):
        try:
            return _plan_with_yaw(planner, cfg, gripper, psi, grasp, place, q_start, q_ready, profile)
        except PlanningError as e:
            errors.append(f'yaw {math.degrees(psi):.0f} deg: {e}')
    raise PlanningError('; '.join(errors))


def _plan_with_yaw(planner, cfg, gripper, psi, grasp, place, q_start, q_ready, profile):
    task, cart = cfg['task'], cfg['cartesian']
    kin = planner.kin
    size = task['box_b_size']
    up = np.array([0.0, 0.0, task['approach_height']])
    lift_top = np.array([grasp[0], grasp[1], task['z_safe']])
    place_top = np.array([place[0], place[1], task['z_safe']])
    R = top_down_pose(0, 0, 0, yaw=psi)[:3, :3]
    try:
        q_above_b = kin.ik_closest(top_down_pose(*lift_top, yaw=psi), q_ready).q
    except Exception as e:
        raise PlanningError(f'pose above Box B unreachable: {e}') from e

    ap, tr = cart['approach'], cart['transfer']
    kw = {'profile': profile}
    steps = []
    theta = [gripper.theta_open]                        # current knuckle reference

    def arm(traj, carrying=False, check=True):
        traj.carrying = carrying
        if check:
            check_box_a_clearance(traj, task, kin, gripper, size if carrying else 0.0)
        steps.append(Step(traj.name, 'arm', trajectory=traj, duration=traj.duration, carrying=carrying))
        return traj.q_end

    def grip(name, th, th_linkage, width, carrying=False):
        steps.append(Step(name, 'gripper', theta=th, theta_linkage=th_linkage, width=width,
                          duration=gripper.move_time(theta[0], th), carrying=carrying))
        theta[0] = th

    # grasp: knuckles squeeze past the contact angle (grip force), linkage at contact (pads parallel)
    th_grasp, th_contact = gripper.grasp_theta(size, task['grip_force'])

    theta[0] = gripper.theta_closed                     # unknown start: allow a full stroke
    grip('open', gripper.theta_open, gripper.theta_open, gripper.open_width)
    q = arm(planner.plan_joint(q_start, q_ready, 'to_ready', **kw))
    q = arm(planner.plan_joint(q, q_above_b, 'to_above_b', **kw))
    q = arm(planner.plan_cartesian(LinePath(lift_top, grasp + up, R), tr['v'], tr['a'], tr['j'], q,
                                   'lower', **kw))
    q = arm(planner.plan_cartesian(LinePath(grasp + up, grasp, R), ap['v'], ap['a'], ap['j'], q,
                                   'approach', **kw))
    grip('grasp', th_grasp, th_contact, size)
    q = arm(planner.plan_cartesian(LinePath(grasp, lift_top, R), tr['v'], tr['a'], tr['j'], q,
                                   'lift', carrying=True, **kw), carrying=True)
    q = arm(planner.plan_cartesian(CylindricalPath(lift_top, place_top, R), tr['v'], tr['a'], tr['j'],
                                   q, 'transfer', carrying=True, **kw), carrying=True)
    # descend / retreat touch Box A on purpose: no clearance check
    q = arm(planner.plan_cartesian(LinePath(place_top, place, R), ap['v'], ap['a'], ap['j'], q,
                                   'descend', carrying=True, **kw), carrying=True, check=False)
    grip('release', gripper.theta_open, gripper.theta_open, gripper.open_width)
    retreat_top = place + np.array([0.0, 0.0, max(task['approach_height'], task['z_safe'] - place[2])])
    q = arm(planner.plan_cartesian(LinePath(place, retreat_top, R), ap['v'], ap['a'], ap['j'], q,
                                   'retreat', **kw), check=False)
    arm(planner.plan_joint(q, q_ready, 'to_ready_end', **kw))
    return PickPlacePlan(steps, psi, grasp, place)
