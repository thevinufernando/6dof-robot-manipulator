"""The pick & place task as a list of planned steps.

All TCP poses are top-down (z_tcp = -Z0) with ONE fixed yaw psi, so from the grasp
to the release the gripper - and therefore Box B - never changes orientation.

  #   step              kind        motion                                  carrying
  1   open              gripper     fingers fully open
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
segment except the deliberate descend/retreat at Box A is checked against Box A
(inflated by the gripper's half width + a margin): a plan that would hit it is rejected.

Grasp yaw psi: a cube looks the same every 90 deg, so psi = yaw_B + k*90 deg; the k
that keeps joint 6 closest to zero at both the grasp and the place pose is chosen
(joint 6 has to turn by the transfer's azimuth change to keep psi fixed in the world).
"""
import math
from dataclasses import dataclass

import numpy as np
import yaml

from gp7_kinematics import GP7Kinematics, find_config_dir, top_down_pose

from .paths import CylindricalPath, LinePath
from .planner import JointLimits, MotionPlanner, PlanningError, Trajectory


@dataclass
class Step:
    name: str
    kind: str                         # 'arm' or 'gripper'
    trajectory: Trajectory = None     # for 'arm'
    finger_position: float = None     # for 'gripper' [m per finger]
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


GRIPPER_HALF_WIDTH = 0.065    # m, fingers/body around the TCP axis (open fingers at +-60.5 mm)
FINGERTIP_BELOW_TCP = 0.02    # m, pad bottom below the TCP
CLEARANCE = 0.03              # m, required gap above / beside Box A


def check_box_a_clearance(traj, task, carrying_size=0.0):
    """Raise PlanningError if the gripper (or the carried box) comes near Box A.

    The gripper is modelled as a vertical column of radius GRIPPER_HALF_WIDTH around the
    TCP whose lowest point is the finger tips (or the carried box's bottom).
    """
    a = task['box_a']
    half = 0.15 + GRIPPER_HALF_WIDTH + CLEARANCE
    p = traj.tcp
    over = (np.abs(p[:, 0] - a['x']) < half) & (np.abs(p[:, 1] - a['y']) < half)
    if not over.any():
        return
    bottom = p[over, 2] - max(FINGERTIP_BELOW_TCP, carrying_size)
    gap = bottom.min() - a['height']
    if gap < CLEARANCE:
        k = np.where(over)[0][np.argmin(bottom)]
        raise PlanningError(f'{traj.name}: gripper passes {gap * 1000:.0f} mm above Box A '
                            f'(needs {CLEARANCE * 1000:.0f} mm) at TCP {np.round(p[k], 3)}')


def choose_yaw(kin, box_yaw, grasp_xyz, place_xyz, q_ref):
    """psi in {box_yaw + k*90deg} with the smallest |q6| over the grasp and place poses."""
    best = None
    for k in range(4):
        psi = math.atan2(math.sin(box_yaw + k * math.pi / 2), math.cos(box_yaw + k * math.pi / 2))
        try:
            qg = kin.ik_closest(top_down_pose(*grasp_xyz, yaw=psi), q_ref).q
            qp = kin.ik_closest(top_down_pose(*place_xyz, yaw=psi), qg).q
        except Exception:
            continue
        cost = max(abs(qg[5]), abs(qp[5]))
        if best is None or cost < best[0]:
            best = (cost, psi)
    if best is None:
        raise PlanningError('no yaw gives a reachable grasp and place pose')
    return best[1]


def plan_pick_place(planner, cfg, box_b, q_start, profile=None):
    """box_b = (x, y, z_bottom, yaw) of Box B; q_start = current joints."""
    task, cart = cfg['task'], cfg['cartesian']
    kin = planner.kin
    bx, by, bz, byaw = box_b
    half = task['box_b_size'] / 2.0
    ax, ay, ah = task['box_a']['x'], task['box_a']['y'], task['box_a']['height']
    grasp = np.array([bx, by, bz + half])
    place = np.array([ax, ay, ah + half + task['place_clearance']])
    up = np.array([0.0, 0.0, task['approach_height']])
    lift_top = np.array([bx, by, task['z_safe']])
    place_top = np.array([ax, ay, task['z_safe']])
    if task['z_safe'] - half <= ah + 0.03:
        raise PlanningError('z_safe too low: Box B would not clear Box A')

    q_ready = kin.ik_closest(top_down_pose(*task['ready_tcp']), q_start).q
    psi = choose_yaw(kin, byaw, grasp, place, q_ready)
    R = top_down_pose(0, 0, 0, yaw=psi)[:3, :3]
    try:
        q_above_b = kin.ik_closest(top_down_pose(*lift_top, yaw=psi), q_ready).q
    except Exception as e:
        raise PlanningError(f'pose above Box B unreachable: {e}') from e

    ap, tr = cart['approach'], cart['transfer']
    kw = {'profile': profile}
    steps = []

    def arm(traj, carrying=False, check=True):
        traj.carrying = carrying
        if check:
            check_box_a_clearance(traj, task, task['box_b_size'] if carrying else 0.0)
        steps.append(Step(traj.name, 'arm', trajectory=traj, duration=traj.duration, carrying=carrying))
        return traj.q_end

    def gripper(name, pos, carrying=False):
        steps.append(Step(name, 'gripper', finger_position=pos, duration=task['gripper_time'],
                          carrying=carrying))

    with open(f'{find_config_dir()}/gripper.yaml') as f:
        stroke = yaml.safe_load(f)['stroke']
    # finger reference = contact position + force / Kp  (the finger PID then pushes with grip_force)
    grip_target = (2 * stroke - task['box_b_size']) / 2 + task['grip_force'] / task['finger_p_gain']

    gripper('open', 0.0)
    q = arm(planner.plan_joint(q_start, q_ready, 'to_ready', **kw))
    q = arm(planner.plan_joint(q, q_above_b, 'to_above_b', **kw))
    q = arm(planner.plan_cartesian(LinePath(lift_top, grasp + up, R), tr['v'], tr['a'], tr['j'], q,
                                   'lower', **kw))
    q = arm(planner.plan_cartesian(LinePath(grasp + up, grasp, R), ap['v'], ap['a'], ap['j'], q,
                                   'approach', **kw))
    gripper('grasp', grip_target)
    q = arm(planner.plan_cartesian(LinePath(grasp, lift_top, R), tr['v'], tr['a'], tr['j'], q,
                                   'lift', carrying=True, **kw), carrying=True)
    q = arm(planner.plan_cartesian(CylindricalPath(lift_top, place_top, R), tr['v'], tr['a'], tr['j'],
                                   q, 'transfer', carrying=True, **kw), carrying=True)
    # descend / retreat touch Box A on purpose: no clearance check
    q = arm(planner.plan_cartesian(LinePath(place_top, place, R), ap['v'], ap['a'], ap['j'], q,
                                   'descend', carrying=True, **kw), carrying=True, check=False)
    gripper('release', 0.0)
    retreat_top = place + np.array([0.0, 0.0, max(task['approach_height'], task['z_safe'] - place[2])])
    q = arm(planner.plan_cartesian(LinePath(place, retreat_top, R), ap['v'], ap['a'], ap['j'], q,
                                   'retreat', **kw), check=False)
    arm(planner.plan_joint(q, q_ready, 'to_ready_end', **kw))
    return PickPlacePlan(steps, psi, grasp, place)
