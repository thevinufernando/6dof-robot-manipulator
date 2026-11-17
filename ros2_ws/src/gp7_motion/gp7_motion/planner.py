"""Trajectory planner: joint-space (PTP) and Cartesian (constant-orientation) moves.

Output is a sampled joint trajectory (t, q, q_dot, q_ddot) at dt (default 10 ms)
that the joint_trajectory_controller can follow (it interpolates between the
samples with splines and uses q_dot as velocity feed-forward).

Joint-space move (plan_joint) - used when nothing is carried
    All joints share ONE normalised profile sigma(t): 0 -> 1, so they start and stop
    together and the motion is a straight line in joint space:
        q(t) = q0 + sigma(t) (q1 - q0)
    Joint i moves |dq_i| * sigma_dot, so sigma's limits are the tightest ratio
        v_sigma = min_i v_i / |dq_i|   (same for a and j)
    which makes the move as fast as the slowest joint allows, with every joint inside
    its own limits.

Cartesian move (plan_cartesian) - used for approach, lift, transfer, descend
    s(t) from the profile with the Cartesian limits (v, a, j of the TCP) ->
    pose T(s) on the path -> q = IK(T, closest to the previous sample, same
    shoulder/elbow/wrist branch) -> q_dot = J(q)^-1 [s_dot * tangent(s); 0, 0, 0]
    (angular velocity 0: orientation is constant) -> q_ddot by central differences.
    If any joint would exceed its speed / acceleration limit, the move is slowed down:
    stretching time by k divides speeds by k and accelerations by k^2, i.e. the
    Cartesian limits become (v/k, a/k^2, j/k^3); then it is re-planned.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from gp7_kinematics import JOINT_NAMES, find_config_dir

from .profiles import make_profile


class PlanningError(RuntimeError):
    pass


@dataclass
class JointLimits:
    lower: np.ndarray       # rad
    upper: np.ndarray       # rad
    v: np.ndarray           # rad/s   (scaled datasheet speeds)
    a: np.ndarray           # rad/s^2
    j: np.ndarray           # rad/s^3

    @classmethod
    def from_config(cls, speed_scale=0.3, accel_time=0.4, jerk_time=0.1, config_dir=None):
        """Datasheet speeds x speed_scale; a = v / accel_time; j = a / jerk_time.

        The datasheet gives no acceleration limits, so they are defined by the time
        a joint needs to reach its (scaled) top speed.
        """
        with open(f'{config_dir or find_config_dir()}/joint_limits.yaml') as f:
            lim = yaml.safe_load(f)
        v = speed_scale * np.radians([lim[n]['velocity_deg_s'] for n in JOINT_NAMES])
        a = v / accel_time
        return cls(np.radians([lim[n]['lower_deg'] for n in JOINT_NAMES]),
                   np.radians([lim[n]['upper_deg'] for n in JOINT_NAMES]),
                   v, a, a / jerk_time)


@dataclass
class Trajectory:
    """Sampled joint trajectory of one motion segment."""
    name: str
    kind: str                       # 'joint' or 'cartesian'
    t: np.ndarray                   # (N,)  s, starting at 0
    q: np.ndarray                   # (N,6) rad
    qd: np.ndarray                  # (N,6) rad/s
    qdd: np.ndarray                 # (N,6) rad/s^2
    tcp: np.ndarray                 # (N,3) TCP position from FK [m]
    orientation_error: np.ndarray   # (N,)  angle between the TCP orientation and the path's [rad]
    carrying: bool = False
    profile: object = None          # the scalar profile used (s for Cartesian, sigma for joint)
    path: object = None             # Cartesian path (None for joint moves)
    s: np.ndarray = None            # (N,4) s, s_dot, s_ddot, s_dddot of the profile
    stretch: float = 1.0            # time-stretch factor applied to meet joint limits
    info: dict = field(default_factory=dict)

    @property
    def duration(self):
        return float(self.t[-1])

    @property
    def q_start(self):
        return self.q[0]

    @property
    def q_end(self):
        return self.q[-1]


def _rotation_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


class MotionPlanner:
    def __init__(self, kin, limits, dt=0.01, profile='scurve',
                 max_step=0.2, wrist_margin=0.05):
        """kin: GP7Kinematics (tool = TCP); limits: JointLimits.
        max_step: largest joint change allowed between two samples [rad] (branch-jump guard).
        wrist_margin: minimum |sin q5| on Cartesian paths (keeps away from the wrist singularity).
        """
        self.kin, self.lim, self.dt, self.profile = kin, limits, dt, profile
        self.max_step, self.wrist_margin = max_step, wrist_margin

    # ------------------------------------------------------------------ helpers
    def _times(self, T):
        n = max(1, int(math.ceil(T / self.dt - 1e-9)))
        return np.linspace(0.0, T, n + 1) if T > 0 else np.zeros(1)

    def _check_joint_limits(self, q, name):
        lo = np.any(q < self.lim.lower - 1e-9, axis=0)
        hi = np.any(q > self.lim.upper + 1e-9, axis=0)
        if lo.any() or hi.any():
            bad = [JOINT_NAMES[i] for i in np.where(lo | hi)[0]]
            raise PlanningError(f'{name}: joint limit exceeded on {bad}')

    def _fk_samples(self, q):
        T = np.array([self.kin.fk(qk) for qk in q])
        return T[:, :3, 3], T[:, :3, :3]

    # ------------------------------------------------------------------ joint space
    def plan_joint(self, q0, q1, name='ptp', profile=None):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        dq = q1 - q0
        self._check_joint_limits(np.vstack([q0, q1]), name)
        moving = np.abs(dq) > 1e-9
        if not moving.any():
            prof = make_profile(profile or self.profile, 0.0, 1.0, 1.0, 1.0)
        else:
            adq = np.abs(dq[moving])
            v_s = np.min(self.lim.v[moving] / adq)
            a_s = np.min(self.lim.a[moving] / adq)
            j_s = np.min(self.lim.j[moving] / adq)
            prof = make_profile(profile or self.profile, 1.0, v_s, a_s, j_s)
        t = self._times(prof.duration)
        sig, sd, sdd, sddd = prof.sample(t)
        q = q0 + np.outer(sig, dq)
        tcp, R = self._fk_samples(q)
        return Trajectory(name, 'joint', t, q, np.outer(sd, dq), np.outer(sdd, dq), tcp,
                          np.full(len(t), np.nan), profile=prof,
                          s=np.stack([sig, sd, sdd, sddd], axis=1))

    # ------------------------------------------------------------------ Cartesian space
    def plan_cartesian(self, path, v, a, j, q_start, name='lin', carrying=False,
                       profile=None, max_iter=10):
        q_start = np.asarray(q_start, float)
        dp, dr = (np.linalg.norm(self.kin.fk(q_start)[:3, 3] - path.pose(0.0)[:3, 3]),
                  _rotation_angle(self.kin.fk(q_start)[:3, :3], path.R))
        if dp > 1e-5 or dr > 1e-5:
            raise PlanningError(f'{name}: q_start is not at the path start '
                                f'({dp * 1000:.2f} mm, {math.degrees(dr):.3f} deg off)')
        branch = self._branch(path.pose(0.0), q_start)

        k = 1.0
        for _ in range(max_iter):
            prof = make_profile(profile or self.profile, path.length, v / k, a / k ** 2, j / k ** 3)
            t = self._times(prof.duration)
            s, sd, sdd, sddd = prof.sample(t)
            q = self._track(path, s, q_start, branch, name)
            tang = path.tangent(s)
            qd = np.array([self.kin.joint_velocities(qk, sdk * tk) for qk, sdk, tk in zip(q, sd, tang)])
            qdd = np.gradient(qd, t, axis=0, edge_order=2) if len(t) > 2 else np.zeros_like(qd)
            ratio_v = float(np.max(np.abs(qd) / self.lim.v))
            ratio_a = float(np.max(np.abs(qdd) / self.lim.a))
            if ratio_v <= 1.0 and ratio_a <= 1.0:
                break
            k *= max(ratio_v, math.sqrt(ratio_a), 1.0) * 1.05
        else:
            raise PlanningError(f'{name}: could not meet the joint limits (stretch {k:.1f})')

        tcp, R = self._fk_samples(q)
        err = np.array([_rotation_angle(Rk, path.R) for Rk in R])
        return Trajectory(name, 'cartesian', t, q, qd, qdd, tcp, err, carrying, prof, path,
                          np.stack([s, sd, sdd, sddd], axis=1), stretch=k,
                          info={'ratio_v': ratio_v, 'ratio_a': ratio_a})

    def _branch(self, T, q_ref):
        sols = self.kin.ik(T, q_ref=q_ref)
        if not sols:
            raise PlanningError('no IK solution at the path start')
        return sols[0].shoulder, sols[0].elbow

    def _track(self, path, s, q_start, branch, name):
        """IK along the path, staying on the start branch and continuous."""
        q = np.empty((len(s), 6))
        q_prev = q_start
        wrist_sign = np.sign(q_start[4]) or 1.0
        for k, sk in enumerate(s):
            sols = [x for x in self.kin.ik(path.pose(sk), q_ref=q_prev)
                    if (x.shoulder, x.elbow) == branch and np.sign(x.q[4]) in (wrist_sign, 0.0)]
            if not sols:
                raise PlanningError(f'{name}: no IK solution on the same branch at s = {sk:.4f} m '
                                    f'(TCP {np.round(path.position(sk), 3)})')
            qk = sols[0].q
            if np.max(np.abs(qk - q_prev)) > self.max_step:
                raise PlanningError(f'{name}: joint jump of {np.max(np.abs(qk - q_prev)):.3f} rad '
                                    f'at s = {sk:.4f} m (configuration change)')
            if abs(math.sin(qk[4])) < self.wrist_margin:
                raise PlanningError(f'{name}: too close to the wrist singularity (q5 = '
                                    f'{math.degrees(qk[4]):.2f} deg) at s = {sk:.4f} m')
            q[k] = q_prev = qk
        self._check_joint_limits(q, name)
        return q
