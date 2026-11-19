"""Forward and inverse kinematics of the YASKAWA GP7 with the 2-finger gripper.

Everything is derived from the DH table in gp7_description/config/kinematics.yaml
(standard DH:  A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i),  theta_i = offset_i + q_i).

Tool frame
----------
The DH chain ends at frame {6} on the flange face. The gripper adds a fixed
tool transform  T_6^tcp = Tz(tcp_z)  (config/gripper.yaml), so

    T_0^tcp(q) = A_1 A_2 ... A_6 . T_6^tcp

and the TCP has the same orientation as frame {6} (z = approach direction,
y = finger closing direction).

Inverse kinematics (closed form, kinematic decoupling)
------------------------------------------------------
The R, B and T axes intersect at the wrist centre O_c (= O_4 = O_5), so position
and orientation decouple:

1. Wrist centre.  T_0^6 = T_0^tcp . (T_6^tcp)^-1, then
       o_c = o_6 - d6 . z_6               (z_6 = 3rd column of R_0^6)

2. theta_1.  GP7's 40 mm offset a1 lies IN the arm plane (along x1), not
   sideways, so the whole arm stays in the vertical plane through z0 at angle q1:
       q1 = atan2(yc, xc)                 ("front")
       q1 = atan2(yc, xc) + pi            ("back", reaching over the top)

3. theta_2, theta_3 (planar problem in that plane).  With radial coordinate
   rho = xc cos q1 + yc sin q1:
       r = rho - a1,  s = zc - d1
       r = -a2 sin q2 - a3 sin(q2+q3) + d4 cos(q2+q3)
       s =  a2 cos q2 + a3 cos(q2+q3) + d4 sin(q2+q3)
   a3 and d4 are perpendicular legs of a rigid "virtual forearm" of length
   L = sqrt(a3^2 + d4^2) at an extra angle delta = atan2(a3, d4). This is a 2-link
   planar arm (lengths a2, L) whose link angles are psi2 = q2 + 90 deg and
   psi3 = q2 + q3 + delta. The law of cosines gives
       cos(beta) = (r^2 + s^2 - a2^2 - L^2) / (2 a2 L),   beta = psi3 - psi2
       q3 = beta - delta + 90 deg                   (beta < 0: elbow up, beta > 0: elbow down)
       q2 = atan2(s, r) - atan2(L sin beta, a2 + L cos beta) - 90 deg

4. theta_4..theta_6.  R_3^6 = (R_0^3)^T R_0^6 and from A4 A5 A6:
             [ c4c5c6 - s4s6   -c4c5s6 - s4c6   c4s5 ]
     R_3^6 = [ s4c5c6 + c4s6   -s4c5s6 + c4c6   s4s5 ]
             [    -s5c6             s5s6          c5  ]
       q5 = atan2(+-sqrt(1 - r33^2), r33)
       q4 = atan2(s5 r23, s5 r13),   q6 = atan2(s5 r32, -s5 r31)   (sign of s5 only)
   Wrist singularity (s5 = 0, z3 parallel to z5): only q4 + q6 (or q4 - q6) is
   defined; q4 is then kept at the reference value.

Up to 2 (shoulder) x 2 (elbow) x 2 (wrist) = 8 solutions. Joint 4 (+-190 deg) and
joint 6 (+-360 deg) can exceed +-180 deg, so each solution is shifted by
multiples of 2 pi to the value inside the limits closest to a reference q.
"""
import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

JOINT_NAMES = ['joint_1_s', 'joint_2_l', 'joint_3_u', 'joint_4_r', 'joint_5_b', 'joint_6_t']

# Structure the closed-form IK relies on: (alpha_deg, theta_offset_deg) per row and
# which a/d parameters must be zero.
_EXPECTED_ALPHA = (90.0, 0.0, 90.0, -90.0, 90.0, 0.0)
_EXPECTED_OFFSET = (0.0, 90.0, 0.0, 0.0, 0.0, 0.0)


class IKError(RuntimeError):
    """No inverse-kinematics solution (unreachable pose or outside joint limits)."""


# --------------------------------------------------------------------------- helpers
def dh_matrix(a, alpha, d, theta):
    """A_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)   (angles in rad)."""
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def make_pose(position, rotation=None):
    """4x4 homogeneous transform from a position and a 3x3 rotation."""
    T = np.eye(4)
    T[:3, 3] = position
    if rotation is not None:
        T[:3, :3] = rotation
    return T


def pose_from_xyz_rpy(xyz, rpy):
    """Pose from position + fixed-axis roll/pitch/yaw [rad]: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return make_pose(xyz, rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0]))


def top_down_pose(x, y, z, yaw=0.0):
    """TCP pose for a vertical (top-down) grasp: approach axis z_tcp = -Z0.

    R = Rz(yaw) Rx(pi). At yaw = 0: x_tcp = +X0, fingers close along y_tcp = -Y0,
    which is the natural wrist pose (q4 = q6 = 0 when q1 = 0). The gripper is
    symmetric, so yaw and yaw + pi grasp a box the same way.
    """
    return make_pose([x, y, z], rot_z(yaw) @ rot_x(math.pi))


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose_error(T_a, T_b):
    """(position error [m], rotation error [rad]) between two poses."""
    dp = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R = T_a[:3, :3].T @ T_b[:3, :3]
    dr = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    return dp, dr


@dataclass
class IKSolution:
    q: np.ndarray        # joint variables q1..q6 [rad] (the starred DH variables)
    shoulder: str        # 'front' or 'back'
    elbow: str           # 'up' or 'down'
    wrist: str           # 'B+' / 'B-' (sign of q5) or 'singular'

    def __repr__(self):
        deg = ', '.join(f'{math.degrees(v):8.3f}' for v in self.q)
        return f'IKSolution([{deg}] deg, {self.shoulder}/{self.elbow}/{self.wrist})'


# --------------------------------------------------------------------------- main class
class GP7Kinematics:
    """FK / IK for the GP7. Joint vector q = (q1..q6) in rad, DH joint directions."""

    def __init__(self, dh_rows, lower=None, upper=None, tool=None):
        """dh_rows: list of dicts with a, alpha_deg, d, theta_offset_deg (kinematics.yaml).
        lower/upper: joint limits [rad] (None = unlimited). tool: 4x4 T_6^tool (None = flange).
        """
        if len(dh_rows) != 6:
            raise ValueError('GP7 needs 6 DH rows')
        for i, row in enumerate(dh_rows):
            if (not math.isclose(row['alpha_deg'], _EXPECTED_ALPHA[i])
                    or not math.isclose(row['theta_offset_deg'], _EXPECTED_OFFSET[i])):
                raise ValueError(f'DH row {i + 1} does not match the GP7 structure this IK assumes')
        zero = [dh_rows[1]['d'], dh_rows[2]['d'], dh_rows[3]['a'], dh_rows[4]['a'],
                dh_rows[4]['d'], dh_rows[5]['a']]
        if any(abs(v) > 1e-12 for v in zero):
            raise ValueError('d2, d3, a4, a5, d5, a6 must be zero for the closed-form IK')

        self.dh = [(r['a'], math.radians(r['alpha_deg']), r['d'], math.radians(r['theta_offset_deg']))
                   for r in dh_rows]
        self.a1, self.d1 = dh_rows[0]['a'], dh_rows[0]['d']
        self.a2 = dh_rows[1]['a']
        self.a3 = dh_rows[2]['a']
        self.d4 = dh_rows[3]['d']
        self.d6 = dh_rows[5]['d']
        self.L = math.hypot(self.a3, self.d4)        # virtual forearm length
        self.delta = math.atan2(self.a3, self.d4)    # its angle offset

        self.lower = np.full(6, -np.inf) if lower is None else np.asarray(lower, float)
        self.upper = np.full(6, np.inf) if upper is None else np.asarray(upper, float)
        self.T_tool = np.eye(4) if tool is None else np.asarray(tool, float)
        self._T_tool_inv = np.linalg.inv(self.T_tool)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config_dir=None, tool='tcp'):
        """Build from gp7_description/config. tool: 'tcp' (gripper TCP) or 'flange'."""
        config_dir = config_dir or find_config_dir()
        with open(os.path.join(config_dir, 'kinematics.yaml')) as f:
            dh_rows = yaml.safe_load(f)['dh']
        with open(os.path.join(config_dir, 'joint_limits.yaml')) as f:
            lim = yaml.safe_load(f)
        lower = [math.radians(lim[n]['lower_deg']) for n in JOINT_NAMES]
        upper = [math.radians(lim[n]['upper_deg']) for n in JOINT_NAMES]
        T_tool = None
        if tool == 'tcp':      # Robotiq 2F-85 TCP (pad centre), from the gripper model
            from .gripper import GripperModel
            T_tool = make_pose([0.0, 0.0, GripperModel.from_config(config_dir).tcp_z])
        elif tool != 'flange':
            raise ValueError("tool must be 'tcp' or 'flange'")
        return cls(dh_rows, lower, upper, T_tool)

    # ------------------------------------------------------------------ forward kinematics
    def fk_frames(self, q):
        """[T_0^0, T_0^1, ..., T_0^6] (DH frames, no tool)."""
        T = [np.eye(4)]
        for (a, alpha, d, off), qi in zip(self.dh, q):
            T.append(T[-1] @ dh_matrix(a, alpha, d, off + qi))
        return T

    def fk(self, q):
        """T_0^tool(q): pose of the tool frame (TCP or flange) in the base frame."""
        return self.fk_frames(q)[6] @ self.T_tool

    # ------------------------------------------------------------------ differential kinematics
    def jacobian(self, q):
        """6x6 geometric Jacobian at the tool point, in base-frame coordinates.

        [v; w] = J(q) q_dot. Every joint is revolute and joint i turns about z_{i-1},
        so column i is   J_v = z_{i-1} x (p_tool - o_{i-1}),   J_w = z_{i-1}.
        """
        T = self.fk_frames(q)
        p = (T[6] @ self.T_tool)[:3, 3]
        J = np.empty((6, 6))
        for i in range(6):
            z, o = T[i][:3, 2], T[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def manipulability(self, q):
        """Yoshikawa manipulability sqrt(det(J J^T)); 0 at a singularity."""
        J = self.jacobian(q)
        return float(math.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def joint_velocities(self, q, v, w=(0.0, 0.0, 0.0)):
        """q_dot for a desired tool twist (v linear [m/s], w angular [rad/s]): q_dot = J^-1 [v; w]."""
        return np.linalg.solve(self.jacobian(q), np.r_[v, w])

    # ------------------------------------------------------------------ inverse kinematics
    def wrist_center(self, T_target):
        """Wrist centre o_c for a desired tool pose."""
        T06 = T_target @ self._T_tool_inv
        return T06[:3, 3] - self.d6 * T06[:3, 2]

    def ik(self, T_target, q_ref=None, respect_limits=True, tol=1e-6):
        """All IK solutions for a tool pose, sorted by distance to q_ref.

        q_ref (default: zeros) picks the 2*pi-equivalent of joints 4/6 and the free
        angle at singularities, and orders the result (closest first).
        """
        q_ref = np.zeros(6) if q_ref is None else np.asarray(q_ref, float)
        T06 = T_target @ self._T_tool_inv
        R06 = T06[:3, :3]
        xc, yc, zc = T06[:3, 3] - self.d6 * R06[:, 2]

        # --- theta_1: front and back solution (free at the shoulder singularity)
        if math.hypot(xc, yc) < 1e-9:
            q1_list = [(q_ref[0], 'front')]
        else:
            phi = math.atan2(yc, xc)
            q1_list = [(phi, 'front'), (wrap_to_pi(phi + math.pi), 'back')]

        raw = []
        for q1, shoulder in q1_list:
            # --- theta_2, theta_3: planar 2-link problem
            rho = xc * math.cos(q1) + yc * math.sin(q1)
            r, s = rho - self.a1, zc - self.d1
            D = (r * r + s * s - self.a2 ** 2 - self.L ** 2) / (2.0 * self.a2 * self.L)
            if abs(D) > 1.0 + 1e-9:
                continue                                    # wrist centre out of reach
            beta0 = math.acos(max(-1.0, min(1.0, D)))
            for beta in ([-beta0, beta0] if beta0 > 1e-12 else [0.0]):
                q3 = beta - self.delta + math.pi / 2
                q2 = (math.atan2(s, r) - math.atan2(self.L * math.sin(beta),
                                                    self.a2 + self.L * math.cos(beta)) - math.pi / 2)
                elbow = 'up' if beta <= 0 else 'down'

                # --- theta_4..6 from R_3^6
                R03 = self.fk_frames([q1, q2, q3, 0.0, 0.0, 0.0])[3][:3, :3]
                R36 = R03.T @ R06
                c5 = max(-1.0, min(1.0, R36[2, 2]))
                s5_abs = math.sqrt(max(0.0, 1.0 - c5 * c5))
                # wrist singularity; s5 = sqrt(1 - c5^2) turns a 1e-16 rounding error in c5
                # into ~1e-8, so the threshold must sit well above that
                if s5_abs < 1e-7:
                    q4 = q_ref[3]
                    if c5 > 0:
                        q5, q6 = 0.0, math.atan2(R36[1, 0], R36[0, 0]) - q4      # q4 + q6 fixed
                    else:
                        q5, q6 = math.pi, q4 - math.atan2(-R36[1, 0], -R36[0, 0])  # q4 - q6 fixed
                    raw.append(([q1, q2, q3, q4, q5, q6], shoulder, elbow, 'singular'))
                else:
                    for sgn in (1.0, -1.0):
                        q5 = math.atan2(sgn * s5_abs, c5)
                        q4 = math.atan2(sgn * R36[1, 2], sgn * R36[0, 2])
                        q6 = math.atan2(sgn * R36[2, 1], -sgn * R36[2, 0])
                        raw.append(([q1, q2, q3, q4, q5, q6], shoulder, elbow,
                                    'B+' if sgn > 0 else 'B-'))

        solutions = []
        for q, shoulder, elbow, wrist in raw:
            q = self._choose_equivalent(np.array(q), q_ref, respect_limits)
            if q is None:
                continue
            dp, dr = pose_error(self.fk(q), T_target)
            if dp > tol or dr > tol:          # safety net: never return a wrong solution
                continue
            if any(np.allclose(q, s.q, atol=1e-9) for s in solutions):
                continue
            solutions.append(IKSolution(q, shoulder, elbow, wrist))
        solutions.sort(key=lambda s: float(np.linalg.norm(s.q - q_ref)))
        return solutions

    def ik_closest(self, T_target, q_ref=None, respect_limits=True):
        """The IK solution closest to q_ref (e.g. the current joint state)."""
        sols = self.ik(T_target, q_ref, respect_limits)
        if not sols:
            raise IKError('pose is unreachable or every solution violates the joint limits')
        return sols[0]

    def within_limits(self, q, margin=0.0):
        q = np.asarray(q, float)
        return bool(np.all(q >= self.lower + margin) and np.all(q <= self.upper - margin))

    # ------------------------------------------------------------------ internals
    def _choose_equivalent(self, q, q_ref, respect_limits):
        """Per joint, the 2*pi-shifted value closest to q_ref (inside the limits)."""
        out = np.empty(6)
        for j in range(6):
            base = wrap_to_pi(q[j])
            cands = [base + k * 2.0 * math.pi for k in (-2, -1, 0, 1, 2)]
            if respect_limits:
                cands = [c for c in cands if self.lower[j] - 1e-9 <= c <= self.upper[j] + 1e-9]
                if not cands:
                    return None
            out[j] = min(cands, key=lambda c: abs(c - q_ref[j]))
        return out


def find_config_dir():
    """gp7_description/config: installed share dir, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('gp7_description'), 'config')
    except Exception:  # not in a sourced ROS environment
        here = os.path.dirname(os.path.realpath(__file__))
        for up in ('..', os.path.join('..', '..')):
            cand = os.path.normpath(os.path.join(here, up, 'gp7_description', 'config'))
            if os.path.isfile(os.path.join(cand, 'kinematics.yaml')):
                return cand
        raise FileNotFoundError('gp7_description/config not found - source the workspace')


_find_config_dir = find_config_dir   # backwards-compatible alias
