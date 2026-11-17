"""Velocity profiles (time scaling) for a point-to-point move of length D.

A profile answers "how far along the path am I at time t?": s(t) goes from 0 to
D with zero velocity at both ends, and the path decides *where* that distance is
(Cartesian line/arc for the TCP, or a straight line in joint space).

TrapezoidalProfile(D, v_max, a_max)
    accelerate at +a_max, cruise at v_max, decelerate at -a_max.
    Acceleration jumps (infinite jerk) at the phase changes.
        T_a = v_max / a_max                       (acceleration time)
        T_v = D / v_max - T_a                     (cruise time)
        T   = 2 T_a + T_v
    If D < v_max^2 / a_max the cruise phase vanishes and the peak speed is
    v_peak = sqrt(D a_max) ("triangular" profile).

SCurveProfile(D, v_max, a_max, j_max)  -- "double S", 7 phases
    jerk +j, 0, -j | 0 (cruise) | -j, 0, +j
    Acceleration ramps instead of jumping, so the motion is jerk-limited:
    smoother for the robot and gentler on a gripped object.
    Symmetric case (zero start/end velocity), Biagiotti & Melchiorri,
    "Trajectory Planning for Automatic Machines and Robots", Sec. 3.4:
      1. v_max and a_max reached:
            T_j = a_max / j_max         if v_max j_max >= a_max^2
            T_a = T_j + v_max / a_max
         else (a_max not reached before v_max):
            T_j = sqrt(v_max / j_max),  T_a = 2 T_j
         T_v = D / v_max - T_a  (>= 0)
      2. T_v < 0 (v_max not reached), a_max still reached:
            T_j = a_max / j_max,  T_a = (a_max^2/j_max + sqrt(a_max^4/j_max^2 + 4 a_max D)) / (2 a_max)
         valid while T_a >= 2 T_j
      3. neither reached: T_j = (D / (2 j_max))^(1/3),  T_a = 2 T_j
    Actual peak acceleration a_lim = j_max T_j, peak velocity v_lim = (T_a - T_j) a_lim,
    total time T = 2 T_a + T_v.

Both classes: .duration, .v_peak, .a_peak, and .sample(t) -> (s, s_dot, s_ddot, s_dddot)
(t may be a scalar or a numpy array; values are clamped to [0, T]).
"""
import math

import numpy as np


class _Profile:
    """Common code: symmetric profile = acceleration phase, cruise, mirrored deceleration."""

    D = 0.0
    T_a = 0.0     # acceleration phase duration (= deceleration phase)
    T_v = 0.0     # cruise duration

    @property
    def duration(self):
        return 2.0 * self.T_a + self.T_v

    def sample(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        s, v, a, j = (np.zeros_like(t) for _ in range(4))
        if self.D <= 0.0:
            return s, v, a, j
        T = self.duration
        acc = t < self.T_a
        cru = (t >= self.T_a) & (t < self.T_a + self.T_v)
        dec = t >= self.T_a + self.T_v
        # acceleration phase
        s[acc], v[acc], a[acc], j[acc] = self._accel(t[acc])
        # cruise
        s[cru] = self.D_acc + self.v_peak * (t[cru] - self.T_a)
        v[cru] = self.v_peak
        # deceleration = acceleration phase mirrored in time
        sd, vd, ad, jd = self._accel(T - t[dec])
        s[dec], v[dec], a[dec], j[dec] = self.D - sd, vd, -ad, jd
        return s, v, a, j

    @property
    def D_acc(self):
        """Distance covered during the acceleration phase."""
        return self.v_peak * self.T_a / 2.0

    def __repr__(self):
        return (f'{type(self).__name__}(D={self.D:.4f}, T={self.duration:.3f}s, '
                f'v_peak={self.v_peak:.4f}, a_peak={self.a_peak:.4f})')


class TrapezoidalProfile(_Profile):
    def __init__(self, D, v_max, a_max):
        if v_max <= 0 or a_max <= 0:
            raise ValueError('v_max and a_max must be positive')
        self.D = float(max(D, 0.0))
        self.v_max, self.a_max = float(v_max), float(a_max)
        if self.D * self.a_max >= self.v_max ** 2:          # cruise phase exists
            self.v_peak = self.v_max
            self.T_a = self.v_max / self.a_max
            self.T_v = self.D / self.v_max - self.T_a
        else:                                              # triangular
            self.v_peak = math.sqrt(self.D * self.a_max)
            self.T_a = self.v_peak / self.a_max
            self.T_v = 0.0
        self.a_peak = self.a_max if self.D > 0 else 0.0

    def _accel(self, t):
        a = np.full_like(t, self.a_max)
        return 0.5 * self.a_max * t * t, self.a_max * t, a, np.zeros_like(t)


class SCurveProfile(_Profile):
    def __init__(self, D, v_max, a_max, j_max):
        if v_max <= 0 or a_max <= 0 or j_max <= 0:
            raise ValueError('v_max, a_max and j_max must be positive')
        self.D = float(max(D, 0.0))
        self.v_max, self.a_max, self.j_max = float(v_max), float(a_max), float(j_max)
        v, a, j, D = self.v_max, self.a_max, self.j_max, self.D
        if D == 0.0:
            self.T_j = self.T_a = self.T_v = 0.0
            self.a_peak = self.v_peak = 0.0
            return

        # case 1: assume v_max is reached
        if v * j >= a * a:
            T_j = a / j
            T_a = T_j + v / a
        else:
            T_j = math.sqrt(v / j)
            T_a = 2.0 * T_j
        T_v = D / v - T_a
        if T_v < 0.0:
            # case 2: v_max not reached, a_max reached
            T_v = 0.0
            T_j = a / j
            T_a = (a * a / j + math.sqrt(a ** 4 / j ** 2 + 4.0 * a * D)) / (2.0 * a)
            if T_a < 2.0 * T_j:
                # case 3: neither reached (pure jerk phases)
                T_j = (D / (2.0 * j)) ** (1.0 / 3.0)
                T_a = 2.0 * T_j
        self.T_j, self.T_a, self.T_v = T_j, T_a, T_v
        self.a_peak = j * T_j
        self.v_peak = (T_a - T_j) * self.a_peak

    def _accel(self, t):
        """Acceleration phase 0 <= t <= T_a: jerk +j, then constant a, then jerk -j."""
        j, Tj, Ta, al, vl = self.j_max, self.T_j, self.T_a, self.a_peak, self.v_peak
        s, v, a, jk = (np.zeros_like(t) for _ in range(4))
        p1 = t < Tj
        p2 = (t >= Tj) & (t < Ta - Tj)
        p3 = t >= Ta - Tj
        t1 = t[p1]
        s[p1], v[p1], a[p1], jk[p1] = j * t1 ** 3 / 6.0, j * t1 ** 2 / 2.0, j * t1, j
        t2 = t[p2]
        s[p2] = al / 6.0 * (3.0 * t2 ** 2 - 3.0 * Tj * t2 + Tj ** 2)
        v[p2] = al * (t2 - Tj / 2.0)
        a[p2] = al
        r = Ta - t[p3]                                   # time left in the phase
        s[p3] = vl * Ta / 2.0 - vl * r + j * r ** 3 / 6.0
        v[p3] = vl - j * r ** 2 / 2.0
        a[p3] = j * r
        jk[p3] = -j
        return s, v, a, jk


def make_profile(kind, D, v_max, a_max, j_max=None):
    """'scurve' or 'trapezoidal' profile for distance D."""
    if kind == 'scurve':
        return SCurveProfile(D, v_max, a_max, j_max)
    if kind == 'trapezoidal':
        return TrapezoidalProfile(D, v_max, a_max)
    raise ValueError(f"unknown profile '{kind}' (use 'scurve' or 'trapezoidal')")
