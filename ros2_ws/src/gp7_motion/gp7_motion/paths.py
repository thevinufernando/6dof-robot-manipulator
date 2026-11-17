"""Cartesian TCP paths with CONSTANT orientation.

Every path is parameterised by arc length s in [0, L]:
    pose(s)    -> 4x4 TCP pose   (orientation is always the same R*)
    tangent(s) -> dp/ds, a unit vector (so the TCP speed is |v| = s_dot)

Constant orientation means the tool's angular velocity is zero along the whole
path, which is exactly the "Box B stays upright" requirement. The joint
velocities then follow from  q_dot = J(q)^-1 [s_dot * tangent(s); 0, 0, 0].

LinePath         straight line p0 -> p1 (approach, descend, lift, retreat)
CylindricalPath  transfer around the robot base: radius rho, azimuth phi and
                 height z of the TCP change linearly together, so the path stays
                 at a sensible distance from the S-axis (a straight line between two
                 points on opposite sides of the robot would pass right over the base,
                 where the arm cannot reach). It swings through the FRONT of the robot,
                 because the S-axis (+-170 deg) cannot turn through the back. The arc
                 length has no closed form, so it is tabulated numerically and
                 inverted by interpolation.
"""
import math

import numpy as np


def _pose(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class LinePath:
    def __init__(self, p0, p1, R):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.R = np.asarray(R, float)
        d = self.p1 - self.p0
        self.length = float(np.linalg.norm(d))
        self._u = d / self.length if self.length > 0 else np.zeros(3)

    def position(self, s):
        s = np.asarray(s, float)
        return self.p0 + np.multiply.outer(s, self._u)

    def pose(self, s):
        return _pose(self.position(s), self.R)

    def tangent(self, s):
        s = np.asarray(s, float)
        return np.broadcast_to(self._u, s.shape + (3,)).copy()

    def __repr__(self):
        return f'LinePath({np.round(self.p0, 4)} -> {np.round(self.p1, 4)}, L={self.length:.4f} m)'


class CylindricalPath:
    def __init__(self, p0, p1, R, through_front=True, n_table=4001):
        """through_front=True swings through the front of the robot (azimuth never
        crosses +-180 deg, which the S-axis (+-170 deg) cannot reach); False takes
        the shorter way around."""
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.R = np.asarray(R, float)
        self.rho0, self.rho1 = math.hypot(*self.p0[:2]), math.hypot(*self.p1[:2])
        if min(self.rho0, self.rho1) < 1e-6:
            raise ValueError('CylindricalPath endpoints must not lie on the z axis')
        self.phi0 = math.atan2(self.p0[1], self.p0[0])
        dphi = math.atan2(self.p1[1], self.p1[0]) - self.phi0      # both in (-pi, pi]
        self.dphi = dphi if through_front else (dphi + math.pi) % (2 * math.pi) - math.pi
        self.drho = self.rho1 - self.rho0
        self.dz = self.p1[2] - self.p0[2]
        # arc length table s(u), u in [0, 1]:  |dp/du|^2 = drho^2 + (rho(u) dphi)^2 + dz^2
        u = np.linspace(0.0, 1.0, n_table)
        speed = np.sqrt(self.drho ** 2 + ((self.rho0 + self.drho * u) * self.dphi) ** 2 + self.dz ** 2)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))])
        self._u_tab, self._s_tab = u, s
        self.length = float(s[-1])

    def _u_of_s(self, s):
        return np.interp(s, self._s_tab, self._u_tab)

    def _point(self, u):
        rho = self.rho0 + self.drho * u
        phi = self.phi0 + self.dphi * u
        return np.stack([rho * np.cos(phi), rho * np.sin(phi), self.p0[2] + self.dz * u], axis=-1)

    def position(self, s):
        return self._point(self._u_of_s(np.asarray(s, float)))

    def pose(self, s):
        return _pose(self.position(s), self.R)

    def tangent(self, s):
        u = self._u_of_s(np.asarray(s, float))
        rho = self.rho0 + self.drho * u
        phi = self.phi0 + self.dphi * u
        dp = np.stack([self.drho * np.cos(phi) - rho * self.dphi * np.sin(phi),
                       self.drho * np.sin(phi) + rho * self.dphi * np.cos(phi),
                       np.full_like(u, self.dz)], axis=-1)
        return dp / np.linalg.norm(dp, axis=-1, keepdims=True)

    def __repr__(self):
        return (f'CylindricalPath({np.round(self.p0, 4)} -> {np.round(self.p1, 4)}, '
                f'dphi={math.degrees(self.dphi):.1f} deg, L={self.length:.4f} m)')
