"""Robotiq 2F-85 gripper model (no ROS): one opening command -> six joint references.

The 2F-85 has one degree of freedom, the knuckle angle theta (0 = open, 0.8 rad =
closed). Its other five joints mimic the knuckle (q_j = multiplier_j * theta). Because
Gazebo's DART engine cannot enforce mimic constraints, the controller drives all six
joints, and this model computes their references.

Grasp geometry comes from gripper_geometry.yaml (tools/robotiq_2f85_geometry.py):
  width(theta)   pad gap, 85 mm open -> 0 closed (tabulated, monotonic)
  TCP            pad centre at the reference object width, measured from the flange
  lever          knuckle axis -> pad centre, for the grip-force model

Grip force: the knuckle PID pushes with tau = p (theta_ref - theta); at the pad that is
F = tau / lever. Closing on an object of width w therefore uses
    knuckles:            theta_ref = theta_contact(w) + F * lever / p   (the squeeze)
    inner knuckles/tips: theta_contact(w)                               (pads parallel)
Only the knuckles get the squeeze: if the finger-tip joints followed theta_ref too, the
pads would tilt by theta_ref - theta_contact (~4 deg at 20 N) and touch the object with
one edge.
"""
import os

import numpy as np
import yaml


class GripperModel:
    def __init__(self, cfg, geo):
        self.joints = dict(cfg['joints'])                 # name -> multiplier (ordered)
        self.joint_names = list(self.joints)
        self.driven_joint = cfg['driven_joint']
        self.theta_open = float(cfg['theta_open'])
        self.theta_closed = float(cfg['theta_closed'])
        self.grip_force = float(cfg['grip_force'])
        self.open_close_time = float(cfg['open_close_time'])
        self.knuckle_p = float(cfg['knuckle_p_gain'])
        self.coupling_length = float(cfg['mount']['coupling']['length'])
        tab = geo['table']
        self._theta = np.asarray(tab['theta'], float)
        self._width = np.asarray(tab['width'], float)          # decreasing with theta
        self.open_width = float(geo['open_width'])
        self.lever = float(geo['lever_knuckle_to_pad'])
        self.tcp_z = self.coupling_length + float(geo['tcp_z'])  # flange -> TCP
        self.pad_half_height = float(geo['pad_half_height'])
        self.outer_half_closing = float(geo['outer_half_x_open'])   # along the tool's Y (closing)
        self.outer_half_side = float(geo['outer_half_y'])           # along the tool's X
        self.reference_width = float(geo['object_width'])

    @classmethod
    def from_config(cls, config_dir=None):
        if config_dir is None:
            from .kinematics import find_config_dir
            config_dir = find_config_dir()
        with open(os.path.join(config_dir, 'gripper.yaml')) as f:
            cfg = yaml.safe_load(f)
        with open(os.path.join(config_dir, 'gripper_geometry.yaml')) as f:
            geo = yaml.safe_load(f)
        return cls(cfg, geo)

    # ------------------------------------------------------------------ geometry
    def clamp(self, theta):
        return float(min(max(theta, self.theta_open), self.theta_closed))

    def theta_for_width(self, width):
        """Knuckle angle at which the pads are `width` apart."""
        return float(np.interp(width, self._width[::-1], self._theta[::-1]))

    def width_for_theta(self, theta):
        return float(np.interp(theta, self._theta, self._width))

    def references(self, theta, theta_linkage=None):
        """Six joint references (joint_names order). The two knuckles get `theta`; the
        inner knuckles and finger tips get `theta_linkage` (default `theta`), the angle at
        which the pads are expected to stop, so they stay parallel there."""
        theta = self.clamp(theta)
        lk = theta if theta_linkage is None else self.clamp(theta_linkage)
        return [m * (theta if 'knuckle_joint' in n and 'inner' not in n else lk)
                for n, m in self.joints.items()]

    # ------------------------------------------------------------------ commands
    def grasp_theta(self, width, force=None):
        """(knuckle reference, linkage reference) that close on an object of `width` with
        `force` per pad."""
        force = self.grip_force if force is None else force
        contact = self.theta_for_width(width)
        return self.clamp(contact + force * self.lever / self.knuckle_p), contact

    def move_time(self, theta_from, theta_to):
        """Open/close duration, proportional to the knuckle travel (at least 0.3 s)."""
        span = self.theta_closed - self.theta_open
        return max(0.3, self.open_close_time * abs(theta_to - theta_from) / span)

    def contact_ok(self, theta_measured, width, tol=0.004):
        """True if the pads stopped at the object's width (+-tol metres)."""
        return abs(self.width_for_theta(theta_measured) - width) <= tol
