"""Denavit-Hartenberg helpers (standard DH), reading the GP7 table from
ros2_ws/src/gp7_description/config/kinematics.yaml."""
import os
import numpy as np
import yaml

KIN_YAML = os.path.join(os.path.dirname(__file__), '..', 'ros2_ws', 'src',
                        'gp7_description', 'config', 'kinematics.yaml')


def load_dh(path=KIN_YAML):
    """List of DH rows (dicts) with a, alpha_deg, d, theta_offset_deg."""
    with open(path) as f:
        return yaml.safe_load(f)['dh']


def dh_matrix(a, alpha, d, theta):
    """A_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)   (angles in rad)."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def fk_frames(q, rows=None):
    """Return [T_0^0, T_0^1, ..., T_0^n] for joint variables q (rad)."""
    rows = rows or load_dh()
    T = [np.eye(4)]
    for row, qi in zip(rows, q):
        T.append(T[-1] @ dh_matrix(row['a'], np.radians(row['alpha_deg']), row['d'],
                                   np.radians(row['theta_offset_deg']) + qi))
    return T
