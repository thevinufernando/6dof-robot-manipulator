#!/usr/bin/env python3
"""Locate the joint axes inside the raw SolidWorks STL exports.

Each STL in 3d_designs/STL_files was exported in its own part coordinate
system, so before writing the URDF we need to know where every joint axis sits
in every mesh. Rotary joints show up as cylindrical bearing surfaces: project
the vertices along the (known) axis direction and least-squares fit a circle
(Kasa fit) to the ring of vertices near an initial guess.

These measurements are what tools/prepare_meshes.py and
gp7_description/config/kinematics.yaml are based on.

Usage:  python3 tools/measure_joint_axes.py
"""
import os
import numpy as np
from stl_utils import load_stl, unique_vertices

STL_DIR = os.path.join(os.path.dirname(__file__), '..', '3d_designs', 'STL_files')


def kasa_circle(p):
    A = np.c_[2 * p, np.ones(len(p))]
    s = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    c = s[:2]
    return c, np.sqrt(s[2] + c @ c)


def fit_axis(mesh, axis, slab, c0, r0, tol):
    """Fit a circle to vertices in `slab` (range along `axis`) near (c0, r0)."""
    V = unique_vertices(load_stl(os.path.join(STL_DIR, mesh)))
    ax = 'xyz'.index(axis)
    other = [i for i in range(3) if i != ax]
    P = V[(V[:, ax] >= slab[0]) & (V[:, ax] <= slab[1])][:, other]
    c, r = np.array(c0, float), r0
    for _ in range(20):                      # iterate: select ring -> refit
        sel = np.abs(np.linalg.norm(P - c, axis=1) - r) < tol
        c, r = kasa_circle(P[sel])
    names = 'xyz'[other[0]] + 'xyz'[other[1]]
    return names, c, r, int(sel.sum())


# (description, mesh, axis direction in mesh, slab along axis, centre guess, radius guess, tol)
MEASUREMENTS = [
    ('J1 (S) axis in link1_s',  'link1_s.stl', 'y', (-0.135, -0.11), (0, 0.035), 0.080, 0.010),
    ('J2 (L) axis in link1_s',  'link1_s.stl', 'x', (0.07, 0.09),    (0, 0),     0.060, 0.006),
    ('J2 (L) axis in link2_L',  'link2_L.stl', 'y', (-0.2, 0.2),     (0, 0),     0.060, 0.006),
    ('J3 (U) axis in link2_L',  'link2_L.stl', 'y', (-0.2, 0.2),     (0, -0.445), 0.053, 0.006),
    ('J3 (U) axis in link3_U',  'link3_U.stl', 'y', (-0.1, 0.1),     (0, 0),     0.050, 0.006),
    ('J4 (R) axis in link3_U',  'link3_U.stl', 'x', (0.05, 0.08),    (0, 0.04),  0.045, 0.006),
    ('J4 (R) axis in link4_R',  'link4_R.stl', 'x', (-0.26, -0.2),   (0, 0),     0.050, 0.006),
    ('J5 (B) axis in link4_R',  'link4_R.stl', 'z', (-0.06, 0.06),   (0.125, 0), 0.030, 0.005),
    ('J5 (B) axis in link5_B',  'link5_B.stl', 'z', (-0.04, 0.04),   (0, 0),     0.037, 0.004),
    ('J6 (T) axis in link5_B',  'link5_B.stl', 'y', (0.04, 0.08),    (0, 0),     0.030, 0.005),
    ('J6 (T) axis in link6_T',  'link6_T.stl', 'x', (-0.02, 0.02),   (0, 0),     0.030, 0.003),
]

if __name__ == '__main__':
    for desc, *args in MEASUREMENTS:
        names, c, r, n = fit_axis(*args)
        print(f'{desc:26s} axis ∥ mesh {args[1]}  centre ({names}) = '
              f'({c[0]:+.4f}, {c[1]:+.4f}) m   r = {r:.4f} m   [{n} pts]')
