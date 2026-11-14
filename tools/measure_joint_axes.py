#!/usr/bin/env python3
"""Locate the joint axes inside the raw SolidWorks STL exports.

Each STL in 3d_designs/STL_files was exported in its own part coordinate
system, so before writing the URDF we need to know where every joint axis sits
in every mesh.

Method - concentric-feature search:
  A rotary joint is surrounded by turned features (bearing seats, bores, bolt
  circles, holes) that are all concentric with the axis. Sample points evenly
  over the mesh surface, project them along the known axis direction, and scan
  candidate centres. For each candidate, histogram the radial distances of the
  points: at the true axis every turned feature collapses onto one radius, so
  the histogram is very "spiky". The score sum(p_i^2) of the normalised
  histogram is maximised there. (Fitting a circle to an outline is NOT reliable:
  cast outlines are often not concentric with the axis.)

Results feed tools/prepare_meshes.py and gp7_description/config/kinematics.yaml.

Usage:  python3 tools/measure_joint_axes.py      (~1 minute)
"""
import os
import numpy as np
from stl_utils import load_stl

STL_DIR = os.path.join(os.path.dirname(__file__), '..', '3d_designs', 'STL_files')


def sample_surface(tris, density=1.5e6, seed=0):
    """Uniform random points on a triangle mesh (~density points per m^2)."""
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    idx = np.repeat(np.arange(len(tris)), np.maximum(1, (area * density).astype(int)))
    u = np.random.default_rng(seed).random((len(idx), 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = tris[idx]
    return t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])


def concentricity(r, bin_width=0.0002):
    h, _ = np.histogram(r, bins=np.arange(0, 0.15, bin_width))
    p = h / max(h.sum(), 1)
    return 1e3 * (p ** 2).sum()


def find_axis(points, axis, slab, c0, span, r_max):
    """Best axis centre (2-D, in the plane normal to `axis`) near c0."""
    ax = 'xyz'.index(axis)
    other = [i for i in range(3) if i != ax]
    Q = points[(points[:, ax] >= slab[0]) & (points[:, ax] <= slab[1])][:, other]
    c = np.asarray(c0, float)
    for half, step in [(span, 0.0005), (0.002, 0.0001)]:     # coarse, then fine grid
        grid = np.arange(-half, half + 1e-12, step)
        best = (-1.0, c)
        for du in grid:
            for dv in grid:
                cand = c + (du, dv)
                r = np.linalg.norm(Q - cand, axis=1)
                r = r[r < r_max]
                if len(r) > 200:
                    s = concentricity(r)
                    if s > best[0]:
                        best = (s, cand)
        c = best[1]
    return 'xyz'[other[0]] + 'xyz'[other[1]], c, best[0]


# (description, mesh, axis direction in mesh, slab along axis, centre guess, search half-width, r_max)
MEASUREMENTS = [
    ('J1 (S) in link1_s',          'link1_s', 'y', (-0.135, -0.08), (0, 0.04),    0.006, 0.10),
    ('J2 (L) in link1_s',          'link1_s', 'x', (0.03, 0.09),    (0, 0),       0.004, 0.07),
    ('J2 (L) in link2_L',          'link2_L', 'y', (0.06, 0.12),    (0, 0),       0.006, 0.065),
    ('J3 (U) in link2_L',          'link2_L', 'y', (0.06, 0.12),    (0, -0.445),  0.006, 0.06),
    ('J3 (U) in link3_U',          'link3_U', 'y', (0.04, 0.08),    (0, 0),       0.006, 0.06),
    ('J4 (R) in link3_U',          'link3_U', 'x', (0.03, 0.071),   (0, 0.04),    0.006, 0.085),
    ('J4 (R) in link4_R',          'link4_R', 'x', (-0.26, -0.15),  (0, 0),       0.004, 0.06),
    ('J5 (B) in link4_R (prong)',  'link4_R', 'z', (0.03, 0.06),    (0.118, 0),   0.006, 0.045),
    ('J5 (B) in link5_B',          'link5_B', 'z', (-0.04, 0.04),   (0, 0),       0.006, 0.045),
    ('J6 (T) in link5_B',          'link5_B', 'y', (0.03, 0.08),    (0, 0),       0.004, 0.04),
    ('J6 (T) in link6_T',          'link6_T', 'x', (-0.02, 0.02),   (0, 0),       0.003, 0.031),
]

if __name__ == '__main__':
    cache = {}
    for desc, mesh, axis, slab, c0, span, r_max in MEASUREMENTS:
        if mesh not in cache:
            cache[mesh] = sample_surface(load_stl(os.path.join(STL_DIR, mesh + '.stl')))
        names, c, score = find_axis(cache[mesh], axis, slab, c0, span, r_max)
        print(f'{desc:27s} axis ∥ mesh {axis}   centre ({names}) = '
              f'({c[0] * 1e3:+7.1f}, {c[1] * 1e3:+7.1f}) mm   [score {score:5.1f}]')
