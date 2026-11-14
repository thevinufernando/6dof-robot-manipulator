#!/usr/bin/env python3
"""Convert the raw SolidWorks STL exports into URDF-ready meshes.

For each of the 7 links this script:
  1. re-expresses the mesh in its URDF link frame (p_link = R @ p_mesh + t),
     so the URDF visual/collision origins are all identity;
  2. writes a convex-hull collision mesh (a few hundred triangles instead of
     ~50k, which keeps Gazebo's physics fast);
  3. computes volume, centre of mass and inertia tensor of the closed mesh
     and scales density so the whole arm weighs 34 kg (GP7 datasheet).

URDF link frame convention (all frames parallel to base_link at home pose,
X forward, Z up):
  base_link : on the floor, under the S-axis
  link_1_s  : on the S-axis, at the L-axis height (d1)
  link_2_l  : on the L-axis
  link_3_u  : on the U-axis
  link_4_r  : on the R-axis, directly above the U-axis (a3)
  link_5_b  : on the B-axis (d4 ahead of link_4_r)
  link_6_t  : centre of the tool flange face (d6 ahead of link_5_b)

Outputs (inside ros2_ws/src/gp7_description):
  meshes/visual/<link>.stl, meshes/collision/<link>.stl, config/inertials.yaml

Usage:  python3 tools/prepare_meshes.py
"""
import os
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from stl_utils import load_stl, save_stl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STL_DIR = os.path.join(ROOT, '3d_designs', 'STL_files')
PKG = os.path.join(ROOT, 'ros2_ws', 'src', 'gp7_description')
TOTAL_MASS = 34.0  # kg, GP7 datasheet

K = yaml.safe_load(open(os.path.join(PKG, 'config', 'kinematics.yaml')))


def cols(ex, ey, ez):
    """Rotation whose columns are where mesh x, y, z axes point in the link frame."""
    R = np.array([ex, ey, ez], dtype=float).T
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    return R


# link name -> (source STL, R, t).  Axis positions from tools/measure_joint_axes.py.
LINKS = {
    # base: mesh is Y-up, origin on floor under S-axis; cable connector box is
    # at mesh +X = the back of the robot, so robot-forward is mesh -X.
    'base_link': ('base_link.stl', cols([-1, 0, 0], [0, 0, 1], [0, 1, 0]), [0, 0, 0]),
    # S-head: mesh origin is ON the L-axis (mesh X); mesh Y is up; the S-axis is
    # 41.5 mm towards mesh +Z, so robot-forward is mesh -Z.
    'link_1_s': ('link1_s.stl', cols([0, -1, 0], [0, 0, 1], [-1, 0, 0]), [K['a1'], 0, 0]),
    # Lower arm: mesh origin on L-axis (mesh Y), arm hangs along mesh -Z.
    # The L-axis passes exactly through the mesh origin (concentric-feature fit).
    # NOTE: mesh +X (the side the arm bows towards) is mapped to robot-forward.
    'link_2_l': ('link2_L.stl', cols([1, 0, 0], [0, -1, 0], [0, 0, -1]), [0, 0, 0]),
    # Upper-arm casing: already Z-up, X-forward, origin on U-axis (mesh Y).
    'link_3_u': ('link3_U.stl', cols([1, 0, 0], [0, 1, 0], [0, 0, 1]), [0, 0, 0]),
    # Forearm: mesh X along R-axis, B-axis is mesh Z and its bores sit 119.3 mm
    # ahead of the mesh origin -> mesh origin is (d4 - 0.1193) ahead of link_4_r.
    'link_4_r': ('link4_R.stl', cols([1, 0, 0], [0, 0, 1], [0, -1, 0]), [K['d4'] - 0.1193, 0, 0]),
    # Wrist: origin on B-axis (mesh Z), T-axis/flange along mesh +Y.
    'link_5_b': ('link5_B.stl', cols([0, 0, 1], [1, 0, 0], [0, 1, 0]), [0, 0, 0]),
    # Flange: axis mesh X, outer face (with the Ø12 pin hole) at mesh x=+15 mm.
    'link_6_t': ('link6_T.stl', cols([1, 0, 0], [0, 1, 0], [0, 0, 1]), [-0.015, 0, 0]),
}


def mass_properties(tris):
    """Volume, centre of mass and inertia (about COM, unit density) of a closed mesh.

    Sums signed tetrahedra (origin, a, b, c). For one tetrahedron with
    det = a·(b×c):  V = det/6,  ∫x dV = V (a+b+c)/4,
    ∫x xᵀ dV = det/120 · (aaᵀ + bbᵀ + ccᵀ + (a+b+c)(a+b+c)ᵀ).
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    vol = det.sum() / 6.0
    s = a + b + c
    com = (det[:, None] * s).sum(0) / 24.0 / vol
    C = sum(np.einsum('i,ij,ik->jk', det, v, v) for v in (a, b, c, s)) / 120.0
    if vol < 0:  # inward-facing normals: flip everything
        vol, C = -vol, -C
    I_origin = np.trace(C) * np.eye(3) - C
    I_com = I_origin - vol * (com @ com * np.eye(3) - np.outer(com, com))
    return vol, com, I_com


def convex_hull_tris(tris):
    pts = np.unique(tris.reshape(-1, 3).round(6), axis=0)
    hull = ConvexHull(pts)
    out = pts[hull.simplices]
    centre = pts[hull.vertices].mean(0)
    nrm = np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])
    flip = np.einsum('ij,ij->i', nrm, out[:, 0] - centre) < 0
    out[flip] = out[flip][:, [0, 2, 1]]    # make all normals point outwards
    return out


def main():
    os.makedirs(os.path.join(PKG, 'meshes', 'visual'), exist_ok=True)
    os.makedirs(os.path.join(PKG, 'meshes', 'collision'), exist_ok=True)
    props = {}
    for link, (src, R, t) in LINKS.items():
        tris = load_stl(os.path.join(STL_DIR, src)) @ R.T + np.asarray(t, float)
        save_stl(os.path.join(PKG, 'meshes', 'visual', f'{link}.stl'), tris)
        hull = convex_hull_tris(tris)
        save_stl(os.path.join(PKG, 'meshes', 'collision', f'{link}.stl'), hull)
        props[link] = mass_properties(tris)
        print(f'{link:10s} {len(tris):6d} tris -> hull {len(hull):4d} tris, '
              f'volume {props[link][0] * 1e3:6.3f} L, '
              f'bbox min {np.round(tris.reshape(-1, 3).min(0), 3)} max {np.round(tris.reshape(-1, 3).max(0), 3)}')

    density = TOTAL_MASS / sum(p[0] for p in props.values())
    out = {}
    for link, (vol, com, I) in props.items():
        m = density * vol
        I = density * I
        out[link] = {
            'mass': round(float(m), 4),
            'com': [round(float(x), 5) for x in com],
            'ixx': float(f'{I[0, 0]:.6g}'), 'ixy': float(f'{I[0, 1]:.6g}'), 'ixz': float(f'{I[0, 2]:.6g}'),
            'iyy': float(f'{I[1, 1]:.6g}'), 'iyz': float(f'{I[1, 2]:.6g}'), 'izz': float(f'{I[2, 2]:.6g}'),
        }
    with open(os.path.join(PKG, 'config', 'inertials.yaml'), 'w') as f:
        f.write('# GENERATED by tools/prepare_meshes.py - do not edit by hand.\n'
                f'# Uniform density ({density:.0f} kg/m^3) scaled so the arm weighs {TOTAL_MASS} kg.\n'
                '# com [m] in the link frame; inertia [kg m^2] about the COM, link-frame axes.\n')
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    for link, p in out.items():
        print(f'{link:10s} mass {p["mass"]:6.2f} kg  com {p["com"]}')


if __name__ == '__main__':
    main()
