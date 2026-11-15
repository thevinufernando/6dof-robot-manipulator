#!/usr/bin/env python3
"""Convert the raw SolidWorks STL exports into URDF-ready meshes in the DH frames.

Two steps per link:
  1. place the raw part where it sits on the assembled robot at the HOME pose,
     in base-frame {0} coordinates:   p_0 = R @ p_mesh + t
     (R, t come from the joint-axis measurements of tools/measure_joint_axes.py;
     t is anchored to the DH frame origins O_i so it follows kinematics.yaml);
  2. express it in its own DH frame {i}:   p_i = (T_0^i at home)^-1 . p_0
so every URDF link frame IS the DH frame and visual/collision origins are identity.

It also writes convex-hull collision meshes and computes volume, centre of mass
and inertia per link (uniform density scaled to the GP7 datasheet weight).

Outputs (inside ros2_ws/src/gp7_description):
  meshes/visual/<link>.stl, meshes/collision/<link>.stl, config/inertials.yaml

Usage:  python3 tools/prepare_meshes.py
"""
import os
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from dh import fk_frames, load_dh
from stl_utils import load_stl, save_stl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STL_DIR = os.path.join(ROOT, '3d_designs', 'STL_files')
PKG = os.path.join(ROOT, 'ros2_ws', 'src', 'gp7_description')
TOTAL_MASS = 37.0  # kg, GP7 datasheet (docs/datasheets/motoman_GP7.pdf)

DH = load_dh()
T_HOME = fk_frames([0.0] * len(DH), DH)          # T_0^i at home, i = 0..6
O = [T[:3, 3] for T in T_HOME]                   # DH origins O_0..O_6 in {0}
X0 = np.array([1.0, 0.0, 0.0])                   # robot forward at home


def cols(ex, ey, ez):
    """Rotation whose columns are where mesh x, y, z axes point in base frame {0} at home."""
    R = np.array([ex, ey, ez], dtype=float).T
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    return R


# link -> (DH frame index, source STL, R, t): raw part pose in {0} at the home pose.
LINKS = {
    # Base: mesh is Y-up, origin on the floor under the S-axis; the cable
    # connector box is at mesh +X = back of the robot, so forward is mesh -X.
    'base_link': (0, 'base_link.stl', cols([-1, 0, 0], [0, 0, 1], [0, 1, 0]), O[0]),
    # S-head: mesh origin ON the L-axis at the centre plane (= O1); mesh Y up;
    # S-axis is 40 mm towards mesh +Z, so forward is mesh -Z.
    'link_1_s': (1, 'link1_s.stl', cols([0, -1, 0], [0, 0, 1], [-1, 0, 0]), O[1]),
    # Lower arm: mesh origin on the L-axis (= O1), arm hangs along mesh -Z.
    # It bows towards mesh +X, which is the BACK of the robot (see the GP7 photo).
    'link_2_l': (2, 'link2_L.stl', cols([-1, 0, 0], [0, 1, 0], [0, 0, -1]), O[1]),
    # U-casing: already Z-up, X-forward, origin on the U-axis (= O2).
    'link_3_u': (3, 'link3_U.stl', cols([1, 0, 0], [0, 1, 0], [0, 0, 1]), O[2]),
    # Forearm: mesh X along the R-axis; B-axis bores (mesh Z) sit 119.3 mm ahead
    # of the mesh origin -> mesh origin is 119.3 mm behind the wrist centre O4.
    'link_4_r': (4, 'link4_R.stl', cols([1, 0, 0], [0, 0, 1], [0, -1, 0]), O[4] - 0.1193 * X0),
    # Wrist: origin on the B-axis (= O5, the wrist centre), T-axis along mesh +Y.
    'link_5_b': (5, 'link5_B.stl', cols([0, 0, 1], [1, 0, 0], [0, 1, 0]), O[5]),
    # Flange: axis mesh X, outer face (with the Ø12 pin hole) at mesh x = +15 mm,
    # i.e. the face centre is O6.
    'link_6_t': (6, 'link6_T.stl', cols([1, 0, 0], [0, 1, 0], [0, 0, 1]), O[6] - 0.015 * X0),
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
    for link, (i, src, R, t) in LINKS.items():
        p0 = load_stl(os.path.join(STL_DIR, src)) @ R.T + np.asarray(t, float)  # home pose, {0}
        Ti = T_HOME[i]
        tris = (p0 - Ti[:3, 3]) @ Ti[:3, :3]                                       # into DH frame {i}
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
                '# com [m] in the link (DH) frame; inertia [kg m^2] about the COM, DH-frame axes.\n')
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    for link, p in out.items():
        print(f'{link:10s} mass {p["mass"]:6.2f} kg  com {p["com"]}')


if __name__ == '__main__':
    main()
