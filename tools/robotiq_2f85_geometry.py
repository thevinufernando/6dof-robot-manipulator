#!/usr/bin/env python3
"""Measure the Robotiq 2F-85 grasp geometry from its URDF + collision meshes.

The 2F-85 is a linkage: one knuckle angle theta (robotiq_85_left_knuckle_joint,
0 = open .. 0.8 rad = closed) moves both fingertips on an ARC while the mimic
joints keep the pads parallel. So the pad gap AND the pad height both depend on
theta. This script sweeps theta, places the fingertip collision meshes with
forward kinematics (mimic joints follow the knuckle) and measures, in the
gripper base frame (z = approach direction, fingers close along x):

  width(theta)            gap between the two pad faces
  pad_center_z(theta)     height of the pad centre above the mounting face
  pad_z_min/max(theta)    extent of the pad face along the approach direction
  tip_end_z(theta)        furthest point of the fingertips along z
  outer half-widths       of the whole gripper (for clearance checks)

and derives the TCP for a given object width plus the lever arms used by the
grip-force model. Writes gp7_description/config/gripper_geometry.yaml (generated), including the box
contact pads (fingertip link frame) that sit `pad.protrusion` proud of the mesh face
(gripper.yaml); widths are measured between those box pads.

Usage:
  xacro <robotiq_description>/urdf/robotiq_2f_85_gripper.urdf.xacro > /tmp/r85.urdf
  python3 tools/robotiq_2f85_geometry.py /tmp/r85.urdf [--object-width 0.05]
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from render_robot import forward_kinematics, mesh_path
from stl_utils import load_stl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG = os.path.join(ROOT, 'ros2_ws', 'src', 'gp7_description', 'config')
OUT = os.path.join(CFG, 'gripper_geometry.yaml')
P = 'robotiq_85_'
DRIVEN = P + 'left_knuckle_joint'
FACE_TOL = 0.0005          # m: vertices within this of the innermost x form the pad face


def collision_mesh(root, link):
    m = root.find(f"link[@name='{link}']/collision/geometry/mesh")
    return load_stl(mesh_path(m.get('filename'))).reshape(-1, 3)


def in_base(T, link, pts):
    Tb = np.linalg.inv(T[P + 'base_link']) @ T[link]
    return pts @ Tb[:3, :3].T + Tb[:3, 3]


def joint_axis_z(root, T, joint):
    """Height (base frame) of a joint's axis origin."""
    j = root.find(f"joint[@name='{joint}']")
    Tb = np.linalg.inv(T[P + 'base_link']) @ T[j.find('parent').get('link')]
    o = [float(v) for v in j.find('origin').get('xyz').split()]
    return float((Tb @ np.r_[o, 1.0])[2])


def pad_box(root, side, protrusion, thickness):
    """Box pad on a fingertip, in the fingertip LINK frame: its outer face is the mesh pad
    face moved `protrusion` towards the object, same rectangle as the mesh face."""
    v = collision_mesh(root, f'{P}{side}_finger_tip_link')
    # left tip: the face looks along -x of the base; find the face axis in the link frame
    # by the extreme coordinate along the link x axis (pad faces are normal to link x).
    sgn = -1.0 if side == 'left' else 1.0
    x_face = v[:, 0].min() if sgn < 0 else v[:, 0].max()
    face = v[np.abs(v[:, 0] - x_face) < FACE_TOL]
    x_out = x_face + sgn * protrusion
    cx = x_out - sgn * thickness / 2
    cy, cz = (face[:, 1].min() + face[:, 1].max()) / 2, (face[:, 2].min() + face[:, 2].max()) / 2
    return {'xyz': [round(float(c), 6) for c in (cx, cy, cz)],
            'size': [thickness, round(float(np.ptp(face[:, 1])), 6), round(float(np.ptp(face[:, 2])), 6)]}


def _plain(x):
    """numpy scalars/containers -> plain Python (yaml.safe_dump cannot write numpy types)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urdf')
    ap.add_argument('--object-width', type=float, default=0.05, help='width the TCP is defined for [m]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    root = ET.parse(a.urdf).getroot()
    with open(os.path.join(CFG, 'gripper.yaml')) as f:
        pad = yaml.safe_load(f)['pad']
    prot = float(pad['protrusion'])
    lim = root.find(f"joint[@name='{DRIVEN}']/limit")
    theta = np.round(np.linspace(float(lim.get('lower')), float(lim.get('upper')), 161), 6)
    links = [l.get('name') for l in root.findall('link') if l.find('collision') is not None]
    meshes = {l: collision_mesh(root, l) for l in links}

    rows = []
    for th in theta:
        T = forward_kinematics(root, {DRIVEN: float(th)})
        left = in_base(T, P + 'left_finger_tip_link', meshes[P + 'left_finger_tip_link'])
        right = in_base(T, P + 'right_finger_tip_link', meshes[P + 'right_finger_tip_link'])
        # the left finger sits on +x and closes towards -x: its pad face is its innermost (min x) side
        lface = left[left[:, 0] < left[:, 0].min() + FACE_TOL]
        rface = right[right[:, 0] > right[:, 0].max() - FACE_TOL]
        allpts = np.vstack([in_base(T, l, m) for l, m in meshes.items()])
        rows.append(dict(
            theta=float(th),
            width=float(left[:, 0].min() - right[:, 0].max()) - 2 * prot,   # between the box pads
            pad_center_z=float((lface[:, 2].min() + lface[:, 2].max()) / 2),
            pad_z_min=float(lface[:, 2].min()), pad_z_max=float(lface[:, 2].max()),
            pad_face_tilt_mm=float(np.ptp(lface[:, 0]) * 1000),
            tip_end_z=float(max(left[:, 2].max(), right[:, 2].max())),
            outer_half_x=float(np.abs(allpts[:, 0]).max()),
            outer_half_y=float(np.abs(allpts[:, 1]).max()),
            lface_center_x=float(lface[:, 0].mean()), rface_center_x=float(rface[:, 0].mean()),
        ))

    w = np.array([r['width'] for r in rows])
    if not np.all(np.diff(w) < 0):
        raise SystemExit('pad gap is not monotonic in theta - check the mesh/joint conventions')
    # theta at which the pads touch an object of the given width (w decreases with theta)
    th_obj = float(np.interp(a.object_width, w[::-1], theta[::-1]))
    T = forward_kinematics(root, {DRIVEN: th_obj})
    col = lambda k: np.array([r[k] for r in rows])
    pad_c = float(np.interp(th_obj, theta, col('pad_center_z')))
    knuckle_z = joint_axis_z(root, T, DRIVEN)
    tip_axis_z = joint_axis_z(root, T, P + 'left_finger_tip_joint')
    data = {
        'description': 'GENERATED by tools/robotiq_2f85_geometry.py - do not edit. Gripper base '
                       'frame: z = approach (from the mounting face), x = closing direction.',
        'driven_joint': DRIVEN,
        'open_width': float(w[0]), 'closed_width': float(w[-1]),
        'object_width': a.object_width,
        'theta_at_object_width': round(th_obj, 6),
        'tcp_z': round(pad_c, 6),     # pad centre height at the object width (from the mounting face)
        'pad_half_height': round(float(np.interp(th_obj, theta, (col('pad_z_max') - col('pad_z_min')) / 2)), 6),
        'tip_end_z_open': round(rows[0]['tip_end_z'], 6),
        'tip_end_z_at_object': round(float(np.interp(th_obj, theta, col('tip_end_z'))), 6),
        'outer_half_x_open': round(rows[0]['outer_half_x'], 6),
        'outer_half_y': round(max(col('outer_half_y')), 6),
        'knuckle_axis_z': round(knuckle_z, 6),
        'tip_axis_z_at_object': round(tip_axis_z, 6),
        'lever_knuckle_to_pad': round(pad_c - knuckle_z, 6),   # vertical arm of the grip force about the knuckle
        'lever_tip_to_pad': round(pad_c - tip_axis_z, 6),       # ... and about the finger-tip joint
        'pad_protrusion': prot,
        'pads': {side: pad_box(root, side, prot, float(pad['thickness'])) for side in ('left', 'right')},
        'table': {k: [round(r[k], 6) for r in rows] for k in
                  ('theta', 'width', 'pad_center_z', 'pad_z_min', 'pad_z_max', 'tip_end_z')},
    }
    data = _plain(data)
    with open(a.out, 'w') as f:
        f.write('# GENERATED by tools/robotiq_2f85_geometry.py - do not edit by hand.\n')
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, width=120)

    print(f'pad gap: {w[0] * 1000:.1f} mm open (theta 0) -> {w[-1] * 1000:.1f} mm at theta {theta[-1]}')
    print(f'pad centre height: {rows[0]["pad_center_z"] * 1000:.1f} mm open -> '
          f'{rows[-1]["pad_center_z"] * 1000:.1f} mm closed (arc motion)')
    print(f'max pad-face tilt over the sweep: {max(col("pad_face_tilt_mm")):.2f} mm across the face')
    print(f'object {a.object_width * 1000:.0f} mm: theta = {th_obj:.4f} rad, TCP (pad centre) z = {pad_c * 1000:.1f} mm, '
          f'pad half height {data["pad_half_height"] * 1000:.1f} mm')
    print(f'fingertip end z: open {data["tip_end_z_open"] * 1000:.1f} mm, at object {data["tip_end_z_at_object"] * 1000:.1f} mm')
    print(f'outer half widths: x (closing, open) {data["outer_half_x_open"] * 1000:.1f} mm, y {data["outer_half_y"] * 1000:.1f} mm')
    print(f'levers: knuckle axis -> pad {data["lever_knuckle_to_pad"] * 1000:.1f} mm, tip axis -> pad {data["lever_tip_to_pad"] * 1000:.1f} mm')
    print(f'written {a.out}')


if __name__ == '__main__':
    main()
