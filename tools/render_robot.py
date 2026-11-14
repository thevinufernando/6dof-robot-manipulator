#!/usr/bin/env python3
"""Render the GP7 assembled from its URDF (offline sanity check, no ROS needed).

Walks the URDF joint tree with forward kinematics (homogeneous transforms),
places every link's visual mesh, and draws shaded orthographic views with the
link frames. Useful to check that meshes meet at the joints in any pose.

Usage:
  xacro ros2_ws/src/gp7_description/urdf/gp7.urdf.xacro > /tmp/gp7.urdf
  python3 tools/render_robot.py /tmp/gp7.urdf out.png [q1 q2 q3 q4 q5 q6 (deg)] ...
Each extra group of 6 angles adds another row of views.
"""
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from stl_utils import load_stl

PKG_ROOT = os.path.join(os.path.dirname(__file__), '..', 'ros2_ws', 'src')


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, q):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def homog(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def forward_kinematics(urdf_root, q):
    """Return {link: 4x4 world transform} for joint values q (dict, rad)."""
    joints = urdf_root.findall('joint')
    child_of = {j.find('parent').get('link'): [] for j in joints}
    for j in joints:
        child_of[j.find('parent').get('link')].append(j)
    root = (set(child_of) - {j.find('child').get('link') for j in joints}).pop()
    T = {root: np.eye(4)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for j in child_of.get(parent, []):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            Tj = homog(rpy_matrix(*rpy), xyz)
            if j.get('type') in ('revolute', 'continuous'):
                axis = [float(v) for v in j.find('axis').get('xyz').split()]
                Tj = Tj @ homog(axis_angle(axis, q.get(j.get('name'), 0.0)), [0, 0, 0])
            child = j.find('child').get('link')
            T[child] = T[parent] @ Tj
            stack.append(child)
    return T


def mesh_path(uri):
    return os.path.join(PKG_ROOT, uri.replace('package://', ''))


def render(urdf_file, out_png, poses_deg):
    root = ET.parse(urdf_file).getroot()
    joint_names = [j.get('name') for j in root.findall('joint') if j.get('type') == 'revolute']
    meshes = {}
    for link in root.findall('link'):
        m = link.find('visual/geometry/mesh')
        if m is not None:
            col = link.find('visual/material/color')
            rgba = [float(v) for v in col.get('rgba').split()] if col is not None else [.5, .5, .5, 1]
            meshes[link.get('name')] = (load_stl(mesh_path(m.get('filename'))), rgba[:3])

    # (title, rotation world->view) ; view x = screen right, y = screen up, z = towards viewer
    views = [('side (from -Y)', np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)),
             ('front (from +X)', np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float))]
    az, el = np.radians(-35), np.radians(25)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    iso = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ Rz
    iso = np.array([[1, 0, 0], [0, np.cos(el), np.sin(el)], [0, -np.sin(el), np.cos(el)]]) @ iso
    views.append(('iso', iso))
    light = np.array([0.3, 0.5, 1.0]) / np.linalg.norm([0.3, 0.5, 1.0])

    fig, axs = plt.subplots(len(poses_deg), 3, figsize=(18, 6.5 * len(poses_deg)), squeeze=False)
    for r, pose in enumerate(poses_deg):
        q = {n: np.radians(v) for n, v in zip(joint_names, pose)}
        T = forward_kinematics(root, q)
        world_tris, colours = [], []
        for name, (tris, rgb) in meshes.items():
            world_tris.append(tris @ T[name][:3, :3].T + T[name][:3, 3])
            colours.append(np.tile(rgb, (len(tris), 1)))
        W, C = np.concatenate(world_tris), np.concatenate(colours)
        for c, (title, V) in enumerate(views):
            ax = axs[r, c]
            P = W @ V.T
            n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.35 + 0.65 * np.abs(n @ light)
            order = np.argsort(P[:, :, 2].mean(1))
            ax.add_collection(PolyCollection(P[order][:, :, :2],
                                             facecolors=np.clip(C[order] * shade[order, None] * 1.4, 0, 1),
                                             edgecolors='none'))
            for link, Tl in T.items():   # link frames: x red, y green, z blue
                o = V @ Tl[:3, 3]
                for k, colr in enumerate('rgb'):
                    e = V @ (Tl[:3, 3] + 0.08 * Tl[:3, k])
                    ax.plot([o[0], e[0]], [o[1], e[1]], colr, lw=2, zorder=5)
            ax.axhline(0, color='k', lw=0.8)
            ax.autoscale()
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.set_title(f'{title}   q = {list(pose)} deg')
    plt.tight_layout()
    plt.savefig(out_png, dpi=60)


if __name__ == '__main__':
    vals = [float(v) for v in sys.argv[3:]]
    poses = [vals[i:i + 6] for i in range(0, len(vals), 6)] or [[0] * 6]
    render(sys.argv[1], sys.argv[2], poses)
