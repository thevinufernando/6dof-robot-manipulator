#!/usr/bin/env python3
"""Verify the URDF against the DH table (forward kinematics check).

For random joint vectors q (within limits) compare every URDF link frame
(link_i, computed by walking the URDF joint tree) with the DH product
T_0^i = A_1 A_2 ... A_i  (computed from kinematics.yaml by tools/dh.py).

Optionally also compare with another URDF (e.g. an older model with a
different joint-sign convention): pass it with a sign vector, and the script
checks that every mesh vertex lands at the same place in the world.

Usage:
  xacro ros2_ws/src/gp7_description/urdf/gp7.urdf.xacro > /tmp/gp7.urdf
  python3 tools/check_dh.py /tmp/gp7.urdf [other.urdf s1 s2 s3 s4 s5 s6]
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from dh import fk_frames, load_dh
from render_robot import forward_kinematics, mesh_path
from stl_utils import load_stl

LIMITS_YAML = __import__('dh').KIN_YAML.replace('kinematics.yaml', 'joint_limits.yaml')


def random_configs(n, rows, seed=1):
    lim = yaml.safe_load(open(LIMITS_YAML))
    rng = np.random.default_rng(seed)
    lo = np.radians([lim[r['joint']]['lower_deg'] for r in rows])
    hi = np.radians([lim[r['joint']]['upper_deg'] for r in rows])
    return [np.zeros(len(rows))] + [rng.uniform(lo, hi) for _ in range(n)]


def main():
    rows = load_dh()
    urdf = ET.parse(sys.argv[1]).getroot()
    names = [r['joint'] for r in rows]
    links = ['base_link'] + [r['link'] for r in rows]
    configs = random_configs(200, rows)

    worst_p = worst_R = 0.0
    for q in configs:
        T_urdf = forward_kinematics(urdf, dict(zip(names, q)))
        T_dh = fk_frames(q, rows)
        for i, link in enumerate(links):
            worst_p = max(worst_p, np.abs(T_urdf[link][:3, 3] - T_dh[i][:3, 3]).max())
            worst_R = max(worst_R, np.abs(T_urdf[link][:3, :3] - T_dh[i][:3, :3]).max())
    print(f'URDF link frames vs DH product over {len(configs)} configs: '
          f'max position error {worst_p:.2e} m, max rotation-matrix error {worst_R:.2e}')

    T6 = fk_frames(np.zeros(6), rows)[6]
    print(f'Home pose flange (frame 6) origin: {np.round(T6[:3, 3], 6)} m,  '
          f'x6={np.round(T6[:3, 0], 3)} z6={np.round(T6[:3, 2], 3)}')

    if len(sys.argv) > 2:
        other = ET.parse(sys.argv[2]).getroot()
        sign = np.array([float(s) for s in sys.argv[3:9]])
        meshes = {}
        for link in links:
            m = urdf.find(f"link[@name='{link}']/visual/geometry/mesh")
            mo = other.find(f"link[@name='{link}']/visual/geometry/mesh")
            meshes[link] = (load_stl(mesh_path(m.get('filename')))[::25].reshape(-1, 3),
                            load_stl(mesh_path(mo.get('filename')))[::25].reshape(-1, 3))
        print(f'Mesh placement vs {sys.argv[2]} (joint map q_other = {sign.astype(int).tolist()} * q):')
        for link in links:
            err = 0.0
            for q in configs[:50]:
                Ta = forward_kinematics(urdf, dict(zip(names, q)))[link]
                Tb = forward_kinematics(other, dict(zip(names, sign * q)))[link]
                va, vb = meshes[link]
                if len(va) != len(vb):
                    err = np.nan
                    break
                err = max(err, np.abs((va @ Ta[:3, :3].T + Ta[:3, 3]) - (vb @ Tb[:3, :3].T + Tb[:3, 3])).max())
            print(f'  {link:10s} max vertex deviation {err * 1e3:9.4f} mm')


if __name__ == '__main__':
    main()
