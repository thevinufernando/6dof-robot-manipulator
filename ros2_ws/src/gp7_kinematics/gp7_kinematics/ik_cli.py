"""Command-line IK / FK for the GP7.

    ros2 run gp7_kinematics ik_solver 0.55 -0.30 0.025               # top-down TCP pose
    ros2 run gp7_kinematics ik_solver 0.55 -0.30 0.025 --yaw 30
    ros2 run gp7_kinematics ik_solver 0.6 0.0 0.5 --rpy 180 0 0      # any orientation [deg]
    ros2 run gp7_kinematics ik_solver 0.56 0 0.815 --rpy 0 90 0 --tool flange
    ros2 run gp7_kinematics ik_solver --fk 0 -30 -20 0 -40 0         # forward kinematics [deg]

Positions are in metres in the base frame; the target is the gripper TCP unless
--tool flange is given.
"""
import argparse
import math

import numpy as np

from .kinematics import GP7Kinematics, pose_error, pose_from_xyz_rpy, top_down_pose


def _fmt_pose(T):
    p = T[:3, 3]
    z = T[:3, 2]
    return (f'position [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}] m,  '
            f'approach z [{z[0]:+.3f} {z[1]:+.3f} {z[2]:+.3f}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz', nargs='*', type=float, help='target position x y z [m]')
    ap.add_argument('--rpy', nargs=3, type=float, metavar=('R', 'P', 'Y'),
                    help='target orientation, fixed-axis roll pitch yaw [deg] (default: top-down)')
    ap.add_argument('--yaw', type=float, default=0.0, help='yaw of a top-down grasp [deg]')
    ap.add_argument('--tool', choices=['tcp', 'flange'], default='tcp')
    ap.add_argument('--ref', nargs=6, type=float, metavar='Q', help='reference joints [deg]')
    ap.add_argument('--no-limits', action='store_true', help='also show solutions outside the limits')
    ap.add_argument('--fk', nargs=6, type=float, metavar='Q', help='forward kinematics for q [deg]')
    a = ap.parse_args()

    kin = GP7Kinematics.from_config(tool=a.tool)

    if a.fk:
        T = kin.fk(np.radians(a.fk))
        print(f'FK ({a.tool}): {_fmt_pose(T)}')
        print(np.array2string(T, precision=4, suppress_small=True))
        return

    if len(a.xyz) != 3:
        ap.error('give x y z (or --fk q1..q6)')
    if a.rpy:
        T = pose_from_xyz_rpy(a.xyz, np.radians(a.rpy))
    else:
        T = top_down_pose(*a.xyz, yaw=math.radians(a.yaw))
    q_ref = np.radians(a.ref) if a.ref else None

    oc = kin.wrist_center(T)
    print(f'target ({a.tool}): {_fmt_pose(T)}')
    print(f'wrist centre: [{oc[0]:+.4f} {oc[1]:+.4f} {oc[2]:+.4f}] m')
    sols = kin.ik(T, q_ref=q_ref, respect_limits=not a.no_limits)
    if not sols:
        print('NO SOLUTION: out of reach or outside the joint limits')
        return
    print(f'{len(sols)} solution(s), closest to the reference first (deg):')
    print('      q1        q2        q3        q4        q5        q6    config        |dp| [m]')
    for s in sols:
        dp, _ = pose_error(kin.fk(s.q), T)
        cfg = f'{s.shoulder}/{s.elbow}/{s.wrist}'
        lim = '' if kin.within_limits(s.q) else '  (outside limits)'
        print(' '.join(f'{math.degrees(v):9.3f}' for v in s.q) + f'   {cfg:13s} {dp:.1e}{lim}')


if __name__ == '__main__':
    main()
