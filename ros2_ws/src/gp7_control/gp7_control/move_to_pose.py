"""Move the GP7 so the gripper TCP reaches a pose (IK + arm_controller).

    ros2 run gp7_control move_to_pose 0.55 -0.30 0.20                 # gripper pointing down
    ros2 run gp7_control move_to_pose 0.55 -0.30 0.20 --yaw 30        # ... rotated about Z
    ros2 run gp7_control move_to_pose 0.50 0.20 0.60 --rpy 150 20 40  # any orientation [deg]
    ros2 run gp7_control move_to_pose 0.4 0.0 0.5 --duration 4        # fixed move time [s]
    ros2 run gp7_control move_to_pose 0.4 0.0 0.5 --dry-run           # only solve the IK
    ros2 run gp7_control move_to_pose --joints 0 -30 -20 0 -40 0      # joint goal [deg]

Position in metres in the base frame (= world). The target frame is gripper_tcp
(or the flange with --tool flange). Orientation defaults to top-down (Z of the
tool pointing at the floor).
"""
import argparse
import math
import sys

import numpy as np
import rclpy

from gp7_kinematics import pose_from_xyz_rpy, top_down_pose

from .commander import GP7Commander


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz', nargs='*', type=float, help='TCP position x y z [m]')
    ap.add_argument('--rpy', nargs=3, type=float, metavar=('R', 'P', 'Y'),
                    help='orientation as fixed-axis roll pitch yaw [deg] (default: top-down)')
    ap.add_argument('--yaw', type=float, default=0.0, help='yaw of the top-down orientation [deg]')
    ap.add_argument('--tool', choices=['tcp', 'flange'], default='tcp')
    ap.add_argument('--duration', type=float, help='move time [s] (default: from joint speed limits)')
    ap.add_argument('--speed-scale', type=float, default=0.25,
                    help='fraction of the datasheet joint speeds for the automatic duration')
    ap.add_argument('--joints', nargs=6, type=float, metavar='Q', help='joint goal [deg] instead of a pose')
    ap.add_argument('--dry-run', action='store_true', help='solve and print, do not move')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = GP7Commander('move_to_pose', tool=args.tool)
    try:
        q0 = node.current_q()
        if args.joints:
            q_goal = np.radians(args.joints)
            print(f'joint goal: {np.round(args.joints, 3)} deg')
            if args.dry_run:
                return
            ok, msg = node.move_joints(q_goal, args.duration, args.speed_scale)
            print(('OK   ' if ok else 'FAIL ') + msg)
            return

        if len(args.xyz) != 3:
            ap.error('give x y z, or --joints q1..q6')
        T = (pose_from_xyz_rpy(args.xyz, np.radians(args.rpy)) if args.rpy
             else top_down_pose(*args.xyz, yaw=math.radians(args.yaw)))
        sols = node.kin.ik(T, q_ref=q0)
        if not sols:
            print('FAIL no IK solution: pose out of reach or outside the joint limits')
            sys.exit(1)
        sol = sols[0]
        dur = args.duration or node.auto_duration(q0, sol.q, args.speed_scale)
        print(f'current joints : {np.round(np.degrees(q0), 2)} deg')
        print(f'IK solution    : {np.round(np.degrees(sol.q), 2)} deg '
              f'({sol.shoulder}/{sol.elbow}/{sol.wrist}; {len(sols)} solution(s), closest chosen)')
        print(f'move time      : {dur:.2f} s')
        if args.dry_run:
            return
        ok, msg, _ = node.move_to_pose(T, dur, args.speed_scale)
        print(('OK   ' if ok else 'FAIL ') + msg)
        if not ok:
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
