"""Open / close the GP7 gripper through gripper_controller.

    ros2 run gp7_control gripper open
    ros2 run gp7_control gripper close              # fully closed (fingers touch)
    ros2 run gp7_control gripper 0.02               # finger position [m per finger]
    ros2 run gp7_control gripper grasp 0.05         # close on a 5 cm object with 20 N
    ros2 run gp7_control gripper grasp 0.05 --force 30
"""
import argparse
import sys

import rclpy

from .commander import GP7Commander


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', help="'open', 'close', 'grasp', or a finger position in metres")
    ap.add_argument('width', nargs='?', type=float, help='object width [m] for grasp')
    ap.add_argument('--force', type=float, default=20.0, help='grip force per finger [N] for grasp')
    ap.add_argument('--duration', type=float, default=0.5, help='closing/opening time [s]')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = GP7Commander('gripper_cli')
    try:
        if args.command == 'open':
            ok, msg = node.gripper(0.0, args.duration)
        elif args.command == 'close':
            ok, msg = node.gripper(node.stroke, args.duration)
        elif args.command == 'grasp':
            if args.width is None:
                ap.error('grasp needs the object width, e.g. grasp 0.05')
            ok, msg = node.grasp(args.width, args.force, args.duration)
        else:
            ok, msg = node.gripper(float(args.command), args.duration)
        print(('OK   ' if ok else 'FAIL ') + msg)
        if not ok:
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
