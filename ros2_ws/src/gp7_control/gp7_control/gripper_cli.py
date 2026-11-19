"""Open / close the Robotiq 2F-85 through gripper_controller.

    ros2 run gp7_control gripper open
    ros2 run gp7_control gripper close              # fully closed (pads touch)
    ros2 run gp7_control gripper 0.06               # pad gap [m]
    ros2 run gp7_control gripper grasp 0.05         # close on a 5 cm object (default 20 N per pad)
    ros2 run gp7_control gripper grasp 0.05 --force 30
    ros2 run gp7_control gripper theta 0.4          # knuckle angle [rad] (0 open .. 0.8 closed)
"""
import argparse
import sys

import rclpy

from .commander import GP7Commander


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', help="'open', 'close', 'grasp', 'theta', or a pad gap in metres")
    ap.add_argument('value', nargs='?', type=float, help='object width [m] for grasp / angle [rad] for theta')
    ap.add_argument('--force', type=float, default=None, help='grip force per pad [N] for grasp (default gripper.yaml)')
    ap.add_argument('--duration', type=float, default=None, help='closing/opening time [s] (default: from the travel)')
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = GP7Commander('gripper_cli')
    try:
        if args.command == 'open':
            ok, msg = node.open_gripper(args.duration)
        elif args.command == 'close':
            ok, msg = node.close_gripper(args.duration)
        elif args.command in ('grasp', 'theta'):
            if args.value is None:
                ap.error(f'{args.command} needs a value, e.g. grasp 0.05 / theta 0.4')
            ok, msg = (node.grasp(args.value, args.force, args.duration) if args.command == 'grasp'
                       else node.gripper(args.value, args.duration))
        else:
            ok, msg = node.gripper_width(float(args.command), args.duration)
        print(('OK   ' if ok else 'FAIL ') + msg)
        if not ok:
            sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
