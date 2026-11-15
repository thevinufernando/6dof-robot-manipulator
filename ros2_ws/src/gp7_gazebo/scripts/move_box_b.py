#!/usr/bin/env python3
"""Move Box B in the running Gazebo simulation (no restart needed).

    ros2 run gp7_gazebo move_box_b.py 0.55 -0.30          # x y [m]
    ros2 run gp7_gazebo move_box_b.py 0.55 -0.30 30       # x y yaw [deg]
    ros2 run gp7_gazebo move_box_b.py --random            # random reachable spot
    ros2 run gp7_gazebo move_box_b.py --random --seed 7   # reproducible random spot

Box B is placed upright on the ground (its bottom face at z = 0). Uses Gazebo's
/world/<world>/set_pose service through the `gz service` CLI.
"""
import argparse
import math
import random
import subprocess
import sys

# Region where the GP7 can pick a 5 cm cube from above with the gripper
# pointing straight down: an annulus around the S-axis, inside the S-axis range.
R_MIN, R_MAX = 0.40, 0.75          # m, horizontal distance from the S-axis
ANGLE_MAX_DEG = 150.0              # |S-axis angle| (datasheet limit is 170)


def random_reachable_xy(rng):
    """Uniformly distributed point in the reachable annulus sector."""
    r = math.sqrt(rng.uniform(R_MIN ** 2, R_MAX ** 2))   # sqrt -> uniform over area
    a = math.radians(rng.uniform(-ANGLE_MAX_DEG, ANGLE_MAX_DEG))
    return r * math.cos(a), r * math.sin(a)


def set_pose(world, name, x, y, z, yaw_deg):
    qz, qw = math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)
    req = (f"name: '{name}', position: {{x: {x}, y: {y}, z: {z}}}, "
           f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}")
    cmd = ['gz', 'service', '-s', f'/world/{world}/set_pose', '--reqtype', 'gz.msgs.Pose',
           '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.returncode == 0 and 'true' in out.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('x', nargs='?', type=float)
    p.add_argument('y', nargs='?', type=float)
    p.add_argument('yaw', nargs='?', type=float, default=0.0, help='yaw about Z [deg]')
    p.add_argument('--random', action='store_true', help='pick a random reachable location')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--world', default='gp7_world')
    a = p.parse_args()

    if a.random:
        a.x, a.y = random_reachable_xy(random.Random(a.seed))
    elif a.x is None or a.y is None:
        p.error('give x y, or --random')

    r = math.hypot(a.x, a.y)
    if not (R_MIN <= r <= R_MAX):
        print(f'warning: r = {r:.3f} m is outside the comfortable pick region '
              f'[{R_MIN}, {R_MAX}] m', file=sys.stderr)
    if set_pose(a.world, 'box_b', a.x, a.y, 0.0, a.yaw):
        print(f'Box B moved to x={a.x:.3f} y={a.y:.3f} yaw={a.yaw:.1f} deg (r={r:.3f} m)')
    else:
        sys.exit('set_pose failed - is Gazebo running with world "%s"?' % a.world)


if __name__ == '__main__':
    main()
