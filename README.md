# GP7 Pick-and-Place Simulation

A simulation of the **YASKAWA Motoman GP7**, a 6-axis industrial robot arm, picking up a small box and placing it on top of another box, keeping it upright the whole way. It runs in **Gazebo Harmonic** with **ROS 2 Jazzy**.

The robot model is built from the manufacturer's CAD files, and every link frame follows the Denavit–Hartenberg convention, so the kinematics in code match the model exactly. A two-finger parallel gripper, modelled on the Robotiq 2F-85, is mounted on the flange.

<!-- Add screenshots here -->

## Features

- GP7 robot description (URDF/xacro) with link frames that match the D-H table
- Two-finger gripper with a tool centre point (TCP) frame, controllable from ROS
- Gazebo world with the robot at the origin, a target box (Box A) and a box to pick (Box B)
- Box B can be placed anywhere at launch, at a random reachable spot, or moved while the simulation runs
- Closed-form inverse kinematics for the gripper TCP (all 8 arm configurations, joint limits, singularities)
- Joint limits and speeds taken from the GP7 datasheet

**In progress:** trajectory planning with a smooth velocity profile, and the full pick-and-place sequence.

## Requirements

- Ubuntu 24.04
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install)
- ROS packages:

  ```bash
  sudo apt install ros-jazzy-ros-gz ros-jazzy-xacro ros-jazzy-robot-state-publisher \
                   ros-jazzy-joint-state-publisher-gui ros-jazzy-rviz2
  ```

- Python 3 with `numpy` and `pyyaml` (plus `scipy` and `matplotlib` for the scripts in `tools/`)

## Project structure

```
.
├── 3d_designs/            # GP7 CAD files (SolidWorks) and per-link STL exports
├── docs/                  # GP7 datasheet and D-H frame assignment
├── ros2_ws/src/
│   ├── gp7_description/   # URDF/xacro, meshes, D-H table, joint limits, gripper, RViz
│   ├── gp7_kinematics/    # Forward and inverse kinematics (Python), IK command-line tool
│   └── gp7_gazebo/        # Gazebo world, box models, launch file, ROS–Gazebo bridge
└── tools/                 # Offline helpers: mesh preparation, D-H utilities, FK check
```

The D-H table lives in `ros2_ws/src/gp7_description/config/kinematics.yaml`. The URDF and the tools both read from it.

## Build

```bash
git clone <repository-url>
cd <repository>/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

**Gazebo simulation** (robot, gripper, Box A and Box B):

```bash
ros2 launch gp7_gazebo gazebo.launch.py
```

Useful launch arguments:

| Argument | Default | Description |
|---|---|---|
| `rviz` | `false` | Also open RViz |
| `headless` | `false` | Run Gazebo without the GUI |
| `box_b_x`, `box_b_y`, `box_b_yaw` | `0.55`, `-0.30`, `0` | Box B position (m) and yaw (deg) |
| `box_b_random` | `false` | Place Box B at a random reachable location |
| `box_a_x`, `box_a_y` | `1.0`, `1.0` | Box A position (m) |

**RViz only**, with a slider for each joint:

```bash
ros2 launch gp7_description display.launch.py
```

## Usage

Open and close the gripper (`0.0` = open, `0.0425` = closed):

```bash
ros2 topic pub --once -w 1 /gp7/gripper/cmd_pos std_msgs/msg/Float64 "{data: 0.0425}"
```

Move Box B while the simulation is running:

```bash
ros2 run gp7_gazebo move_box_b.py 0.5 0.2      # x y [yaw_deg]
ros2 run gp7_gazebo move_box_b.py --random
```

Read Box B's current pose:

```bash
ros2 topic echo /box_b/pose
```

Solve the inverse kinematics for a gripper pose (position in metres; by default the gripper points straight down):

```bash
ros2 run gp7_kinematics ik_solver 0.55 -0.30 0.025            # top-down grasp
ros2 run gp7_kinematics ik_solver 0.5 0.2 0.6 --rpy 150 20 40 # any orientation (deg)
```

## License

This project is released under the [MIT License](LICENSE). The GP7 CAD models and datasheet belong to YASKAWA.
