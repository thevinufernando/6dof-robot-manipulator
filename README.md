# GP7 Pick-and-Place Simulation

A simulation of the **YASKAWA Motoman GP7**, a 6-axis industrial robot arm, picking up a small box and placing it on top of another box, keeping it upright the whole way. It runs in **Gazebo Harmonic** with **ROS 2 Jazzy**.

The robot model is built from the manufacturer's CAD files, and every link frame follows the Denavit–Hartenberg convention, so the kinematics in code match the model exactly. A two-finger parallel gripper, modelled on the Robotiq 2F-85, is mounted on the flange.

<!-- Add screenshots here -->

## Features

- GP7 robot description (URDF/xacro) with link frames that match the D-H table
- Two-finger gripper with a tool centre point (TCP) frame
- ros2_control in Gazebo: velocity-controlled arm and force-controlled gripper, both driven through `joint_trajectory_controller`
- Gazebo world with the robot at the origin, a target box (Box A) and a box to pick (Box B)
- Box B can be placed anywhere at launch, at a random reachable spot, or moved while the simulation runs
- Closed-form inverse kinematics for the gripper TCP (all 8 arm configurations, joint limits, singularities)
- Trajectory planner with trapezoidal and S-curve velocity profiles: synchronized joint-space moves, and Cartesian moves that keep the gripper orientation fixed while carrying
- Joint limits and speeds taken from the GP7 datasheet

- Complete pick and place in Gazebo: Box B is picked from any reachable spot (and yaw) and placed on the centre of Box A while staying upright

## Requirements

- Ubuntu 24.04
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install)
- ROS packages:

  ```bash
  sudo apt install ros-jazzy-ros-gz ros-jazzy-xacro ros-jazzy-robot-state-publisher \
                   ros-jazzy-joint-state-publisher-gui ros-jazzy-rviz2 \
                   ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-gz-ros2-control
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
│   ├── gp7_control/       # Move to end-effector poses, gripper commands (ROS 2 nodes)
│   ├── gp7_motion/        # Trajectory planning: velocity profiles, Cartesian paths, pick & place
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

**Gazebo simulation** (robot, gripper, Box A, Box B and the controllers):

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
| `box_a_x`, `box_a_y` | `0.5`, `0.5` | Box A position (m) |
| `sim_control` | `ros2_control` | `ros2_control`, or `gz_plugins` for Gazebo-only joint control |

**RViz only**, with a slider for each joint:

```bash
ros2 launch gp7_description display.launch.py
```

## Controllers

| Controller | Type | Joints | Command interface |
|---|---|---|---|
| `joint_state_broadcaster` | `joint_state_broadcaster/JointStateBroadcaster` | all | – |
| `arm_controller` | `joint_trajectory_controller/JointTrajectoryController` | `joint_1_s` … `joint_6_t` | velocity (feed-forward + position feedback) |
| `gripper_controller` | `joint_trajectory_controller/JointTrajectoryController` | both finger joints | effort (PID) |

The configuration is in `ros2_ws/src/gp7_gazebo/config/controllers.yaml`.

## Usage

Move the gripper to an end-effector pose. The inverse kinematics picks the joint angles closest to the current ones, and `arm_controller` moves there (position in metres; by default the gripper points straight down):

```bash
ros2 run gp7_control move_to_pose 0.55 -0.30 0.20               # top-down
ros2 run gp7_control move_to_pose 0.45 0.10 0.30 --yaw 30       # rotated about the vertical
ros2 run gp7_control move_to_pose 0.50 0.20 0.60 --rpy 150 20 40 # any orientation (deg)
ros2 run gp7_control move_to_pose --joints 0 -20 -10 0 -60 0    # joint angles (deg)
```

Open and close the gripper:

```bash
ros2 run gp7_control gripper open
ros2 run gp7_control gripper close
ros2 run gp7_control gripper grasp 0.05        # close on a 5 cm object with 20 N per finger
```

Send pose goals as messages instead:

```bash
ros2 run gp7_control pose_goal_node
ros2 topic pub --once /gp7/tcp_goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: base_link}, pose: {position: {x: 0.55, y: -0.3, z: 0.2},
    orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}}}"
```

The controllers can also be driven directly through their actions:

```bash
ros2 action send_goal /arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_1_s, joint_2_l, joint_3_u, joint_4_r, joint_5_b, joint_6_t],
    points: [{positions: [0.3, -0.4, 0.2, 0.0, -0.6, 0.0], time_from_start: {sec: 3}}]}}"
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

## Pick and place

Start the simulation, then run the task. The node reads Box B's position from Gazebo, plans the whole motion, shows the planned path in RViz, and executes it:

```bash
ros2 launch gp7_gazebo gazebo.launch.py rviz:=true box_b_random:=true
ros2 launch gp7_control pick_place.launch.py
```

| Argument | Default | Description |
|---|---|---|
| `profile` | from config (`scurve`) | `scurve` or `trapezoidal` |
| `dry_run` | `false` | Plan and show the path in RViz without moving |
| `log_dir` | – | Save the planned and measured motion as CSV files |
| `box_a_x`, `box_a_y` | `0.5`, `0.5` | Box A position (must match the simulation) |

To run it again, move Box B (`ros2 run gp7_gazebo move_box_b.py --random`) and launch `pick_place.launch.py` once more. Plots of a logged run (planned vs measured joint velocities, Box B height and tilt):

```bash
ros2 launch gp7_control pick_place.launch.py log_dir:=runs/run1
ros2 run gp7_motion plot_run runs/run1
```

## Trajectory planning

The planner can be run offline (no simulation needed). This plans the whole pick and place for a Box B position and writes plots of the velocity profiles, joint trajectories, TCP path and gripper orientation:

```bash
ros2 run gp7_motion plot_plan 0.55 -0.30 --yaw 25 --out plan_plots
ros2 run gp7_motion plot_plan 0.55 -0.30 --profile trapezoidal
```

Speeds, accelerations, heights and Box A's position are set in `ros2_ws/src/gp7_motion/config/pick_place.yaml`.

## Troubleshooting

If `ros2 topic list` or `ros2 control ...` commands hang, the ROS 2 daemon is stuck. Restart it:

```bash
ros2 daemon stop || pkill -f ros2cli.daemon
ros2 daemon start
```

## License

This project is released under the [MIT License](LICENSE). The GP7 CAD models and datasheet belong to YASKAWA.
