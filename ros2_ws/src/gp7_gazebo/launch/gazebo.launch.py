"""Start Gazebo Harmonic with the GP7 (+ gripper) at the world origin, Box A and Box B.

    ros2 launch gp7_gazebo gazebo.launch.py                          # Gazebo GUI
    ros2 launch gp7_gazebo gazebo.launch.py rviz:=true               # + RViz
    ros2 launch gp7_gazebo gazebo.launch.py headless:=true           # server only
    ros2 launch gp7_gazebo gazebo.launch.py box_b_x:=0.5 box_b_y:=-0.4 box_b_yaw:=20
    ros2 launch gp7_gazebo gazebo.launch.py box_b_random:=true [box_b_seed:=7]

While running, move Box B without restarting:
    ros2 run gp7_gazebo move_box_b.py 0.55 -0.30 [yaw_deg]  |  --random

Pipeline:
  xacro -> robot_description -> robot_state_publisher
  ros_gz_sim create  : spawns "gp7" (gz_ros2_control loads controllers.yaml), "box_a", "box_b"
  spawner            : joint_state_broadcaster, arm_controller, gripper_controller
  ros_gz_bridge      : /clock, /box_b/pose  (Gazebo -> ROS)
"""
import math
import os
import random

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Same pick region as scripts/move_box_b.py: annulus around the S-axis, inside the
# S-axis range, and not on (or right next to) Box A.
R_MIN, R_MAX, ANGLE_MAX_DEG = 0.40, 0.75, 150.0
BOX_A_HALF, BOX_A_MARGIN = 0.15, 0.10


def random_box_b_xy(rng, ax, ay):
    while True:
        r = math.sqrt(rng.uniform(R_MIN ** 2, R_MAX ** 2))
        a = math.radians(rng.uniform(-ANGLE_MAX_DEG, ANGLE_MAX_DEG))
        x, y = r * math.cos(a), r * math.sin(a)
        if max(abs(x - ax), abs(y - ay)) > BOX_A_HALF + BOX_A_MARGIN:
            return x, y


def spawn_boxes(context):
    gz_pkg = get_package_share_directory('gp7_gazebo')
    cfg = {k: LaunchConfiguration(k).perform(context) for k in
           ('box_a_x', 'box_a_y', 'box_b_x', 'box_b_y', 'box_b_yaw', 'box_b_random', 'box_b_seed')}
    ax, ay = float(cfg['box_a_x']), float(cfg['box_a_y'])
    bx, by, byaw = float(cfg['box_b_x']), float(cfg['box_b_y']), float(cfg['box_b_yaw'])
    if cfg['box_b_random'].lower() == 'true':
        rng = random.Random(int(cfg['box_b_seed']) if cfg['box_b_seed'] else None)
        bx, by = random_box_b_xy(rng, ax, ay)

    def create(name, x, y, yaw_deg=0.0):
        return Node(package='ros_gz_sim', executable='create', output='screen',
                    arguments=['-file', os.path.join(gz_pkg, 'models', name, 'model.sdf'),
                               '-name', name, '-x', f'{x}', '-y', f'{y}', '-z', '0',
                               '-Y', f'{math.radians(yaw_deg)}'])

    return [
        LogInfo(msg=f'Box A at ({ax}, {ay}, 0);  Box B at ({bx:.3f}, {by:.3f}, 0) '
                    f'yaw {byaw:.1f} deg (r = {math.hypot(bx, by):.3f} m)'),
        create('box_a', ax, ay),
        create('box_b', bx, by, byaw),
    ]


def generate_launch_description():
    desc_pkg = get_package_share_directory('gp7_description')
    gz_pkg = get_package_share_directory('gp7_gazebo')
    controllers_file = os.path.join(gz_pkg, 'config', 'controllers.yaml')

    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    rviz = LaunchConfiguration('rviz')

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(desc_pkg, 'urdf', 'gp7.urdf.xacro'),
                 ' gazebo:=true controllers_file:=', controllers_file]),
        value_type=str)

    # Gazebo resolves package://gp7_description/... and package://robotiq_description/...
    # (the 2F-85 meshes) by searching these paths.
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(desc_pkg))
    gz_resource_path_robotiq = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(get_package_share_directory('robotiq_description')))

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            # -r: start running, -s: server only (no GUI window)
            'gz_args': PythonExpression([
                "'-r -v 3 ' + ('-s ' if '", headless, "' == 'true' else '') + '", world, "'"]),
            'on_exit_shutdown': 'true',
        }.items())

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn_robot = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'gp7',
                   '-x', '0', '-y', '0', '-z', '0'])

    # Start the controllers once the robot (and with it gz_ros2_control) exists.
    spawn_controllers = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_state_broadcaster', 'arm_controller', 'gripper_controller',
                   '--controller-manager', '/controller_manager',
                   '--controller-manager-timeout', '60'])
    start_controllers = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[spawn_controllers]))

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge', output='screen',
        parameters=[{'config_file': os.path.join(gz_pkg, 'config', 'gz_bridge.yaml'),
                     'use_sim_time': True}])
    rviz_node = Node(
        package='rviz2', executable='rviz2', condition=IfCondition(rviz),
        arguments=['-d', os.path.join(desc_pkg, 'rviz', 'display.rviz')],
        parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=os.path.join(gz_pkg, 'worlds', 'gp7_world.sdf')),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('box_a_x', default_value='0.5', description='Box A centre x [m]'),
        DeclareLaunchArgument('box_a_y', default_value='0.5', description='Box A centre y [m]'),
        DeclareLaunchArgument('box_b_x', default_value='0.55', description='Box B centre x [m]'),
        DeclareLaunchArgument('box_b_y', default_value='-0.30', description='Box B centre y [m]'),
        DeclareLaunchArgument('box_b_yaw', default_value='0.0', description='Box B yaw [deg]'),
        DeclareLaunchArgument('box_b_random', default_value='false',
                              description='true: random reachable Box B location (overrides x/y)'),
        DeclareLaunchArgument('box_b_seed', default_value='', description='seed for box_b_random'),
        gz_resource_path,
        gz_resource_path_robotiq,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        start_controllers,
        OpaqueFunction(function=spawn_boxes),
        bridge,
        rviz_node,
    ])
