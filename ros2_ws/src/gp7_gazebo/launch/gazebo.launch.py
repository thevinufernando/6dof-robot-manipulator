"""Start Gazebo Harmonic with the GP7 bolted to the world origin.

    ros2 launch gp7_gazebo gazebo.launch.py            # Gazebo GUI
    ros2 launch gp7_gazebo gazebo.launch.py rviz:=true # + RViz (joint states from Gazebo)
    ros2 launch gp7_gazebo gazebo.launch.py headless:=true

Pipeline:
  xacro -> robot_description -> robot_state_publisher (topic /robot_description)
  ros_gz_sim create  : reads /robot_description and spawns the model "gp7"
  ros_gz_bridge      : /clock and /joint_states from Gazebo to ROS
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_pkg = get_package_share_directory('gp7_description')
    gz_pkg = get_package_share_directory('gp7_gazebo')

    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    rviz = LaunchConfiguration('rviz')

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(desc_pkg, 'urdf', 'gp7.urdf.xacro'),
                 ' gazebo:=true gz_hold:=true']),
        value_type=str)

    # Gazebo resolves package://gp7_description/... by searching this path.
    gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(desc_pkg))

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

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'gp7',
                   '-x', '0', '-y', '0', '-z', '0'])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
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
        gz_resource_path,
        gz_sim,
        robot_state_publisher,
        spawn,
        bridge,
        rviz_node,
    ])
