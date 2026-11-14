"""View the GP7 in RViz with joint sliders (no Gazebo).

    ros2 launch gp7_description display.launch.py

Use the joint_state_publisher_gui sliders to move each joint and watch the
TF frames - handy for checking frame assignment before writing the D-H table.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('gp7_description')
    xacro_file = os.path.join(pkg, 'urdf', 'gp7.urdf.xacro')
    rviz_file = os.path.join(pkg, 'rviz', 'display.rviz')

    gui = LaunchConfiguration('gui')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' gazebo:=false']), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show joint_state_publisher_gui sliders'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             condition=UnlessCondition(gui)),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_file]),
    ])
