"""Run the pick & place on the running Gazebo simulation.

    ros2 launch gp7_control pick_place.launch.py
    ros2 launch gp7_control pick_place.launch.py dry_run:=true
    ros2 launch gp7_control pick_place.launch.py profile:=trapezoidal log_dir:=runs/trap

Start the simulation first:  ros2 launch gp7_gazebo gazebo.launch.py [rviz:=true]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'profile': ('', 'scurve | trapezoidal (empty: use config/pick_place.yaml)'),
        'dry_run': ('false', 'plan and publish the preview only'),
        'log_dir': ('', 'directory for CSV logs of the planned / measured motion'),
        'box_a_x': ('0.5', 'Box A centre x [m] (must match the simulation)'),
        'box_a_y': ('0.5', 'Box A centre y [m] (must match the simulation)'),
    }
    decls = [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in args.items()]
    node = Node(
        package='gp7_control', executable='pick_place_node', output='screen',
        parameters=[{
            'use_sim_time': True,
            'profile': LaunchConfiguration('profile'),
            'dry_run': ParameterValue(LaunchConfiguration('dry_run'), value_type=bool),
            'log_dir': LaunchConfiguration('log_dir'),
            'box_a_x': ParameterValue(LaunchConfiguration('box_a_x'), value_type=float),
            'box_a_y': ParameterValue(LaunchConfiguration('box_a_y'), value_type=float),
        }])
    return LaunchDescription(decls + [node])
