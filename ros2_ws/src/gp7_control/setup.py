from setuptools import find_packages, setup

package_name = 'gp7_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='user@todo.todo',
    description='Move the GP7 to end-effector poses and drive its gripper.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'move_to_pose = gp7_control.move_to_pose:main',
            'gripper = gp7_control.gripper_cli:main',
            'pose_goal_node = gp7_control.pose_goal_node:main',
        ],
    },
)
