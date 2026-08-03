from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    qcar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('qcar2_nodes'),
                'launch',
                'qcar2_cartographer_launch.py'
            )
        )
    )

    path_mpc = Node(
        package='qcar_science_night_pkg',
        executable='path_mpc',
        name='path_mpc',
        output='screen'
    )

    lidar_overtake = Node(
        package='qcar_science_night_pkg',
        executable='lidar_overtake',
        name='lidar_overtake',
        output='screen'
    )

    lane_centering = Node(
        package='qcar_science_night_pkg',
        executable='lane_centering_node',
        name='lane_centering',
        output='screen'
    )
    depth_emergency_node= Node(
        package='qcar_science_night_pkg',
        executable='depth_emergency_node',
        name='depth_emergency_node',
        output='screen'
    )
    sound_node = Node(
        package='qcar_science_night_pkg',
        executable='sound_node',
        name='sound_node',
        output='screen'
    )

    return LaunchDescription([
        qcar_launch,
        path_mpc,
        lidar_overtake,
        lane_centering,
        depth_emergency_node,
        sound_node
    ])