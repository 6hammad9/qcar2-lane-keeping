"""Launch QCar2 hardware and AMCL localization without autonomous motion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    science_share = get_package_share_directory("qcar_science_night_pkg")
    qcar2_share = get_package_share_directory("qcar2_nodes")

    default_map = os.path.expanduser("~/ros2_ws/track_map_new.yaml")
    default_trajectory = os.path.expanduser(
        "~/ros2_ws/recorded_path_amcl_final_long.npy"
    )
    amcl_params_file = os.path.join(science_share, "config", "amcl_config.yaml")

    map_file = LaunchConfiguration("map_file")
    trajectory_file = LaunchConfiguration("trajectory_file")
    enable_camera = LaunchConfiguration("enable_camera")
    steering_scale = LaunchConfiguration("steering_scale")
    steering_offset = LaunchConfiguration("steering_offset")
    steer_bias = LaunchConfiguration("steer_bias")
    enable_lidar_overtake = LaunchConfiguration("enable_lidar_overtake")
    enable_lane_centering = LaunchConfiguration("enable_lane_centering")
    enable_depth_emergency = LaunchConfiguration("enable_depth_emergency")
    enable_sound = LaunchConfiguration("enable_sound")
    use_trajectory_initial_pose = LaunchConfiguration(
        "use_trajectory_initial_pose"
    )
    trajectory_initial_pose_index = LaunchConfiguration(
        "trajectory_initial_pose_index"
    )

    launch_arguments = [
        DeclareLaunchArgument(
            "map_file",
            default_value=default_map,
            description="Absolute path to the occupancy-map YAML used by AMCL.",
        ),
        DeclareLaunchArgument(
            "trajectory_file",
            default_value=default_trajectory,
            description=(
                "Absolute path to the selected Nx3/Nx4 trajectory. The baseline "
                "localization launch uses it only when trajectory initial pose is enabled."
            ),
        ),
        DeclareLaunchArgument(
            "enable_camera",
            default_value="false",
            description="Start the RealSense camera node.",
        ),
        DeclareLaunchArgument(
            "steering_scale",
            default_value="1.0",
            description="QCar steering direction/gain; calibrate wheels raised.",
        ),
        DeclareLaunchArgument(
            "steering_offset",
            default_value="0.0",
            description="Software steering offset; normally keep at zero.",
        ),
        DeclareLaunchArgument(
            "steer_bias",
            default_value="-0.035",
            description="QCar board steering-bias calibration in radians.",
        ),
        DeclareLaunchArgument(
            "enable_lidar_overtake",
            default_value="false",
            description="Start the LiDAR overtaking behavior node.",
        ),
        DeclareLaunchArgument(
            "enable_lane_centering",
            default_value="false",
            description="Start the camera lane-centering behavior node.",
        ),
        DeclareLaunchArgument(
            "enable_depth_emergency",
            default_value="false",
            description="Start the depth-camera emergency-stop node.",
        ),
        DeclareLaunchArgument(
            "enable_sound",
            default_value="false",
            description="Start the optional sound node.",
        ),
        DeclareLaunchArgument(
            "use_trajectory_initial_pose",
            default_value="false",
            description=(
                "Publish /initialpose from the selected trajectory. Enable this "
                "only when the physical car is placed at that waypoint."
            ),
        ),
        DeclareLaunchArgument(
            "trajectory_initial_pose_index",
            default_value="0",
            description="Waypoint index used for the optional trajectory initial pose.",
        ),
    ]

    qcar2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(qcar2_share, "launch", "qcar2_launch.py")
        ),
        launch_arguments={
            "enable_camera": enable_camera,
            "steering_scale": steering_scale,
            "steering_offset": steering_offset,
            "steer_bias": steer_bias,
        }.items(),
    )

    ekf_node = Node(
        package="qcar2_nodes",
        executable="ekf_fusor.py",
        name="ekf_fusor",
        output="screen",
        parameters=[{"tf_pub": True}],
    )

    # The EKF owns odom -> base_link, so its local odometry origin is always zero.
    # This is separate from AMCL's map-frame initial pose.
    init_ekf_pose = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "topic",
                    "pub",
                    "--once",
                    "/init_pose",
                    "geometry_msgs/msg/PoseWithCovarianceStamped",
                    (
                        "{header: {frame_id: 'odom'}, pose: {pose: {position: "
                        "{x: 0.0, y: 0.0, z: 0.0}, orientation: "
                        "{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
                    ),
                ],
                output="screen",
            )
        ],
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "yaml_filename": map_file,
            "use_sim_time": False,
        }],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[amcl_params_file],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    trajectory_initial_pose = Node(
        package="qcar_science_night_pkg",
        executable="trajectory_initial_pose",
        name="trajectory_initial_pose",
        output="screen",
        condition=IfCondition(use_trajectory_initial_pose),
        parameters=[{
            "trajectory_file": trajectory_file,
            "waypoint_index": ParameterValue(
                trajectory_initial_pose_index,
                value_type=int,
            ),
        }],
    )

    lidar_overtake = Node(
        package="qcar_science_night_pkg",
        executable="lidar_overtake",
        name="lidar_overtake",
        output="screen",
        condition=IfCondition(enable_lidar_overtake),
    )

    lane_centering = Node(
        package="qcar_science_night_pkg",
        executable="lane_centering_node",
        name="lane_centering",
        output="screen",
        condition=IfCondition(enable_lane_centering),
    )

    depth_emergency_node = Node(
        package="qcar_science_night_pkg",
        executable="depth_emergency_node",
        name="depth_emergency_node",
        output="screen",
        condition=IfCondition(enable_depth_emergency),
    )

    sound_node = Node(
        package="qcar_science_night_pkg",
        executable="sound_node",
        name="sound_node",
        output="screen",
        condition=IfCondition(enable_sound),
    )

    return LaunchDescription(
        launch_arguments
        + [
            qcar2_launch,
            ekf_node,
            init_ekf_pose,
            map_server,
            amcl,
            lifecycle_manager,
            trajectory_initial_pose,
            lidar_overtake,
            lane_centering,
            depth_emergency_node,
            sound_node,
        ]
    )
