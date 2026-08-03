"""Launch the physical QCar2 hardware nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    enable_camera = LaunchConfiguration("enable_camera")
    steering_scale = LaunchConfiguration("steering_scale")
    steering_offset = LaunchConfiguration("steering_offset")
    steer_bias = LaunchConfiguration("steer_bias")
    converter_command_timeout = LaunchConfiguration(
        "converter_command_timeout"
    )
    hardware_command_timeout = LaunchConfiguration(
        "hardware_command_timeout"
    )

    enable_camera_arg = DeclareLaunchArgument(
        "enable_camera",
        default_value="false",
        description=(
            "Start the RealSense RGB-D node. It is disabled by default because "
            "localization only requires the LiDAR."
        ),
    )

    steering_scale_arg = DeclareLaunchArgument(
        "steering_scale",
        default_value="1.0",
        description=(
            "Converter steering direction/gain. Test with wheels raised before "
            "using -1.0."
        ),
    )
    steering_offset_arg = DeclareLaunchArgument(
        "steering_offset",
        default_value="0.0",
        description=(
            "Software steering offset in radians. Keep zero when using the "
            "QCar steer_bias board option."
        ),
    )
    steer_bias_arg = DeclareLaunchArgument(
        "steer_bias",
        default_value="-0.035",
        description="QCar board steering-bias calibration in radians.",
    )
    converter_timeout_arg = DeclareLaunchArgument(
        "converter_command_timeout",
        default_value="0.25",
        description="Seconds before the converter forces a stop.",
    )
    hardware_timeout_arg = DeclareLaunchArgument(
        "hardware_command_timeout",
        default_value="0.30",
        description="Seconds before the hardware driver forces a stop.",
    )

    lidar_node = Node(
        package="qcar2_nodes",
        executable="lidar",
        name="Lidar",
        output="screen",
    )

    qcar2_nav2_converter = Node(
        package="qcar2_nodes",
        executable="nav2_qcar2_converter",
        name="nav2_qcar2_converter",
        output="screen",
        parameters=[{
            "steering_scale": ParameterValue(steering_scale, value_type=float),
            "steering_offset": ParameterValue(steering_offset, value_type=float),
            "command_timeout_sec": ParameterValue(
                converter_command_timeout, value_type=float
            ),
        }],
    )

    realsense_camera_node = Node(
        package="qcar2_nodes",
        executable="rgbd",
        name="RealsenseCamera",
        output="screen",
        condition=IfCondition(enable_camera),
    )

    qcar2_hardware = Node(
        package="qcar2_nodes",
        executable="qcar2_hardware",
        name="qcar2_hardware",
        output="screen",
        parameters=[{
            "steer_bias": ParameterValue(steer_bias, value_type=float),
            "motor_command_timeout_sec": ParameterValue(
                hardware_command_timeout, value_type=float
            ),
        }],
    )

    qcar2_sensor_tf_node = Node(
        package="qcar2_nodes",
        executable="fixed_lidar_frame",
        name="fixed_lidar_frame",
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": ParameterValue(
                Command([
                    FindExecutable(name="xacro"),
                    " ",
                    os.path.join(
                        get_package_share_directory("qcar_lane_pkg"),
                        "urdf",
                        "qcar_ros2_original.urdf.xacro",
                    ),
                ]),
                value_type=str,
            )
        }],
    )

    return LaunchDescription([
        enable_camera_arg,
        steering_scale_arg,
        steering_offset_arg,
        steer_bias_arg,
        converter_timeout_arg,
        hardware_timeout_arg,
        lidar_node,
        qcar2_nav2_converter,
        qcar2_sensor_tf_node,
        realsense_camera_node,
        qcar2_hardware,
        robot_state_publisher_node,
    ])
