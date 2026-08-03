from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


qcar2_scan_match_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        [get_package_share_directory('qcar2_nodes') + '/launch/qcar2_cartographer_launch.py']
    )
)

# Map server (if you need localization)
map_server_node = Node(
    package='nav2_map_server',
    executable='map_server',
    name='map_server',
    output='screen',
    parameters=[{
        'use_sim_time': False,
        'yaml_filename': '/home/nvidia/track_map.yaml'
    }]
)

# AMCL for localization
amcl_node = Node(
    package='nav2_amcl',
    executable='amcl',
    name='amcl',
    output='screen',
    parameters=[{
        'use_sim_time': False,
        'global_frame_id': 'map',
        'odom_frame_id': 'odom',
        'base_frame_id': 'base_link',
        'scan_topic': '/scan',
        'update_min_d': 0.01,
        'update_min_a': 0.01,
        'resample_interval': 1,
        'initial_pose_x': -0.0037,
        'initial_pose_y': -0.0069,
        'initial_pose_a': 0.2534,
        'transform_tolerance': 1.0,
        'min_particles': 500,
        'max_particles': 2000,
        'laser_model_type': 'likelihood_field',
        'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
        'set_initial_pose': True,
        'always_reset_initial_pose': True,
        'first_map_only': True,
        'alpha1': 0.1,
        'alpha2': 0.1,
        'alpha3': 0.1,
        'alpha4': 0.1,
    }]
)

# Nav2 Lifecycle Manager for localization nodes
lifecycle_manager_localization = Node(
    package='nav2_lifecycle_manager',
    executable='lifecycle_manager',
    name='lifecycle_manager_localization',
    output='screen',
    parameters=[{
        'use_sim_time': False,
        'autostart': True,
        'node_names': ['map_server', 'amcl']
    }]
)

# Nav2 Controller Server with Regulated Pure Pursuit
controller_server_node = Node(
    package='nav2_controller',
    executable='controller_server',
    name='controller_server',
    output='screen',
    parameters=['/home/nvidia/ros2_ws/src/qcar_lane_pkg/config/rpp_controller.yaml']  # Create this file
)

# Nav2 Planner Server (optional, if you need global planning)
planner_server_node = Node(
    package='nav2_planner',
    executable='planner_server',
    name='planner_server',
    output='screen',
    parameters=['/home/nvidia/ros2_ws/config/planner_server.yaml']  # Optional
)

# Nav2 Lifecycle Manager for controller and planner
lifecycle_manager_navigation = Node(
    package='nav2_lifecycle_manager',
    executable='lifecycle_manager',
    name='lifecycle_manager_navigation',
    output='screen',
    parameters=[{
        'use_sim_time': False,
        'autostart': True,
        'node_names': ['controller_server']
    }]
)

# EKF for odometry fusion
ekf_node = Node(
    package='robot_localization',
    executable='ekf_node',
    name='ekf_filter_node',
    output='screen',
    parameters=[
        '/home/nvidia/ros2_ws/src/qcarlane_pkg/config/ekf_config.yaml'
    ]
)

# QCar odometry node
qcar2_odom_node = Node(
    package='qcar_lane_pkg',
    executable='qcar2_odom_node',
    name='qcar2_odom_node',
    output='screen'
)

# Modified pure pursuit node (now only publishes path, no control)
pure_pursuit_path_publisher_node = Node(
    package='qcar_lane_pkg',
    executable='pure_pursuit_path_publisher',  # Modified version (see below)
    name='pure_pursuit_path_publisher',
    output='screen'
)

# Cartographer (if using)
cartographer_node = Node(
    package='cartographer_ros',
    executable='cartographer_node',
    output='screen',
    parameters=[{'use_sim_time': False}],
    arguments=['-configuration_directory', '/home/nvidia/ros2_ws/src/qcar2_nodes/config',
               '-configuration_basename', 'qcar2_2d.lua'],
    remappings=[('/imu', '/qcar2_imu'),
                ('/odom', '/ekf_odom') ]
)

# Nav2 QCar converter (bridges commands)
qcar2_nav2_converter = Node(
    package='qcar2_nodes',
    executable='nav2_qcar2_converter',
    name='nav2_qcar2_converter',
    output='screen'
)

# Lane keeping node (if needed)
lane_keeping_node = Node(
    package='qcar_lane_pkg',
    executable='lane_keeping_node',
    name='lane_keeping_node',
    output='screen'
)

# QCar hardware interface
qcar2_hardware = Node(
    package='qcar2_nodes',
    executable='qcar2_hardware',
    name='qcar2_hardware',
)

# Environment variables for performance
env_vars = [
    SetEnvironmentVariable(
        name='LD_PRELOAD',
        value='/lib/aarch64-linux-gnu/libgomp.so.1'
    ),
    SetEnvironmentVariable(
        name='OMP_NUM_THREADS',
        value='1'
    )
]

def generate_launch_description():
    return LaunchDescription(
        env_vars + [
            qcar2_scan_match_launch,
            #qcar2_nav2_converter,
            
            # Uncomment if using EKF
            # ekf_node,
            # qcar2_odom_node,
            
            # Uncomment if using localization
            # map_server_node,
            # amcl_node,
            # lifecycle_manager_localization,
            
            # Regulated Pure Pursuit controller
            controller_server_node,
            lifecycle_manager_navigation,
            
            # Modified path publisher (replaces your pure pursuit controller)
            pure_pursuit_path_publisher_node,
            
            # Optional: Uncomment if needed
            # cartographer_node,
            # lane_keeping_node,
            # planner_server_node,
        ]
    )