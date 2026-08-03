from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory



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
qcar_launch= IncludeLaunchDescription(
     PythonLaunchDescriptionSource(
        [get_package_share_directory('qcar2_nodes') + '/launch/qcar2_launch.py']
    )
)
lane_keeping_node = Node(
    package='qcar_lane_pkg',
    executable='lane_keeping_node',
    name='lane_keeping_node',
    output='screen'
)
def generate_launch_description():
    return LaunchDescription(
        env_vars + [
            qcar_launch,
            lane_keeping_node
        ]
    )