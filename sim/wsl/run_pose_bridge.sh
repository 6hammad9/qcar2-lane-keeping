#!/bin/bash
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
exec ros2 run ros_gz_bridge parameter_bridge \
  '/world/sami_track/set_pose@ros_gz_interfaces/srv/SetEntityPose'
