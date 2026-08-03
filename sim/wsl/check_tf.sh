#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/hammad/rosbot_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1
echo "=== TF frames being published ==="
timeout 6 ros2 topic echo /tf --once 2>/dev/null | grep -E 'frame_id|child_frame_id'
echo
echo "=== can we get map -> base_link? ==="
timeout 5 ros2 run tf2_ros tf2_echo map base_link 2>&1 | head -8
