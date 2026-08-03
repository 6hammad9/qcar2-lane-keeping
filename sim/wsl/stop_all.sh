#!/bin/bash
# Stop only process groups recorded by start_all.sh; never broad-pkill the host.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
exec 9>"$SIM_STATE_DIR/control.lock"
flock 9

echo "commanding zero velocity"
timeout 2 ros2 topic pub --once /model/qcar2/cmd_vel \
  geometry_msgs/msg/Twist '{}' >/dev/null 2>&1 || true

for name in mpc rosbot lane lidar receiver amcl pose_bridge cmd_bridge sim; do
  if component_running "$name"; then
    echo "stopping $name"
  fi
  stop_component "$name"
done
echo "simulation stopped"

