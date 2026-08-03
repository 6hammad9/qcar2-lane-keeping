#!/bin/bash
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
exec python3 "$SIM_DIR/cmd_bridge.py" --ros-args \
  -p use_sim_time:=true -p command_timeout_sec:=0.5
