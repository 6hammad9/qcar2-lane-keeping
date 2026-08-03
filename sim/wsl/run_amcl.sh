#!/bin/bash
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
exec python3 "$SIM_DIR/sim_amcl_shim.py" --ros-args -p use_sim_time:=true
