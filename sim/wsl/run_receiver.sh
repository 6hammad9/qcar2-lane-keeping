#!/bin/bash
# V2V receiver in WSL against the sim centerline (no build needed).
source /opt/ros/jazzy/setup.bash
export ROS_LOCALHOST_ONLY=1
export PYTHONPATH="/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/src/qcar_science_night_pkg:$PYTHONPATH"
exec python3 -m qcar_science_night_pkg.v2v_receiver_node --ros-args \
  -p trajectory_file:=/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/assets/qcar_half_map_centerline.npy \
  -p path_spacing:=0.05 \
  -p log_file:=/tmp/v2v_rx_log.csv
