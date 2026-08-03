#!/bin/bash
# LiDAR overtake / safety node. Sim publishes /qcar2/lidar/scan; the node
# (like the real car) subscribes to /scan, so remap.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
PATH_POINTS="$(python3 -c "import numpy as np; print(len(np.load('$SIM_DIR/assets/qcar_half_map_centerline.npy')))" )"
exec python3 -m qcar_science_night_pkg.lidar_overtake_node --ros-args \
  -r /scan:=/qcar2/lidar/scan \
  -p use_sim_time:=true \
  -p front_offset_deg:=0.0 \
  -p sensor_timeout_sec:=1.0 \
  -p v2v_fusion_enable:=true \
  -p v2v_detect_range:=2.5 \
  -p v2v_return_clearance_m:=0.80 \
  -p overtake_offset_m:=0.52 \
  -p min_return_progress_m:=0.90 \
  -p path_spacing:=0.05 \
  -p path_point_count:="$PATH_POINTS"
