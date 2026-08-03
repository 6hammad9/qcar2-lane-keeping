#!/bin/bash
# Camera lane-centering is a bounded correction on top of the map path MPC.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros

# The source package is run directly via PYTHONPATH.  Provide its config to
# ament_index_python without copying/building over the user's ROS workspace.
OVERLAY="$SIM_STATE_DIR/source_overlay"
mkdir -p "$OVERLAY/share/ament_index/resource_index/packages" \
         "$OVERLAY/share/qcar_science_night_pkg"
: > "$OVERLAY/share/ament_index/resource_index/packages/qcar_science_night_pkg"
ln -sfn "$REPO_DIR/src/qcar_science_night_pkg/config" \
  "$OVERLAY/share/qcar_science_night_pkg/config"
export AMENT_PREFIX_PATH="$OVERLAY${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"

exec python3 -m qcar_science_night_pkg.lane_centering_node --ros-args \
  -p use_sim_time:=true \
  -p config_file:="$SIM_DIR/lane_params.yaml" \
  -r /camera/color_image:=/qcar2/front_camera/image
