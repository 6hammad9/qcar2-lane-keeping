#!/bin/bash
# QCar MPC in sim, with V2V enabled, driving the same centerline.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
LOOP_PATH="${QCAR_SIM_LOOP_PATH:-false}"
TARGET_LAPS="${QCAR_SIM_TARGET_LAPS:-1}"

# Tuning notes for THIS path (measured, not guessed):
#
#  max_steer 0.46 — matches the Gazebo Ackermann model's effective central
#    steering ceiling. The canonical path needs at most 0.436 rad, so the
#    optimizer no longer assumes steering authority the actuator cannot give.
#
#  max_speed 0.15 — conservative until the rebuilt route is measured in a
#    complete lap. Curves are capped at 0.10 m/s.
python3 "$SIM_DIR/validate_sim_geometry.py"

exec python3 -m qcar_science_night_pkg.path_mpc_node --ros-args \
  -p use_sim_time:=true \
  -p enable_v2v:=true \
  -p enable_reference_offsets:=true \
  -p max_reference_offset:=0.52 \
  -p overtake_max_curvature:=0.40 \
  -p overtake_mean_curvature:=0.25 \
  -p lane_change_distance_m:=0.90 \
  -p solver_timeout_sec:=0.20 \
  -p trajectory_file:="$SIM_DIR/assets/qcar_half_map_centerline.npy" \
  -p map_file:="$SIM_DIR/assets/track_map.yaml" \
  -p path_spacing:=0.05 \
  -p loop_path:="$LOOP_PATH" \
  -p target_laps:="$TARGET_LAPS" \
  -p max_steer:=0.46 \
  -p max_speed:=0.15 \
  -p curve_speed:=0.10 \
  -p maneuver_speed:=0.10 \
  -p require_map_validation:=true \
  -p tracking_log_file:=/tmp/mpc_tracking_log.csv
