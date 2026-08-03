#!/bin/bash
# Simulated ROSbot: drives the real recorded path AND broadcasts real V2V.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
cd "$SIM_DIR"
ROSBOT_START_IDX="${ROSBOT_SIM_START_IDX:-60}"
# The physical follower's own cruise: PurePursuitConfig(max_speed=0.40,
# curve_speed=0.12) in trajectory_follower_node. It is NOT a slow vehicle --
# it is roughly 2.7x the QCar's 0.15 m/s. The old 0.07/0.03 values existed
# only to force the "QCar passes a slower lead" scenario, which inverted the
# real relationship between the two robots.
ROSBOT_SPEED="${ROSBOT_SIM_SPEED:-0.40}"
ROSBOT_CURVE_SPEED="${ROSBOT_SIM_CURVE_SPEED:-0.12}"
ROSBOT_STOP_DISTANCE="${ROSBOT_SIM_STOP_DISTANCE:-0.40}"
ROSBOT_SLOW_DISTANCE="${ROSBOT_SIM_SLOW_DISTANCE:-0.90}"
ROSBOT_CLEAR_DISTANCE="${ROSBOT_SIM_CLEAR_DISTANCE:-0.50}"
# Speed MUST be below the QCar's max_speed (0.15). The V2V design is
# asymmetric: the QCar reacts to traffic AHEAD of it, so the scenario only
# exists when the QCar is the faster vehicle closing on slower traffic --
# the IDEAM emergency-vehicle framing. With the ROSbot faster it simply
# rear-ends the QCar, which nothing in the QCar's V2V logic can prevent.
# There is deliberately no scripted stop: the ROSbot loops continuously and
# sends a moving predicted horizon. Its independent physical-style proximity
# governor slows progressively inside 0.90 m and reserves a full stop for the
# 0.40 m emergency zone; this is longitudinal sensing, not cooperative MPC.
# --obstacles makes the prop report its OWN blocked state, which is what
# justifies a pass at all: a merely-slower lead whose lane is clear is not a
# reason for QCar to change lanes. Only meaningful when the world was
# generated with the same file (see SIM_README "obstacle scenario").
ROSBOT_OBSTACLES="${ROSBOT_SIM_OBSTACLES:-}"
exec python3 sim_rosbot.py \
  --csv assets/smoothed_trajectory.csv \
  --world sami_track --target 127.0.0.1 \
  --speed "$ROSBOT_SPEED" --start-idx "$ROSBOT_START_IDX" \
  --stop-distance "$ROSBOT_STOP_DISTANCE" \
  --slow-distance "$ROSBOT_SLOW_DISTANCE" \
  --clear-distance "$ROSBOT_CLEAR_DISTANCE" \
  --curve-speed "$ROSBOT_CURVE_SPEED" \
  --obstacles "$ROSBOT_OBSTACLES" \
  --command-port 47101
