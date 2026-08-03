#!/bin/bash
# Deterministic, supervised QCar + ROSbot V2V simulation bringup.
set -e
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros

HEADLESS=false
NO_MPC=false
NO_LANE=false
V2V_DEMO=false
for arg in "$@"; do
  case "$arg" in
    --headless) HEADLESS=true ;;
    --no-mpc) NO_MPC=true ;;
    --no-lane) NO_LANE=true ;;
    --v2v-demo) V2V_DEMO=true ;;
    *) echo "usage: $0 [--headless] [--no-mpc] [--no-lane] [--v2v-demo]" >&2; exit 2 ;;
  esac
done

# Fast, deterministic moving-traffic acceptance scenario. Waypoints 416--468
# are the longest measured straight on the canonical route (2.65 m). QCar
# starts 17 points (about 0.85 m centre-to-centre) behind a continuously
# moving, slower ROSbot.  That is about 0.46 m of physical bumper clearance.
# QCar starts at the first curvature-approved waypoint and remains held by its
# following governor until the lead has moved beyond the 0.60 m LiDAR passing
# threshold.  This removes startup drift without inventing a scheduled stop.
# All consumers use the same start-index variables, so map->odom remains exact.
if [ "$V2V_DEMO" = true ]; then
  export QCAR_SIM_START_IDX="${QCAR_SIM_START_IDX:-419}"
  export ROSBOT_SIM_START_IDX="${ROSBOT_SIM_START_IDX:-436}"
  # No speed override: the ROSbot runs its real cruise (0.40 m/s straight,
  # 0.12 in curves, from trajectory_follower_node's PurePursuitConfig). It is
  # the FASTER robot -- roughly 2.7x QCar's 0.15 m/s -- so on a shared loop the
  # recurring encounter is it catching QCar from behind. The old 0.03 m/s
  # override inverted that and made the demo show a scenario the hardware
  # does not actually produce.
  export QCAR_SIM_LOOP_PATH="${QCAR_SIM_LOOP_PATH:-true}"
  export QCAR_SIM_TARGET_LAPS="${QCAR_SIM_TARGET_LAPS:-2}"
  echo "V2V demo: QCar idx $QCAR_SIM_START_IDX, ROSbot idx $ROSBOT_SIM_START_IDX at its own cruise (${ROSBOT_SIM_SPEED:-0.40} m/s)"
fi

exec 9>"$SIM_STATE_DIR/control.lock"
flock -n 9 || { echo "another start/stop operation is active" >&2; exit 1; }

# A second invocation is rejected instead of silently creating duplicate
# broadcasters/controllers.  Use stop_all.sh first for an intentional restart.
for name in sim pose_bridge receiver amcl lidar rosbot lane cmd_bridge mpc; do
  if component_running "$name"; then
    echo "ERROR: simulation already running ('$name'). Run $WSL_DIR/stop_all.sh first." >&2
    exit 1
  fi
done

failed_cleanup() {
  echo "bringup failed; stopping components started by this run" >&2
  for name in mpc cmd_bridge lane rosbot lidar amcl receiver pose_bridge sim; do
    stop_component "$name"
  done
}
trap failed_cleanup ERR
trap 'failed_cleanup; exit 130' INT TERM

echo "[1/6] Gazebo world and QCar2"
if [ "$HEADLESS" = true ]; then
  start_component sim bash "$WSL_DIR/run_sim.sh" --headless
else
  start_component sim bash "$WSL_DIR/run_sim.sh"
fi
# `gz model --list` can lag the GUI scene broadcaster even after the blocking
# create service has succeeded. The supervised simulator logs its success only
# after that service returns, which is the authoritative spawn-ready signal.
for _ in $(seq 1 100); do
  grep -q '^QCar2 spawned at canonical waypoint' \
    "$SIM_LOG_DIR/sim.log" 2>/dev/null && break
  component_running sim || { echo "Gazebo component exited" >&2; false; }
  sleep 0.5
done
grep -q '^QCar2 spawned at canonical waypoint' \
  "$SIM_LOG_DIR/sim.log" 2>/dev/null || {
  echo "timed out waiting for QCar2 spawn" >&2
  false
}

echo "[2/6] ROS-Gazebo pose service and V2V receiver"
start_component pose_bridge bash "$WSL_DIR/run_pose_bridge.sh"
start_component receiver bash "$WSL_DIR/run_v2v_chain.sh"

echo "[3/6] localization and LiDAR safety"
start_component amcl bash "$WSL_DIR/run_amcl.sh"
start_component lidar bash "$WSL_DIR/run_lidar.sh"

if [ "$NO_LANE" = false ]; then
  echo "[4/6] camera lane-centering correction"
  start_component lane bash "$WSL_DIR/run_lane.sh"
else
  echo "[4/6] camera lane-centering skipped"
fi

if [ "$NO_MPC" = false ]; then
  echo "[5/6] safe command bridge and MPC"
  start_component cmd_bridge bash "$WSL_DIR/run_cmd_bridge.sh"
  start_component mpc bash "$WSL_DIR/run_mpc.sh"
else
  echo "[5/6] MPC skipped"
fi

# Start the moving lead last. This prevents it drifting out of the validated
# straight while localization and the QCar MPC are still initializing.
echo "[6/6] continuously moving ROSbot V2V traffic"
start_component rosbot bash "$WSL_DIR/run_rosbot.sh"

# `-r` should start the world running, but some GUI sessions restore a paused
# state. Apply the requested running state explicitly after all consumers are
# ready, so a healthy demo cannot appear motionless.
gz service -s /world/sami_track/control \
  --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
  --timeout 5000 --req 'pause: false' >/dev/null || true

if [ "$HEADLESS" = false ]; then
  bash "$WSL_DIR/focus_qcar_view.sh" || true
fi

trap - ERR INT TERM
echo "Simulation started. Logs: $SIM_LOG_DIR"
echo "Status: bash $WSL_DIR/status_all.sh"
echo "V2V:    bash $WSL_DIR/watch_demo_live.sh"
echo "Stop:   bash $WSL_DIR/stop_all.sh"
