#!/bin/bash
# Launch the local world directly, then spawn and bridge the QCar2 model from
# the existing read-only QCar simulation workspace.
set -e
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros

HEADLESS=false
if [ "${1:-}" = "--headless" ]; then
  HEADLESS=true
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--headless]" >&2
  exit 2
fi

WORLD_FILE="$SIM_DIR/worlds/sami_track.sdf"
URDF="$QCAR_SIM_WS/install/qcar2_description/share/qcar2_description/urdf/QCar2.urdf"
BRIDGE_CONFIG="$QCAR_SIM_WS/install/qcar2_bringup/share/qcar2_bringup/config/bridge.yaml"
QCAR_START_IDX="${QCAR_SIM_START_IDX:-0}"
ROSBOT_START_IDX="${ROSBOT_SIM_START_IDX:-60}"
for required in "$WORLD_FILE" "$URDF" "$BRIDGE_CONFIG"; do
  [ -f "$required" ] || { echo "missing required file: $required" >&2; exit 1; }
done

export GZ_SIM_RESOURCE_PATH="$SIM_DIR/worlds:$QCAR_SIM_WS/install/qcar2_description/share:$QCAR_SIM_WS/install/qcar2_worlds/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/opt/gz_sim_vendor/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
if [ "$HEADLESS" = false ]; then
  export DISPLAY="${DISPLAY:-:0}"
  # A stale /run/user/1000 value is common after SSH/systemd sessions and is
  # not writable from WSLg. Use its actual runtime directory unless explicitly
  # overridden for this simulator.
  export XDG_RUNTIME_DIR="${QCAR_SIM_XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
fi

children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

gz_args="-r -v 3 $WORLD_FILE"
[ "$HEADLESS" = true ] && gz_args="-r -s -v 3 $WORLD_FILE"
ros2 launch ros_gz_sim gz_sim.launch.py "gz_args:=$gz_args" &
gz_pid=$!
children+=("$gz_pid")

# Bridge /clock, commands, odometry, TF, camera and LiDAR from the installed
# QCar bridge configuration.
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p "config_file:=$BRIDGE_CONFIG" &
children+=("$!")

ros2 run robot_state_publisher robot_state_publisher "$URDF" --ros-args \
  -p use_sim_time:=true &
children+=("$!")

for _ in $(seq 1 60); do
  gz service -l 2>/dev/null | grep -q '/world/sami_track/create' && break
  kill -0 "$gz_pid" 2>/dev/null || {
    echo "Gazebo exited before the world became ready" >&2
    exit 1
  }
  sleep 0.5
done
gz service -l 2>/dev/null | grep -q '/world/sami_track/create' || {
  echo "timed out waiting for /world/sami_track/create" >&2
  exit 1
}

# The SDF's proxy has a neutral authoring pose at waypoint 0. Move it to the
# deterministic scenario start before spawning QCar, avoiding
# an initial model overlap / collision impulse.
read -r rosbot_x rosbot_y rosbot_qz rosbot_qw < <(
  python3 -c "import csv,math; rows=list(csv.DictReader(open('$SIM_DIR/assets/smoothed_trajectory.csv'))); i=int('$ROSBOT_START_IDX'); assert 0 <= i < len(rows); r=rows[i]; y=float(r.get('theta',r.get('yaw'))); print(r['x'],r['y'],math.sin(y/2),math.cos(y/2))"
)
gz service -s /world/sami_track/set_pose \
  --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 \
  --req "name: \"sim_rosbot\", position: {x: $rosbot_x, y: $rosbot_y, z: 0.11}, orientation: {z: $rosbot_qz, w: $rosbot_qw}" \
  >/dev/null

read -r qcar_x qcar_y qcar_yaw < <(
  python3 -c "import numpy as np; a=np.load('$SIM_DIR/assets/qcar_half_map_centerline.npy'); i=int('$QCAR_START_IDX'); assert 0 <= i < len(a); p=a[i]; print(p[0],p[1],p[2])"
)

ros2 run ros_gz_sim create -world sami_track -name qcar2 \
  -allow_renaming=false -file "$URDF" \
  -x "$qcar_x" -y "$qcar_y" -z 0.07 -Y "$qcar_yaw"

echo "QCar2 spawned at canonical waypoint $QCAR_START_IDX in: $WORLD_FILE"
wait "$gz_pid"
