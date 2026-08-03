#!/bin/bash
# Complete QCar-side V2V chain in sim.
#  - static map->odom  (QCar spawned at the map-frame trajectory start,
#    so this transform IS the spawn pose; world frame == map frame)
#  - v2v_receiver reading the same centerline the MPC uses
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
QCAR_START_IDX="${QCAR_SIM_START_IDX:-0}"
LOOP_PATH="${QCAR_SIM_LOOP_PATH:-false}"

read -r qcar_x qcar_y qcar_yaw < <(
  python3 -c "import numpy as np; a=np.load('$SIM_DIR/assets/qcar_half_map_centerline.npy'); i=int('$QCAR_START_IDX'); assert 0 <= i < len(a); p=a[i]; print(p[0],p[1],p[2])"
)

ros2 run tf2_ros static_transform_publisher \
  --x "$qcar_x" --y "$qcar_y" --z 0 --yaw "$qcar_yaw" --pitch 0 --roll 0 \
  --frame-id map --child-frame-id odom > /tmp/tf_map_odom.log 2>&1 &
echo "map->odom publisher: $!"
sleep 2

# command_ip closes the loop the other way: QCar tells the ROSbot to hold
# while it passes, because the ROSbot's own detour check cannot see a vehicle
# overtaking it from behind. Both robots are local in sim.
exec python3 -m qcar_science_night_pkg.v2v_receiver_node --ros-args \
  -p use_sim_time:=true \
  -p command_ip:=127.0.0.1 \
  -p command_port:=47101 \
  -p loop_path:="$LOOP_PATH" \
  -p trajectory_file:="$SIM_DIR/assets/qcar_half_map_centerline.npy" \
  -p path_spacing:=0.05 \
  -p log_file:=/tmp/v2v_rx_log.csv
