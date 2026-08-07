#!/bin/bash
# Bring up the whole QCar stack. Run on the car.
#   ./run_all.sh          everything EXCEPT arming (car cannot move)
#   ./run_all.sh --arm    also starts lidar_overtake -> THE CAR DRIVES
set -u

cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1

MAP=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755
ROUTE=/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy

say() { echo ""; echo "=== $* ==="; }

say "1/6  localization + hardware + LiDAR + camera"
nohup ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=${MAP}.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05 enable_camera:=true > /tmp/t1.log 2>&1 &

printf "     waiting for TF"
for i in $(seq 1 24); do
  if timeout 3 ros2 run tf2_ros tf2_echo map base_link >/dev/null 2>&1; then
    echo " -- up after $((i*3))s"; break
  fi
  printf "."; sleep 3
done
ros2 run tf2_ros tf2_echo map base_link 2>&1 | grep -m1 Translation

say "2/6  V2V receiver (UDP 47100 -> ROS topics)"
nohup ros2 run qcar_science_night_pkg v2v_receiver > /tmp/v2vrx.log 2>&1 &
sleep 3

say "3/6  bird's-eye lane detection"
nohup python3 lane_bev.py --frame-skip 3 > /tmp/lanebev.log 2>&1 &
sleep 3

say "4/6  dashboard + camera views"
nohup python3 v2v_dashboard.py --role qcar2 --peer-host 192.168.0.110 \
  > /tmp/dash.log 2>&1 &
nohup python3 camera_web_view.py --topic /lane_bev_debug --port 8082 \
  > /tmp/web_lane.log 2>&1 &
nohup python3 camera_web_view.py --topic /camera/color_image --port 8080 \
  > /tmp/web_raw.log 2>&1 &
sleep 5

say "5/6  MPC  (car stays still -- this is correct)"
nohup ros2 run qcar_science_night_pkg path_mpc --ros-args \
  -p trajectory_file:=${ROUTE} \
  -p map_file:=${MAP}.yaml \
  -p require_map_validation:=true \
  -p require_amcl_quality:=false \
  -p loop_path:=true -p target_laps:=999 -p path_spacing:=0.05 \
  -p speed_limit_ceiling:=2.0 -p max_speed:=0.6 -p curve_speed:=0.40 \
  -p startup_speed:=0.30 -p maneuver_speed:=0.30 \
  -p max_decel:=1.5 -p brake_lookahead_m:=1.20 \
  -p search_window_forward:=90 \
  -p enable_reference_offsets:=true \
  -p enable_v2v:=true \
  > /tmp/t2.log 2>&1 &

printf "     waiting for start alignment"
for i in $(seq 1 20); do
  if grep -q "alignment accepted" /tmp/t2.log 2>/dev/null; then echo ""; break; fi
  if grep -q "Motion blocked: start pose" /tmp/t2.log 2>/dev/null; then
    echo ""; echo "  !! START POSE REJECTED -- move the car onto the route"
    grep "Motion blocked: start pose" /tmp/t2.log | tail -1
    break
  fi
  printf "."; sleep 3
done
grep -E "validation: PASS|alignment accepted|Motion blocked" /tmp/t2.log | tail -3

if [ "${1:-}" = "--arm" ]; then
  say "6/6  ARMING -- THE CAR WILL NOW DRIVE"
  nohup ros2 run qcar_science_night_pkg lidar_overtake --ros-args \
    -p path_point_count:=777 -p path_spacing:=0.05 \
    -p front_offset_deg:=180.0 \
    -p v2v_fusion_enable:=false \
    > /tmp/t3.log 2>&1 &
  sleep 6
  grep -o "state=[A-Z_]*" /tmp/t3.log | sort | uniq -c
else
  say "6/6  NOT ARMED.  To make it drive:  ./run_all.sh --arm"
fi

say "views"
echo "  http://192.168.0.53:8090   dashboard (both cameras + V2V state)"
echo "  http://192.168.0.53:8082   lane overlay"
echo "  http://192.168.0.53:8080   raw camera"
echo ""
echo "  STOP:  pkill -INT -x lidar_overtake       (SIGINT, not plain kill)"
