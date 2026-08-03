# QCar steering/localization safety update

This update addresses the observed "turns only one way" behavior without
blindly reversing the steering sign. The recorded route contains both positive
and negative steering commands, and the historical tracking log shows that
commanded steering and measured yaw rate normally have the same sign. The
default `steering_scale` therefore remains `1.0`.

## Confirmed causes and safeguards

- The committed `recorded_path_amcl_final_long.npy` does **not** match
  `track_map_new.yaml`: 9.7% of points are outside the map, 5.1% are occupied,
  and 23.3% violate 0.10 m clearance. Its loop seam also requires about 1.37 rad
  steering while the QCar limit is 0.58 rad. MPC now rejects this combination.
- The previous launch initialized AMCL with a stale pose/yaw. The selected
  trajectory starts near -151 degrees, so initializing it near 0 degrees can
  make MPC saturate continuously in one direction. Initial pose can now be
  published directly from the selected waypoint.
- Overtaking/lane/camera behavior nodes are disabled in the baseline launch.
  This prevents a perception state from applying a fixed left offset while
  steering/localization is being checked.
- The converter and hardware driver now have independent command watchdogs.
  A stale command becomes zero after 0.25/0.30 seconds.
- More than one `/cmd_vel_nav` publisher causes a fail-stop.
- Motor values are decoded by `motor_names`, not array position.
- EKF wheel-speed conversion now agrees with the hardware driver, and time
  jumps no longer produce a large integration step.

## Preserve the trajectory already on the physical car

The physical QCar workspace previously contained a locally modified
`recorded_path_amcl_final_long.npy`. Do not replace it with the older committed
file. On the QCar, back it up before copying or building anything:

```bash
cd ~/ros2_ws
backup_dir="$HOME/qcar-backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"
cp -a recorded_path_amcl_final_long.npy "$backup_dir/"
cp -a track_map_new.yaml track_map_new.pgm "$backup_dir/" 2>/dev/null || true
sha256sum recorded_path_amcl_final_long.npy track_map_new.yaml
git status --short
```

Copy source packages only after reviewing that status. Do not copy root-level
NPY/map files from the laptop over the car.

## Build on the QCar (ROS 2 Humble)

Every QCar terminal must use domain 42; domain 0 previously collided with the
laptop's ROS 2 Jazzy participants and caused `std::bad_alloc` crashes.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
unset ROS_LOCALHOST_ONLY ROS_STATIC_PEERS ROS_AUTOMATIC_DISCOVERY_RANGE

colcon build --symlink-install \
  --packages-select qcar2_nodes qcar_science_night_pkg
source install/setup.bash
```

## 1. Validate the exact map/trajectory pair

```bash
ros2 run qcar_science_night_pkg validate_path_map \
  --trajectory /home/nvidia/ros2_ws/recorded_path_amcl_final_long.npy \
  --map /home/nvidia/ros2_ws/track_map_new.yaml
```

Proceed only if the last line is `PASS`. Do not use
`allow_unsafe_path:=true` with the wheels on the floor.

## 2. Wheels-raised steering check

Launch only hardware, LiDAR, odometry, map, and AMCL:

```bash
ros2 launch qcar_science_night_pkg science_night_slam.launch.py \
  map_file:=/home/nvidia/ros2_ws/track_map_new.yaml \
  trajectory_file:=/home/nvidia/ros2_ws/recorded_path_amcl_final_long.npy \
  use_trajectory_initial_pose:=true \
  trajectory_initial_pose_index:=0
```

The physical car must actually be placed at waypoint 0 before using that pose.
With all four wheels raised, run these one at a time in another sourced/domain-42
terminal:

```bash
timeout 1s ros2 topic pub -r 20 /cmd_vel_nav geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.15}}"

timeout 1s ros2 topic pub -r 20 /cmd_vel_nav geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: -0.15}}"

timeout 1s ros2 topic pub -r 20 /cmd_vel_nav geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

Positive and negative commands must visibly turn in opposite directions, and
zero must return near straight. Only if both signs are physically reversed,
restart the launch with `steering_scale:=-1.0`. Correct a small center bias with
the launch argument `steer_bias:=...`; keep `steering_offset:=0.0` so the board
bias is not applied twice.

## 3. Verify localization before motion

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /amcl
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo map base_link
ros2 topic echo /amcl_pose --once
ros2 topic info /cmd_vel_nav --verbose
```

Both lifecycle nodes must be `active`; both transforms must exist. Verify in
RViz that `/scan` overlays the map walls. Do not change the LiDAR yaw/range
direction merely from code inspection: retain it if the real scan aligns, and
calibrate it separately if the scan is mirrored.

## 4. Start waypoint MPC at low speed

The baseline launch deliberately does not start MPC. In a second terminal:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42

ros2 run qcar_science_night_pkg path_mpc --ros-args \
  -p trajectory_file:=/home/nvidia/ros2_ws/recorded_path_amcl_final_long.npy \
  -p map_file:=/home/nvidia/ros2_ws/track_map_new.yaml \
  -p max_speed:=0.10 \
  -p curve_speed:=0.08 \
  -p startup_speed:=0.08 \
  -p loop_path:=false \
  -p target_laps:=1 \
  -p enable_reference_offsets:=false \
  -p enable_v2v:=false
```

MPC starts disabled. Confirm `ros2 topic info /cmd_vel_nav --verbose` reports
exactly one publisher, then enable it from a third terminal:

```bash
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: true}"
```

It will still refuse motion if path/map validation or start alignment fails.

## Stop commands

Normal stop:

```bash
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: false}"
```

Forced stop (leave this publishing while stopping MPC with Ctrl+C):

```bash
ros2 topic pub -r 20 /cmd_vel_nav geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

This intentionally creates a second publisher if MPC is still alive; the new
converter detects the conflict and holds zero. If either command process dies,
the hardware watchdog also returns speed and steering to zero.

## If the validator rejects the car's current path

Do not force the old route. Localize against `track_map_new`, drive the intended
route once manually, and record a new candidate:

```bash
ros2 run qcar_science_night_pkg trajectory_recorder --ros-args \
  -p output_file:=/home/nvidia/ros2_ws/recorded_path_track_map_new.npy \
  -p output_spacing:=0.03
```

Stop the recorder with Ctrl+C. It preserves a `_raw.npy`, creates a uniformly
spaced MPC NPY and CSV, and reminds you to validate the candidate before use.
