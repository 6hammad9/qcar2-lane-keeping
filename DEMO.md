# Demo runbook — QCar 2 + ROSbot, terminal by terminal

Everything needed to reproduce the working setup from a cold start. Nothing
here is optional unless it says so.

**Addresses**

| | |
|---|---|
| QCar | `192.168.0.53`, user `nvidia` |
| ROSbot | `192.168.0.110`, user `husarion` |
| QCar workspace | `~/qcar_v2v_ws` |
| ROSbot V2V scripts | `~/v2v` |

**Two ROS distros, deliberately isolated.** The QCar runs Humble, the ROSbot
runs Jazzy. Mixing them on one DDS domain crashes with `std::bad_alloc`, which
is why the QCar sets `ROS_LOCALHOST_ONLY=1` and V2V goes over **raw UDP**
rather than DDS. Do not "fix" this by putting both on one domain.

---

## The fast path

If you just want it running, this is the whole thing:

```bash
ssh nvidia@192.168.0.53          # password: see your notes
cd ~/qcar_v2v_ws
./run_all.sh --arm               # THE CAR DRIVES
```

Then open **http://192.168.0.53:8090**. To stop:

```bash
pkill -INT -x lidar_overtake
```

The sections below are the same thing done by hand, one terminal each, for
when something needs to be inspected or a step has to be skipped.

---

## Before you start

Put the car **on the route, facing along it**, roughly where it normally
starts. It does not need to be exact — it aligned at 0.028 m and 0.5° from a
hand placement — but it does need to be on the recorded line.

If the car has been **carried or slid** since it last drove, restart the whole
stack rather than just the MPC. Cartographer pure-localization fuses wheel
odometry with scan matching, and moving the car by hand feeds it motion that
never happened. The pose you see afterwards may be wrong, and no amount of
repositioning against a wrong pose will converge.

---

## Terminal 1 — localization, hardware, LiDAR, camera

```bash
ssh nvidia@192.168.0.53
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1

ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05 \
  enable_camera:=true
```

`enable_camera:=true` is **required** and defaults to false. Without it there
is no `/camera/color_image`, so both the raw view and the lane overlay stay
blank — this is the single most common reason "the camera isn't publishing".

Wait for the `map → base_link` transform before continuing:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

Cartographer needs **motion** to converge — it scan-matches. On a stationary
car the transform can take ~30–70 s to appear. That is normal, not a fault.

## Terminal 2 — V2V receiver

```bash
source /opt/ros/humble/setup.bash && source ~/qcar_v2v_ws/install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1

ros2 run qcar_science_night_pkg v2v_receiver --ros-args \
  -p command_ip:=192.168.0.110 \
  -p command_port:=47101
```

The executable is `v2v_receiver`, **not** `v2v_receiver_node`. Getting this
wrong fails with `No executable found` and V2V is then silently dead while
everything else looks healthy.

`command_ip` is what lets the QCar hold and release the ROSbot during a pass.
Omit it and you get the log line *"V2V command transmission disabled
(command_ip unset); the ROSbot will not be sequenced during passes"* — the
link still works, but the demo sequence does not.

## Terminal 3 — bird's-eye lane detection

```bash
cd ~/qcar_v2v_ws
python3 lane_bev.py --frame-skip 3
```

**Keep `--frame-skip 3`.** At full rate this node takes ~305% CPU — three of
eight cores — which drives load past 6, starves the control loop, and makes
the machine stop answering SSH while still replying to ping. At skip 3 it
sits near 101%.

## Terminal 4 — dashboard and camera views

```bash
cd ~/qcar_v2v_ws
python3 v2v_dashboard.py --role qcar2 --peer-host 192.168.0.110 &
python3 camera_web_view.py --topic /camera/color_image --port 8080 &
python3 camera_web_view.py --topic /lane_bev_debug     --port 8082 &
```

| URL | What it shows |
|---|---|
| http://192.168.0.53:8090 | dashboard — both cameras plus V2V state |
| http://192.168.0.53:8080 | raw camera, no overlay |
| http://192.168.0.53:8082 | lane overlay (filled blue lane) |
| http://192.168.0.110:8081 | ROSbot camera |

## Terminal 5 — MPC (the car still does not move)

```bash
source /opt/ros/humble/setup.bash && source ~/qcar_v2v_ws/install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1

ros2 run qcar_science_night_pkg path_mpc --ros-args \
  -p trajectory_file:=/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy \
  -p map_file:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.yaml \
  -p require_map_validation:=true \
  -p require_amcl_quality:=false \
  -p loop_path:=true -p target_laps:=999 -p path_spacing:=0.05 \
  -p speed_limit_ceiling:=2.0 -p max_speed:=0.6 -p curve_speed:=0.40 \
  -p startup_speed:=0.30 -p maneuver_speed:=0.30 \
  -p max_decel:=1.5 -p brake_lookahead_m:=1.20 \
  -p search_window_forward:=90 \
  -p enable_reference_offsets:=true \
  -p enable_v2v:=true
```

Look for both of these before going on:

```
Path/map validation: PASS
Start alignment accepted at idx=768: position_error=0.028 m, yaw_error=0.5 deg
```

If instead you get `Motion blocked: start pose does not match trajectory`, the
car is not on the route. Limits are 0.35 m and 35°. Reposition and restart
this terminal — see the note about carrying the car above.

## Terminal 6 — arm it (the car drives)

```bash
source /opt/ros/humble/setup.bash && source ~/qcar_v2v_ws/install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1

ros2 run qcar_science_night_pkg lidar_overtake --ros-args \
  -p path_point_count:=777 \
  -p path_spacing:=0.05 \
  -p front_offset_deg:=180.0 \
  -p v2v_fusion_enable:=false
```

`front_offset_deg:=180.0` matches the physical scanner mounting. The Gazebo
model uses 0.0; using the wrong one points every detection box backwards.

`v2v_fusion_enable:=false` keeps passes LiDAR-driven. Turn it on only together
with the MPC's `enable_v2v`, since that is the half that provides the barrier
and speed governor behind a V2V-initiated pass.

---

## ROSbot

The base stack (bringup, LiDAR, camera, EKF, rosbridge) comes up on boot. Only
the two V2V nodes need starting.

### Terminal A — broadcaster

```bash
ssh husarion@192.168.0.110
cd ~/v2v
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

python3 rosbot_v2v_broadcaster.py --ros-args \
  -p map_frame:=odom \
  -p target_ip:=192.168.0.53 \
  -p target_port:=47100 \
  -p vehicle_id:=rosbot3
```

`map_frame:=odom` because **this ROSbot has no `map` frame** — it runs EKF
only, no AMCL and no SLAM, so its TF tree is `odom → base_link`. Leave the
default `map` and the broadcaster finds no transform.

### Terminal B — gate (this is what stops and resumes it)

```bash
cd ~/v2v
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash

python3 rosbot_v2v_gate.py --ros-args -p bind_port:=47101
```

The gate owns the real `/rosbot3/cmd_vel` and republishes from
`/rosbot3/cmd_vel_raw`. For it to actually gate anything, whatever drives the
robot must publish to the `_raw` topic:

```bash
ros2 run rosbot_lane trajectory_follower_node --ros-args \
  -r /rosbot3/cmd_vel:=/rosbot3/cmd_vel_raw
```

A hold is a **bounded lease, never a latch** — each command carries a TTL and
the gate resumes passthrough when it expires, so losing the link mid-pass
costs a second of standing still rather than stranding the robot. With no
command ever received the gate is a pure passthrough.

---

## Checking it works

```bash
# V2V link health — rx climbing, zero errors
ros2 topic echo /v2v/stats --once

# behaviour states as the pass runs
ros2 topic echo /drive_state
```

Healthy link looks like:

```json
{"rx": 488, "parse_errors": 0, "seq_drops": 0, "age_s": 0.194}
```

`"alive": false` with `"gap": -1.473` is **expected right now**: without a
`map` frame on the ROSbot its pose cannot be placed in the QCar's map, so
geometric fusion stays inert. The transport is proven and the hold/release
command path works; only position fusion is missing. Fixing it properly means
`ros-jazzy-nav2-amcl` on the ROSbot plus a shared map.

A complete pass reads:

```
DRIVE → LANE_PROBE → OVERTAKE_LEFT → RETURN_RIGHT → DRIVE
```

---

## Stopping

```bash
pkill -INT -x lidar_overtake     # disarm: car stops, everything else stays up
```

Three things that will waste your time if you do them differently:

- **`ros2 topic pub --once /motion_enable false` does not stop the car.**
  `lidar_overtake` republishes within ~100 ms.
- **Use `-INT`, not plain `pkill`.** The LiDAR's `rplidar_close()` only runs on
  SIGINT; kill it any other way and the motor keeps spinning after the node is
  gone.
- **Use `pkill -x`, not `pkill -f`.** `-f` matches your own SSH command line
  and kills the shell issuing it. Also note Linux truncates process names to
  15 characters, so `pgrep -x cartographer_node` never matches — it is
  `cartographer_no`.

Full shutdown:

```bash
cd ~/qcar_v2v_ws && ./stop_stack.sh
pkill -INT -x cartographer_no
pkill -INT -x qcar2_hardware
pkill -INT -x rplidar_composi
```

---

## Rebuilding after a code change

```bash
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select qcar_science_night_pkg
source install/setup.bash
```

Then restart only `path_mpc` and `lidar_overtake`; Cartographer and the camera
can stay up.

Run the tests on the car, not just on a laptop:

```bash
cd ~/qcar_v2v_ws/src/qcar_science_night_pkg
python3 -m pytest test/ -q \
  --ignore=test/test_pep257.py \
  --ignore=test/test_copyright.py \
  --ignore=test/test_flake8.py
```

The car is on **Python 3.8**. Code that passes on a modern laptop can still
fail there — a doubled `@staticmethod` decorator, for instance, is callable
from 3.10 onward and raises `TypeError` on 3.8, which took out route loading
entirely while every local test passed.

---

## If something is wrong

| Symptom | Cause |
|---|---|
| Camera topic silent | `enable_camera:=true` missing from the Cartographer launch |
| `No executable found` | it is `v2v_receiver`, not `v2v_receiver_node` |
| `Motion blocked: start pose` | car is off the route; limits 0.35 m / 35° |
| SSH dies, ping still replies | load too high — `lane_bev.py` without `--frame-skip` |
| No ping at all, no ARP entry | the car is powered off or off Wi-Fi, not a software fault |
| Car stops for nothing | a wall inside `front_wide_backstop_m`; check whether `nc=0` in the log |
| Stops mid-pass, `max_curv` huge | passing lane folding — the route needs re-recording, not a bigger parameter |
| LiDAR spins after shutdown | killed without `-INT` |

### Reading the `lidar_overtake` log line

Every cycle prints one line. The fields that matter:

```
context=STRAIGHT  state=DRIVE  obs=False
front=1.15  fc=76      <- WIDE box: distance and hit count
narrow=-1.00  nc=0     <- the corridor the car actually sweeps; -1.00 means nothing seen
front_limit=1.45       <- corridor reach
allow_raw / allow_final / enough_dist   <- the three gates a pass needs
```

`front` populated with `nc=0` is scenery beside the path, not an obstacle in
it. That combination is what the wide-box backstop exists to filter, and what
`front_wide_backstop_m=0.65` was tuned against a wall measured at 0.70 m.

A pass needs `allow_raw`, `allow_final` **and** `enough_dist` all true at the
same moment. `enough_dist=False` while parked means the car has closed inside
`overtake_start_min_distance_m` and can no longer commit — `probe_min_gap_m`
is derived as that distance plus 0.05 m specifically so `LANE_PROBE` stops
creeping while a pass is still possible.
