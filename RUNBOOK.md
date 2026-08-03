# QCar Runbook — physical car, autonomous lap

Everything needed to drive the QCar around a mapped track, and to add a new
route later. Written after the first successful physical run on 2026-08-02.

---

## 0. What is actually driving the car

Worth understanding before you change anything, because it determines what
matters when something goes wrong:

```
Cartographer  --(map -> odom)-->  TF  -->  path_mpc  -->  /cmd_vel_nav  -->  car
                                            ^
                                            |  /drive_state, /avoidance_offset
                                       lidar_overtake
```

- **`path_mpc`** follows a recorded `.npy` route using the vehicle's pose on
  the map. This is what keeps the car in lane. It is purely geometric.
- **`lidar_overtake`** is the safety/behaviour layer. It publishes
  `/motion_enable`, so **the car does not move until this node runs.** It also
  stops the car for obstacles.
- **The camera / lane-centering node is NOT lane keeping.** It is an optional
  ±4 cm correction on straights (`max_lane_offset = 0.04`) and is off by
  default. The first successful lap ran with it disabled entirely.

Three terminals, always in this order. The MPC needs `/drive_state` from
`lidar_overtake`, and `lidar_overtake` needs `/scan` from the hardware.

---

## 1. Every-run sequence

One-time setup (build, map, route) is in sections 3 and 4. Day-to-day it is
only this.

### Environment — every terminal

```bash
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
```

> Use `tmux new -s car` first. SSH drops kill the MPC mid-lap otherwise.
> `Ctrl+b c` new window, `Ctrl+b 0/1/2` switch, `tmux attach -t car` to rejoin.

### Terminal 1 — localization + hardware + LiDAR

```bash
ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05
```

Wait, then confirm in Terminal 2:

```bash
ros2 topic hz /scan                       # ~10 Hz
ros2 run tf2_ros tf2_echo map base_link   # must resolve
```

If `map -> base_link` says *"not part of the same tree"*, Cartographer has not
relocalized. Put the car where mapping started and push it a metre by hand —
static scans give it nothing to match against.

### Terminal 2 - the controller

**Known-good, 2026-08-03. These settings ran continuous laps.**

```bash
ros2 run qcar_science_night_pkg path_mpc --ros-args   -p trajectory_file:=/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy   -p map_file:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.yaml   -p require_map_validation:=true   -p require_amcl_quality:=false   -p loop_path:=true   -p target_laps:=999   -p path_spacing:=0.05   -p speed_limit_ceiling:=2.0   -p max_speed:=1.2   -p curve_speed:=0.70   -p startup_speed:=0.40   -p maneuver_speed:=0.30   -p max_decel:=1.5   -p brake_lookahead_m:=1.20   -p search_window_forward:=30   -p enable_reference_offsets:=false   -p enable_v2v:=false
```

Wait for `Path/map validation: PASS`, `QCar MPC ready`, and
`Start alignment accepted`. The car stays still - that is correct.

### Terminal 3 - arm it

**Clear the track. Hand on the E-stop.** These are the ORIGINAL narrow
detection boxes. Widening them is what made the car stop at the loop seam -
see section 3a.

```bash
PTS=$(python3 -c "import numpy as np; print(len(np.load('/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy')))")

ros2 run qcar_science_night_pkg lidar_overtake --ros-args   -p path_point_count:=$PTS   -p path_spacing:=0.05   -p front_offset_deg:=180.0   -p v2v_fusion_enable:=false   -p lidar_max_range_m:=2.0   -p front_box_max_m:=0.90   -p emergency_box_max_m:=0.70   -p side_box_max_m:=0.70   -p lane_width_m:=0.43   -p emergency_half_width_m:=0.12   -p front_narrow_half_width_m:=0.16   -p min_front_narrow_points:=2   -p front_stop_straight_m:=0.90   -p emergency_stop_straight_m:=0.70   -p front_stop_curve_m:=0.45   -p emergency_stop_curve_m:=0.65
```

`front_narrow_half_width_m` is the corridor that detects obstacles **in a
curve**, where the wide front box is not consulted at all. It follows
`/path_curvature` from `path_mpc`. Keep it above the car's half-width and
below the lane half-width (0.215 m at `lane_width_m:=0.43`); the node refuses
to start outside that range. See HANDOFF §3.1.

Sanity check in a bend, before trusting it:
```bash
ros2 topic echo /path_curvature      # must be non-zero, and change sign between bends
```
If it is missing or stuck at 0.0, the corridor is straight and curve
detection is back to its old blind behaviour — check `path_mpc` was
redeployed too.

State goes `STARTUP_WAIT` -> `DRIVE` and **the car moves**.

### Optional - camera, lane centering, browser view

None of these are needed to drive. **The camera is already started by the
Terminal 1 launch** - do NOT run `rgbd` separately; two processes on one
RealSense gives `-13 An operating system function returned an unrecognized
error`.

```bash
ros2 run qcar_science_night_pkg lane_centering_node --ros-args   -p config_file:=$(ros2 pkg prefix qcar_science_night_pkg)/share/qcar_science_night_pkg/config/lane_params.yaml

python3 ~/qcar_v2v_ws/camera_web_view.py --topic /camera/color_image --port 8080
```

Lane centering only reaches the controller with
`enable_reference_offsets:=true`, and even then it is a +/-4 cm nudge on
straights only.

## 3a. The loop-seam trap

The route was closed by synthesising a 1.07 m bridge between its end and its
start - ground the car never drove. There is a wall about 0.55 m ahead and
0.36 m to the right at that point.

With the **narrow** boxes above the car cannot see it and laps run fine. With
widened boxes (`emergency_half_width_m:=0.22`, `lane_width_m:=0.60`,
`front_box_max_m:=2.00`) it does see it, stops there every lap, and - because
a curve also forbids overtaking and the inside edge blocks the right sector -
the state machine has no exit and sits in `WAIT_FOR_CLEAR`.

So the current settings work by **not seeing** that wall, which also means a
bottle or another robot 0.20 m off centre is not seen either. To get both,
re-record the route so it closes on ground you actually drove; the wide boxes
then become safe.

### Stop

```bash
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: false}"
```

Or `tmux kill-session -t car`. Emergency: physical E-stop.

---

## 2. Why each non-obvious parameter is there

| Parameter | Why |
|---|---|
| `target_laps:=999` | Laps are counted when the waypoint index reaches the **end of the array**, not when you have travelled a full loop from your start point. Starting at index 493 of 526 means 1.65 m counts as "lap 1". 999 = run until stopped. |
| `path_spacing:=0.05` | Must match the route's actual spacing. Wrong value corrupts every gap and lookahead calculation. |
| `path_point_count:=526` | Must equal the route's point count, or progress miscomputes across the loop seam. |
| `curve_speed:=0.15` | **Below ~0.05 m/s the drivetrain does not turn at all.** Most of this tight loop counts as "curved" (threshold κ=0.25, route mean κ=0.63), so the car sits at curve speed nearly everywhere. Setting it to 0.04 produced `target_v=0.04, v=0.00` — commanded but stationary. |
| `max_reference_offset:=0.40` | Offsetting a path of curvature κ by distance d gives κ/(1−κd). This route peaks at κ=1.82, so the 0.65 default gives 1.82×0.65 = 1.18 > 1 — the offset path folds through its own centre of curvature and reports `max_curv` in the tens. Any value below 1/1.82 = 0.55 is well-defined. |
| `require_amcl_quality:=false` | Cartographer publishes no `/amcl_pose`, so that gate could never pass. **This removes the only automatic localization-quality check** — watch `track_err` instead (aborts at 0.75 m). |
| `front_offset_deg:=180.0` | Physical LiDAR mounting. The Gazebo model uses 0.0. If the front sector seems to watch the wrong direction, test it: put a box 0.5 m in front and see whether `front=` drops. |

---

## 3. Tuning a false stop at a tight corner

Symptom: the car stops at a wall in one particular turn.

In a curve the broad front rectangle is not consulted
(`limit_status_by_path_context`); the curvature-following corridor and the
emergency corridor stop the car there. Both bend along `/path_curvature`, so
a wall the route curves away from should now stay outside them — if the car
still stops at one, check `nc` in the log first: **`nc=0` means the corridor
is empty and the stop came from the emergency box**, so lower that value:

```bash
ros2 run qcar_science_night_pkg lidar_overtake --ros-args \
  -p path_point_count:=526 -p path_spacing:=0.05 \
  -p front_offset_deg:=180.0 -p v2v_fusion_enable:=false \
  -p emergency_stop_curve_m:=0.45
```

All six distances are parameters:

| Parameter | Default | Applies |
|---|---|---|
| `front_stop_straight_m` | 0.90 | obstacle lookahead, straights |
| `emergency_stop_straight_m` | 0.70 | hard stop, straights |
| `front_stop_curve_m` | 0.45 | obstacle lookahead, curves |
| `emergency_stop_curve_m` | 0.65 | **hard stop, curves — lower this for a wall-in-corner stop** |
| `overtake_start_min_distance_m` | 0.60 | minimum room to begin a pass |
| `hard_stop_front_distance_m` | 0.40 | absolute stop distance |
| `front_narrow_half_width_m` | 0.16 | width of the curve corridor |
| `min_front_narrow_points` | 2 | returns needed before the corridor counts |

Confirm from the `lidar_overtake` log which one fired:

```
context=CURVE | obs=False | emg=True | front=0.52 | narrow=-1.00 | nc=0 | emg_min=0.52 | emg_limit=0.65
```

Reading it: `nc=0` and `emg=True` is the emergency box — lower
`emergency_stop_curve_m`. `nc>0` and `obs=True` is the corridor, so something
really is on the route; raising `min_front_narrow_points` only masks it.

`emg=True` with `front` just under `emg_limit` is the corner case above. Do not
go below ~0.35 m — that is real stopping distance.

---

## 4. One-time setup

### Build

```bash
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
colcon build --symlink-install --packages-select qcar2_nodes qcar_science_night_pkg
source install/setup.bash
ros2 pkg executables qcar_science_night_pkg | wc -l    # expect 11
```

### Adding a new route

**A · Map the track** (only if the physical layout changed)

```bash
ros2 launch qcar2_nodes qcar2_cartographer_launch.py
# drive the whole track slowly, return near the start
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState \
  "{filename: '/home/nvidia/qcar_v2v_ws/mapping_output/track_NEW.pbstream'}"
ros2 run nav2_map_server map_saver_cli -f /home/nvidia/qcar_v2v_ws/mapping_output/track_NEW
```

Then **stop Cartographer completely** before the next step.

**B · Localize against the saved map**

```bash
ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=/home/nvidia/qcar_v2v_ws/mapping_output/track_NEW.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05
```

**C · Record the route** — one clean lap, one direction, **no doubling back**

```bash
ros2 run qcar_science_night_pkg trajectory_recorder --ros-args \
  -p output_file:=/home/nvidia/qcar_v2v_ws/mapping_output/route_NEW.npy \
  -p output_spacing:=0.05
```

Drive manually, end back at the start, `Ctrl+C`. It prints the seam:

```
Saved N points ... end/start gap=X m, yaw gap=Y deg
```

Want gap ≤ 0.15 m and yaw ≤ 25° for `loop_path:=true`.

> **Do not record while Cartographer is mapping.** It re-optimizes its pose
> graph on every loop closure, which retroactively moves the `map` frame — so
> early waypoints end up in a frame that no longer exists. Symptoms if you do:
> a trace several times the loop length, `yaw/tangent error` near 180°, and
> `required steering` above 1.5 rad. Localization mode does not do this.

**D · Validate — this is the gate**

```bash
ros2 run qcar_science_night_pkg validate_path_map \
  --trajectory /home/nvidia/qcar_v2v_ws/mapping_output/route_NEW.npy \
  --map /home/nvidia/qcar_v2v_ws/mapping_output/track_NEW.yaml \
  --loop-path
```

Must print `PASS`. Reading the failures:

| Failure | Meaning | Fix |
|---|---|---|
| `outside map` | route leaves the map bounds | wrong map/route pair, or re-export a larger grid from the `.pbstream` |
| `occupied cells` | route crosses a wall | re-record |
| `clearance violations` | route passes close to a wall | usually cosmetic on a narrow track — relax with `--clearance 0.05` and pass the same to the MPC as `-p path_clearance_m:=0.05` |
| `spacing not uniform` | not resampled | re-record, or resample with `resample_trajectory` |
| `headings disagree with tangents` | route reverses direction | you recorded a doubling-back trace — re-record one clean lap |
| `steering above 0.58 rad` | physically unturnable | route folds back on itself; re-record |

**Never use `allow_unsafe_path:=true` with wheels on the floor.** It disables
the outside-map and occupied-cell checks — the two that put the car into a wall.

**E · Update section 1** with the new file paths, and `path_point_count` to
match the new route's length:

```bash
python3 -c "
import numpy as np
p=np.load('/home/nvidia/qcar_v2v_ws/mapping_output/route_NEW.npy')
print('point_count =',len(p))"
```

---

## 5. Diagnosing a bad run

Read these three, in this order.

**Car does not move at all** — `lidar_overtake` window:

```
state=STARTUP_WAIT   motion_enabled=false
```

No `/scan`. Terminal 1 is dead or the LiDAR is not publishing.

**Car crawls or stutters** — `path_mpc` window:

```
target_v=0.04, v=0.00      -> below drivetrain deadband; raise curve_speed
track_err rising           -> route/map frame fit is drifting
TF unavailable             -> Cartographer losing lock
max_curv in the tens       -> degenerate offset path; lower max_reference_offset
```

**Car stops mid-lap:**

```
Lap N/N complete. Target laps complete.   -> raise target_laps
Motion blocked: trajectory/map validation -> see section 4D
emg=True in lidar_overtake                -> see section 3
```

**Restart after a completed mission** without relaunching:

```bash
ros2 topic pub --once /mission_restart std_msgs/msg/Bool "{data: true}"
```

---

## 6. Known-good configuration (2026-08-02)

| Item | Value |
|---|---|
| Workspace | `~/qcar_v2v_ws` |
| Map | `mapping_output/qcar_real_20260802-014755.yaml` + `.pbstream` |
| Route | `mapping_output/qcar_half_map_centerline_real.npy` |
| Route stats | 526 points, 26.25 m, spacing 0.05 m, seam 0.05 m / 1.1° |
| Validation | `PASS` — 100% free cells, 0% clearance violations, steering max 0.44 rad |
| Observed tracking | `track_err` ≈ 0.15 m steady |
| ROS | Humble, `ROS_DOMAIN_ID=42`, `ROS_LOCALHOST_ONLY=1` |
| Camera / lane centering | **off** |
| V2V | **off** |

The route was produced by fitting the simulator centerline into the physical
Cartographer frame with a rigid transform: **rotation −84.42°, translation
(−3.615, −1.334) m**. Keep that recorded — if the map is ever regenerated, the
transform must be refitted or the route re-recorded.
