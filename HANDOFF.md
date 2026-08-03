# QCar handoff — state as of 2026-08-03

Written for a fresh session. Everything below is either measured from logs on
the physical car or read from the source; where something is assumed rather
than verified it says so.

---

## 1. What works right now

The car drives continuous autonomous laps on the physical track.

| | |
|---|---|
| Workspace | `~/qcar_v2v_ws` on the QCar (`nvidia@192.168.0.53`) |
| ROS | Humble, `ROS_DOMAIN_ID=42`, `ROS_LOCALHOST_ONLY=1` |
| Map | `mapping_output/qcar_real_20260802-014755.yaml` + `.pbstream` |
| Route | `mapping_output/my_route_loop.npy` — 777 pts, 38.80 m, 0.05 m spacing |
| Validation | `PASS` — 100% free cells, 0.5% clearance, steering max 0.46 rad |
| Localization | **Cartographer pure-localization** from the pbstream. NOT AMCL. |
| Measured tracking | `track_err` **0.02–0.06 m** for whole laps |
| Camera / lane centering | **off** (see §4) |
| V2V | **off** (see §5) |

### Start sequence — three terminals, this order

Every terminal:
```bash
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
```

**T1 — localization + hardware + LiDAR** (also starts the camera node):
```bash
ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05
```
Wait for TF before continuing, or the MPC spams `TF unavailable`:
```bash
until timeout 3 ros2 run tf2_ros tf2_echo map base_link >/dev/null 2>&1; do sleep 2; done
```

**T2 — MPC.** Car stays still; this is correct.
```bash
ros2 run qcar_science_night_pkg path_mpc --ros-args \
  -p trajectory_file:=/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy \
  -p map_file:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.yaml \
  -p require_map_validation:=true \
  -p require_amcl_quality:=false \
  -p loop_path:=true \
  -p target_laps:=999 \
  -p path_spacing:=0.05 \
  -p speed_limit_ceiling:=2.0 \
  -p max_speed:=1.2 \
  -p curve_speed:=0.70 \
  -p startup_speed:=0.40 \
  -p maneuver_speed:=0.30 \
  -p max_decel:=1.5 \
  -p brake_lookahead_m:=1.20 \
  -p search_window_forward:=30 \
  -p enable_reference_offsets:=false \
  -p enable_v2v:=false
```
Wait for `Path/map validation: PASS`, `QCar MPC ready`, `Start alignment accepted`.

**T3 — arm it. This is what makes the car move** (`lidar_overtake` publishes
`/motion_enable`). Clear the track first.
```bash
PTS=$(python3 -c "import numpy as np; print(len(np.load('/home/nvidia/qcar_v2v_ws/mapping_output/my_route_loop.npy')))")

ros2 run qcar_science_night_pkg lidar_overtake --ros-args \
  -p path_point_count:=$PTS \
  -p path_spacing:=0.05 \
  -p front_offset_deg:=180.0 \
  -p v2v_fusion_enable:=false \
  -p lidar_max_range_m:=2.0 \
  -p front_box_max_m:=0.90 \
  -p emergency_box_max_m:=0.70 \
  -p side_box_max_m:=0.70 \
  -p lane_width_m:=0.43 \
  -p emergency_half_width_m:=0.12 \
  -p front_stop_straight_m:=0.90 \
  -p emergency_stop_straight_m:=0.70 \
  -p front_stop_curve_m:=0.45 \
  -p emergency_stop_curve_m:=0.65
```

Stop: `ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: false}"`

---

## 2. How the system actually fits together

```
T1: cartographer --(map->odom)--> TF --> path_mpc --> /cmd_vel_nav --> car
                       ^                    ^
              EKF (odom->base_link)         | /drive_state /avoidance_offset /motion_enable
                                       lidar_overtake  (T3)
```

Facts that repeatedly caused confusion:

- **`path_mpc` is the lane keeping.** It tracks the recorded route using the
  map pose. Purely geometric. The camera plays no part.
- **The MPC starts disabled** (`self.motion_enabled = False`). It will not
  move until `lidar_overtake` publishes `/motion_enable`.
- **The MPC also requires `lidar_overtake` to stay alive** —
  `behavior_data_fresh()` needs `/drive_state` and `/avoidance_offset` under
  0.7 s old.
- **When the safety layer stops the car, the MPC logs NOTHING**
  (`path_mpc_node.py:1377-1390` returns `should_stop` silently). If the car
  halts with no message in the MPC window, the reason is in the
  `lidar_overtake` window or in `/drive_state`.
- Entry points were renamed: it is **`path_mpc`**, not `mpc_controller`, and
  **`trajectory_recorder`**, not `qcar2_trajectory_recorder`. The root
  `README.md` is stale on this.

---

## 3. Open problems

### 3.1 LiDAR does not detect the ROSbot or a bottle  — ROOT-CAUSED AND FIXED IN CODE, NOT YET RUN ON THE CAR

A person standing in front reliably stops the car. **A bottle or the ROSbot
does not.** The cause was neither of the two hypotheses previously listed
here. It was a third one, in `overtake_safety.limit_status_for_path_context`:

```python
curve_wall_only = context == "CURVE" and not status.emergency
obstacle_ahead = status.obstacle_ahead and ... and not curve_wall_only
```

**Whenever the route was too curved to overtake, `obstacle_ahead` was
discarded outright**, leaving the ±0.12 m emergency corridor as the *only*
detector. Per §4 this route is above the curvature threshold **for most of
its length**, so that was the normal driving case, not an edge case. A
person is wide enough to always clip the centreline; a ROSbot standing a
little off centre is not. Hence: person stops the car, ROSbot does not.

This also explains why widening `emergency_half_width_m` to 0.22 "worked" —
it was widening the only box still doing anything.

**The fix.** A curve now consults a corridor that follows the route's own
curvature instead of a straight rectangle:

- `path_mpc` publishes **`/path_curvature`** (`Float32`, signed mean
  curvature over the MPC horizon; positive = left).
- `LidarSectorAnalyzer` bends both the new narrow front corridor and the
  emergency corridor along `y = κx²/2`.
- In a curve, `obstacle_ahead` comes from that corridor
  (`front_narrow_half_width_m`, default 0.16 m) rather than being zeroed.

This keeps the property the blanket suppression was protecting — the road
edge the route bends *away* from stays outside the corridor — while no
longer also discarding whatever is genuinely in the car's path. If
`/path_curvature` is stale the corridor straightens, which is exactly the
old behaviour.

Second bug found in the same function: the emergency stop was gated on
`status.front_min`, which belongs to a **different, wider and longer box**
than the emergency corridor. It now uses the emergency corridor's own range.

Tests: `test_lidar_curve_corridor.py`, 10 cases. Suite is **130 passing**.

**Still to verify on the car (nothing below is done):**

1. **The scan height.** From the URDF the answer is already known:
   `body_lidar_joint` is `xyz="-0.01227 -0.00045 0.16152"` in all three of
   `qcar.urdf.xacro`, `qcar_ros2.urdf.xacro`, `qcar_ros2_original.urdf.xacro`
   (the last is the one `qcar2_launch.py` actually loads). `base_link` sits
   at the ground plane — the wheel axles are at z=0.03338, which is the wheel
   radius. So **the scan plane is ~0.16 m above the floor.**

   A ROSbot at ~0.22 m tall is therefore **not** under the beam, which is why
   height was not the ROSbot's problem. **A bottle shorter than 16 cm is
   genuinely invisible and no parameter can fix it.** Confirm the real mount
   matches the URDF:
   ```bash
   ros2 run tf2_ros tf2_echo base_link lidar
   ```
2. **Deploy and re-test** the walk-in check. The log now carries the corridor
   directly:
   ```
   front=1.42 | narrow=0.61 | nc=14 | emg_min=-1.00 | kappa=+1.20 | fc=87 | obs=True
   ```
   `nc` is the corridor point count and `kappa` the curvature it is bent by.
   If `obs=False` while `nc` is high, the limit is `front_stop_curve_m`
   (0.45 m — short; consider 0.60). If `nc=0` while the object is plainly in
   the path, the corridor is aimed wrong: check `kappa` is non-zero and the
   right sign in a bend.

### 3.2 The loop seam has a wall

The route was closed by **synthesising a 1.07 m straight bridge** between the
recorded end and start — ground the car never drove. Measured at that point:
`front=0.55 m`, `right=0.36 m`. There is a wall there.

The map validator passed it because the occupancy grid says those cells are
free; free cells are not the same as drivable with the LiDAR boxes.

Consequence: with the narrow boxes the car cannot see the wall and laps run
fine. With wide boxes it sees it, stops there every lap, and — because a
curve also forbids overtaking and the inside edge blocks the right sector —
`WAIT_FOR_CLEAR` has no exit and it sits there forever.

**So the current config works by not seeing that wall.** That is the same
reason it cannot see a bottle. The two problems are one problem.

**Status after the §3.1 fix.** They are one problem, and the corridor is the
common answer to both: the seam wall is *lateral* to the route, so it falls
outside a corridor that follows the route, while an object in the path does
not. That is what makes the two separable at last. The regression test
`test_wall_in_a_curve_with_an_empty_corridor_is_still_not_an_obstacle` pins
the seam case specifically.

This is not a substitute for re-recording. The bridge is still 1.07 m of
ground the car never drove, and `front=0.55, right=0.36` there is genuinely
tight. What the fix buys is that the seam no longer forces the detection
boxes to be too narrow to see anything else.

**Proper fix, still worth doing:** re-record the route driving all the way
back to the start so it closes on real ground. Procedure in `RUNBOOK.md` §4.

### 3.3 Cartographer localization jumps

Two events observed:
- `position_error=3.777 m, yaw_error=-96.6°`, constant while stationary
- `position_error=1.42 m, yaw_error=0.1°` blocking start alignment

`require_amcl_quality:=false` is **required** (Cartographer publishes no
`/amcl_pose`, so that gate can never pass) but it removes the only automatic
localization-quality check. `track_err` is the only backstop and it fires at
0.75 m — after the car is already that far wrong.

**Unverified and important:** whether the Cartographer config actually loaded
is in pure-localization mode. An earlier log showed it building trajectory
`(1, …)` and inserting submaps, which means it was still mapping — every loop
closure then re-optimises `map->odom` under the controller, which is exactly
what a jump like this looks like.
```bash
ros2 pkg prefix qcar2_nodes
grep -nE "include|pure_localization|max_submaps|return" \
  $(ros2 pkg prefix qcar2_nodes)/share/qcar2_nodes/config/qcar2_2d_localization.lua
```
Wanted:
```lua
TRAJECTORY_BUILDER.pure_localization_trimmer = { max_submaps_to_keep = 3 }
```
Note the launch resolved `qcar2_nodes` from **`~/ros2_ws`**, not
`~/qcar_v2v_ws` — check which file is really being loaded.

### 3.4 Camera dead

`ros2 node list` shows no camera node; `/camera/color_image` appears in
`ros2 topic list` only because a subscriber exists. The `rgbd` node started
and then died with:
```
rgb stream: -13 -> An operating system function returned an unrecognized error.
```
`-13` is the RealSense refusing to open. Confirmed cause once: **two `rgbd`
processes** (the T1 launch starts one; another was started by hand). Do not
run `ros2 run qcar2_nodes rgbd` separately.

If it recurs with a single instance: `lsusb | grep -i intel`, then replug —
the D435i needs USB 3, and a stale handle from a killed process can poison
the device until power-cycled.

---

## 4. Does camera perception keep the car in lane?

**No. Not at all, currently — and only marginally even when working.**

Three independent reasons:

1. **The camera node is dead** (§3.4).
2. **`enable_reference_offsets:=false`** in the working config. The MPC
   returns the reference at `path_mpc_node.py:1392` **before** the
   lane-centering block runs, so the offset is computed and then discarded.
   Observed: `lane_valid=True, lane_active=True, lane=0.000`.
3. Even fully enabled it is a **±4 cm** correction (`max_lane_offset = 0.04`)
   applied only where `mean_curv < 0.45`. This route is above that threshold
   for most of its length.

The lane node itself **does work** — verified on the car:
```
valid=True, mode=CL+RL, cl=21.6, rl=467.1, offset=0.031
```
It detects both painted boundaries and produces sensible offsets in the
±0.076 range.

To enable it: start `lane_centering_node`, and set
`-p enable_reference_offsets:=true`. **Caution:** that also re-enables
overtaking, which reintroduces the seam deadlock in §3.2.

Honest assessment: with `track_err` already at 0.02–0.06 m, a ±4 cm nudge on
straights has almost nothing to contribute. It is not the thing keeping the
car in lane and it is not worth prioritising.

---

## 5. Code changes made this session

All in `qcar-izhan/src/qcar_science_night_pkg/`. Deployed to the car by
`scp` + `colcon build --symlink-install --packages-select qcar_science_night_pkg`.

**Deployed and in use:**

| File | Change |
|---|---|
| `lidar_overtake_node.py` | Detection distances and box extents made parameters; **curve-wall suppression bug fixed** |
| `overtake_safety.py` | `limit_status_for_path_context()` extracted as a pure function so it is testable without ROS |
| `path_mpc_node.py` | `speed_limit_ceiling`, `max_decel`, `brake_lookahead_m`, `curve_kappa_threshold`, `search_window_*` made parameters |
| `path_utils.py` | `closest_point()` search window made configurable |
| `utils/camera_web_view.py` | **new** — MJPEG server, browser camera view without X11 |

**Written and tested, NOT yet deployed to the car** (the §3.1 fix):

| File | Change |
|---|---|
| `path_mpc_node.py` | `signed_curvature_preview()` + publishes `/path_curvature` |
| `lidar_sector_analyzer.py` | `corridor_points()` — curvature-following corridors; narrow front corridor and the emergency corridor both bend with the route |
| `overtake_types.py` | `ObstacleStatus` gains `front_narrow_min/count`, `emergency_min/count` (defaulted, so every existing construction still works) |
| `overtake_safety.py` | curve branch uses the corridor instead of discarding `obstacle_ahead`; emergency gated on its own box's range |
| `lidar_overtake_node.py` | subscribes `/path_curvature`; new params `front_narrow_half_width_m`, `min_front_narrow_points`, `path_curvature_timeout_s`; log gains `narrow`/`nc`/`emg_min`/`kappa` |
| `test/test_lidar_curve_corridor.py` | **new** — 10 cases |
| `test/test_path_mpc_node.py` | +4 cases for signed curvature |

Deploy with the same rsync + colcon as above. **`path_mpc` and
`lidar_overtake` must be deployed together** — the corridor is straight
without `/path_curvature`, which silently restores the old blind behaviour.

**Written but NOT deployed or used** (V2V work, `enable_v2v:=false`):
`v2v_conflict.py` (new), `v2v_common.py`, `v2v_receiver_node.py`,
plus `rosbot_v2v_gate.py` and changes to `rosbot_v2v_broadcaster.py` in the
`opta2-sami_ahmed` repo. See `V2V_README.md`.

Tests: **116 passing** —
`PYTHONPATH=src/qcar_science_night_pkg python3 -m pytest src/qcar_science_night_pkg/test sim/test_sim_rosbot_safety.py sim/test_sim_geometry.py -q --ignore=.../test_flake8.py --ignore=.../test_pep257.py --ignore=.../test_copyright.py`
(the three ament linter tests need a ROS environment and are skipped on a laptop).

### Bugs found and fixed, with the evidence

**`brake_lookahead_m` was 0.35 m, hardcoded.** At 2.0 m/s the car needs
2.19 m to shed speed before a corner. It accelerated to full speed on a
straight, hit a corner 0.4 s later, saturated steering at 0.50 rad and
tripped the tracking abort:
```
idx=752  v=0.98  track_err=0.059      <- fine
idx=762  v=2.00  yaw_err=16.0
idx=769  v=2.00  track_err=0.372  delta=0.50
ERROR: Tracking safety stop: position_error=0.477 m, yaw_error=76.0 deg
```
Now a parameter, and the node logs an error at startup if it is too short for
the configured speeds.

**Curve-wall suppression never ran when the wall was close.** The logic was
nested inside `if front_too_far or emergency_too_far:` — true only when the
obstacle is far enough to be discarded anyway. A wall at 0.54 m against a
0.70 m limit left both flags `False`, skipped the block, and kept
`obstacle_ahead=True`. Now runs unconditionally; three regression tests pin
it.

**Lane offset discarded before use** when `enable_reference_offsets=false`
(§4). Not a bug — but it explains `lane_valid=True … lane=0.000`.

**`max_reference_offset` default 0.65 produces a degenerate offset path.**
Offsetting curvature κ by d gives κ/(1−κd); this route peaks at κ=1.82, so
1.82×0.65 = 1.18 > 1 and the offset curve folds through its own centre,
reporting `max_curv` in the tens. Use < 1/κ_max ≈ 0.55.

---

## 6. Things that wasted time — do not repeat

- **Recording a route while Cartographer is mapping.** Loop closure
  retroactively moves the `map` frame, so early waypoints end up in a frame
  that no longer exists. Symptoms: trace several times the loop length,
  `yaw/tangent error` near 180°, `required steering` above 1.5 rad. Record
  only in localization mode, after the map is saved.
- **Using the Cartographer pbstream trajectory as a route.** That is the
  whole mapping session — multiple laps and reversals. Use
  `trajectory_recorder` for a clean single pass.
- **Driving with keyboard teleop using `j`/`l` alone.** Those send angular
  velocity with zero linear, so the car pivots in place; the recorder then
  captures points 3 cm apart with large yaw jumps and the apparent curvature
  explodes (47% of turns over the steering limit). Use `u`/`i`/`o` only.
- **`scp -r src/pkg host:~/ws/src/`** nests a copy inside itself when the
  target exists. Use `rsync -av --delete src/pkg/ host:~/ws/src/pkg/`.
- **`tmux` is not installed and apt is broken** (`nvidia-container-toolkit`
  unmet deps). Do NOT run `apt --fix-broken` — it can pull or remove NVIDIA
  packages. Use `nohup` + `kill -INT` (the recorder only saves on SIGINT).

---

## 7. Suggested order of work

1. **Deploy and validate the §3.1 corridor fix.** It is written and tested
   but has never run on hardware. Deploy `path_mpc` + `lidar_overtake`
   together, confirm `ros2 topic echo /path_curvature` is non-zero in a bend,
   then walk the ROSbot in and watch `nc`/`obs` in the log. Do this on blocks
   with the car jacked up, or at `max_speed:=0.4`, before a full-speed lap.
2. **Confirm the LiDAR mount matches the URDF** (`tf2_echo base_link lidar`,
   expect z≈0.1615). One command. If the real mount is lower than the URDF
   says, short objects are affected and the corridor cannot help.
3. **Verify the Cartographer config really is pure-localization** (§3.3).
   Until this is known, every localization jump is unexplained.
4. **Re-record the route** closing it on real ground (§3.2). Still worth
   doing; the corridor fix makes it no longer urgent.
5. Camera and lane centering last — they contribute nothing to lane keeping
   and the car already tracks at 0.02–0.06 m without them.

Step 1 is the one that changes obstacle detection. Steps 3 and 4 are the ones
that change how reliably the car drives.
