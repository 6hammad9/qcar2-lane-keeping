# ROSbot 3 — Recording, Smoothing, and Path Following Guide

This guide covers the full loop used in this repo: recording a new map + driven
path ("map coordinates"), smoothing that path, and running the robot
autonomously back along it (`trajectory_follower_node.py`, the "lane
following" step referenced below).

All commands assume:
- You're in the repo root (`~/rosbot3`), so every path below is
  workspace-relative — no editing paths per machine.
- ROS 2 is sourced and the robot is discoverable (`ros2 topic list` shows
  `/rosbot3/...` topics). If not, see **Troubleshooting → Robot not
  discoverable** at the bottom.
- The robot is running Husarion firmware 2.0 / rosbot_ros `jazzy` branch —
  this build does **not** publish `/rosbot3/scan_filtered`. Everything in
  this repo has been switched to use raw `/rosbot3/scan` instead (see
  Troubleshooting for why that mattered).

There is a separate, unrelated pair of nodes in this package —
`lane_keeping_node.py` / `straight_lane_node.py` — that do **camera-based**
white-line following instead of map/trajectory following. This guide is
about the map + trajectory pipeline; see the note at the very end if you
actually wanted the camera-based nodes instead.

---

## Part 1 — Recording a New Path

You need 4 terminals, all opened at the repo root.

**Terminal 1 — TF Relay**

Bridges the robot's namespaced `/rosbot3/tf` topics to the global `/tf` that
SLAM/AMCL expect. Must be running before anything else that touches TF.

```bash
python3 rosbot_lane/tf_relay.py
```

**Terminal 2 — SLAM Toolbox**

Builds the map live and publishes the `map` frame the recorder needs.

```bash
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=config/slam_params.yaml \
  use_sim_time:=false
```

**Terminal 3 — Trajectory Recorder**

Polls `map → base_link` TF at 20 Hz and appends a point to
`config/slam_trajectory.csv` every time the robot moves ≥2 cm.

```bash
python3 config/slam_trajectory_recorder.py
```

Watch its log — if you see repeated `TF lookup failed` warnings instead of
`Point N: x=... y=...` lines, SLAM hasn't published the `map` frame yet
(give it a second, or check Terminal 2's output for errors).

**Terminal 4 — Drive the Robot**

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/rosbot3/cmd_vel -p stamped:=true
```

Drive the full path you want the robot to later repeat. Go slowly and
smoothly — SLAM's scan-matching degrades with fast, jerky motion, and
`slam_trajectory_recorder.py` records raw poses that `smooth.py` will later
try to clean up, but it isn't magic. If your track has an out-and-back
section (the robot reverses direction), that's fine — the smoother detects
direction flips automatically (see Part 2).

**When done driving, save the map** (new terminal, or reuse Terminal 4 once
teleop is stopped):

```bash
ros2 run nav2_map_server map_saver_cli -f config/track_map
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: 'config/track_map'}"
```

This writes `config/track_map.pgm`, `.yaml`, `.data`, and `.posegraph`. The
`.posegraph`/`.data` pair lets you resume/extend this exact SLAM session
later if needed; the `.pgm`/`.yaml` pair is what AMCL localizes against in
Part 3.

> **Verify it actually wrote both pairs** — don't just trust the command not
> erroring. If you split the `serialize_map` call across multiple lines with
> a trailing `\`, a missing closing quote leaves your shell stuck at a `>`
> continuation prompt *without ever running the command* — no error, no
> output, it just silently never executes. Check
> `ls -la config/track_map.data config/track_map.posegraph` and confirm the
> timestamp matches *now*, not a previous session, before moving on. Keeping
> the whole call on one line (as above) avoids the trap entirely.

You can now Ctrl+C terminals 2–4. Leave Terminal 1 (`tf_relay`) running if
you're moving straight on to Part 2 or 3.

---

## Part 2 — Smoothing the Recorded Path

`config/smooth.py` reads the raw `config/slam_trajectory.csv`, splits it into
segments at direction-reversal points, fits a B-spline per segment, and
resamples to uniform 5 cm spacing. Output goes to
`config/smoothed_trajectory.csv` — this is the exact file
`trajectory_follower_node.py` reads.

```bash
.venv/bin/python3 config/smooth.py
```

`smooth.py` needs `pandas`/`numpy`/`scipy`, which aren't part of the system
Python here (and this machine is externally-managed, so no sudo/plain
`pip install` — see the venv setup note in README.md's Environment section
if `.venv/` doesn't exist yet). It's a standalone script (no `rclpy`), so a
plain venv with no ROS access is enough — just remember the `.venv/bin/`
prefix, not plain `python3`.

> **Do not `source .venv/bin/activate` in a terminal you're also using for
> ROS nodes.** This venv is isolated on purpose (needed to avoid version
> conflicts) and does *not* include the system `dist-packages` that
> `rclpy` itself depends on (e.g. `PyYAML`) — activating it and then
> running `tf_relay.py` / `trajectory_follower_node.py` / any ROS node in
> that same shell will fail with `ModuleNotFoundError: No module named
> 'yaml'` or similar. Use `.venv/bin/python3 config/smooth.py` (no
> activation needed) for this one script, and plain `python3 ...` in a
> clean (non-activated) shell for everything else in Part 1 and Part 3.

Expected output:
```
Smoothed trajectory saved to config/smoothed_trajectory.csv with N uniform waypoints.
```

> **Must be run from the repo root.** `smooth.py` uses the relative paths
> `config/slam_trajectory.csv` → `config/smoothed_trajectory.csv`. Running it
> from inside `config/` (or anywhere else) will fail to find the input file.

If the resulting path still looks jittery in RViz/plotting, the spline
smoothing factor is `s=0.02` in `smooth.py`'s `splprep(...)` call — raise it
for a looser fit, lower it to hug the raw recording more closely.

---

## Part 3 — Running the Robot on the Recorded Path

Again, 4 terminals at the repo root.

**Terminal 1 — TF Relay** (if not already running from Part 1)

```bash
python3 rosbot_lane/tf_relay.py
```

**Terminal 2 — AMCL Localization**

```bash
ros2 launch nav2_bringup localization_launch.py \
  map:=config/track_map.yaml \
  params_file:=config/amcl_params.yaml \
  use_sim_time:=false
```

**Terminal 3 — Activate Lifecycle Nodes + Global Localization**

```bash
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
ros2 lifecycle set /amcl configure
ros2 lifecycle set /amcl activate

ros2 service call /reinitialize_global_localization std_srvs/srv/Empty
```

`reinitialize_global_localization` spreads AMCL's particle filter across the
whole map since we don't set an initial pose. **Nudge the robot slightly
(a short forward/back or a small rotation) after calling this** — AMCL needs
a bit of motion combined with scan matching to disambiguate the pose,
especially on a track with repeated/symmetric geometry. Watch for
convergence with:

```bash
ros2 topic echo /amcl_pose --once
```

(covariance values shrinking = converging). You can also confirm the `map`
frame exists at all with:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

If this hangs on "Invalid frame ID... frame does not exist" for more than a
few seconds, AMCL isn't receiving scans — see Troubleshooting.

**Terminal 4 — Trajectory Follower**

```bash
python3 rosbot_lane/trajectory_follower_node.py
```

This is a pure-pursuit controller (`rosbot_lane/core/pure_pursuit.py`) that
follows `config/smoothed_trajectory.csv` using live `map → base_link` TF for
pose. It starts in `WAITING_FOR_LOCALIZATION` and won't move until that TF
lookup succeeds — so it will just sit still (not error) if AMCL hasn't
localized yet. It also does forward-obstacle stopping and a left-side
overtake maneuver, both driven off `/rosbot3/scan`.

Useful things to know about this node while it's running:
- Position the robot at roughly the same start pose it was recorded from —
  it will rotate-then-drive itself the rest of the way to the exact
  trajectory start point before following begins.
- `_pause_points` near the top of the file (around line 125) are hardcoded
  waypoint coordinates where it'll stop for 20s — edit/comment these out per
  map if you don't want scripted stops.
- It has a keyboard pause toggle (runs a background stdin listener) if you
  need to halt it manually mid-run without killing the process.

---

## Troubleshooting

**Robot sits still, never starts driving, no errors.**
`trajectory_follower_node.py` is stuck in `WAITING_FOR_LOCALIZATION` because
`map → base_link` TF isn't resolving. Check, in order:
1. `ros2 lifecycle get /amcl` and `/map_server` — both should say `active [3]`.
2. `ros2 topic hz /rosbot3/scan` — confirm the raw LIDAR is actually
   publishing (~10 Hz). If not, that's a robot/driver-side problem, not this
   pipeline.
3. `ros2 topic info /rosbot3/scan --verbose` — confirm `amcl` shows up as a
   subscriber. If AMCL's `scan_topic` param doesn't match what's actually
   being published, it'll sit silently with zero scans and never broadcast
   `map`. (This bit us once already this session — firmware 2.0 doesn't
   publish `/rosbot3/scan_filtered`, only raw `/rosbot3/scan`. Everything in
   this repo now points at `/rosbot3/scan`; if you're on an older
   husarion image that *does* publish a filtered topic, you may want to
   switch back.)
4. `ros2 run tf2_ros tf2_echo map base_link` — confirm the `map` frame exists
   at all.

**Robot follows a path that doesn't match what you just drove.**
`trajectory_follower_node.py` always reads `config/smoothed_trajectory.csv`.
If you ran `smooth.py` from somewhere other than the repo root, or from an
older copy of this repo before the output-path fix, double check the
timestamp on `config/smoothed_trajectory.csv` against `config/slam_trajectory.csv`
— the smoothed file should be newer:

```bash
ls -la config/slam_trajectory.csv config/smoothed_trajectory.csv
```

**Robot not discoverable at all (`ros2 topic list` shows nothing but
`/parameter_events` and `/rosout`).**
This is a laptop/network-specific issue seen previously, not a robot
problem — confirm by running the same check from `rosbot-server`, which
reaches the robot fine. If you must run standalone from a laptop again, you
likely need `ROS_STATIC_PEERS` pointed at the robot's IP since plain
multicast discovery may be filtered by the WiFi AP:

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=OFF
export ROS_STATIC_PEERS="192.168.0.110"
```

---

## Note: camera-based lane following (different pipeline)

If what you actually want is vision-based white-line following instead of
map/trajectory following, that's `lane_keeping_node.py` /
`straight_lane_node.py` — these use the OAK camera (`/rosbot3/oak/rgb/image_raw`)
and `config/lane_params.yaml`, and have no dependency on SLAM, AMCL, or any
recorded trajectory:

```bash
ros2 run rosbot_lane lane_keeping_node
# or
ros2 run rosbot_lane straight_lane_node
```

These are unrelated to Parts 1–3 above and don't need a map at all.
