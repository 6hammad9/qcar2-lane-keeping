# Project Structure

This document maps every file in the `rosbot_lane` package and what it does.
There are two independent pipelines living in this one package:

1. **Map/trajectory following** — record a path with SLAM, smooth it, then
   drive it back autonomously with AMCL localization + pure pursuit.
   (`tf_relay.py`, `slam_trajectory_recorder.py`, `smooth.py`,
   `trajectory_follower_node.py`, `core/pure_pursuit.py`, `core/trajectory.py`)
2. **Camera-based lane keeping** — follow painted white lane lines using the
   OAK RGB camera, no map/SLAM involved.
   (`lane_keeping_node.py`, `straight_lane_node.py`, and most of the other
   `core/*.py` modules)

See [guide.md](guide.md) for the step-by-step run instructions for pipeline 1,
and [README.md](README.md) for a quick overview + version info.

```
rosbot3/
├── README.md
├── guide.md
├── structure.md            (this file)
├── memory.md
├── commands.txt
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── rosbot_lane
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── config/
│   ├── amcl_params.yaml
│   ├── lane_params.yaml
│   ├── slam_params.yaml
│   ├── slam_trajectory_recorder.py
│   ├── smooth.py
│   ├── slam_trajectory.csv
│   ├── smoothed_trajectory.csv
│   ├── smoothed_trajectory.csv.bak
│   ├── track_map.yaml
│   ├── track_map.pgm
│   ├── track_map.data
│   └── track_map.posegraph
└── rosbot_lane/
    ├── __init__.py
    ├── tf_relay.py
    ├── trajectory_follower_node.py
    ├── lane_keeping_node.py
    ├── straight_lane_node.py
    └── core/
        ├── __init__.py
        ├── config.py
        ├── control.py
        ├── debug.py
        ├── detection.py
        ├── geometry.py
        ├── logging.py
        ├── obstacle_tracker.py
        ├── pure_pursuit.py
        ├── segmentation.py
        ├── trajectory.py
        ├── obstacle.wav
        ├── pickup.wav
        └── delivery.wav
```

---

## Root files

| File | Purpose |
|---|---|
| `README.md` | Quick overview, environment/version info, and the condensed run commands for recording and following a path. |
| `guide.md` | The full step-by-step walkthrough: recording, smoothing, running, plus a troubleshooting section built from real issues hit on this robot (missing `scan_filtered` topic, stuck localization, stray smoothed-file path). |
| `structure.md` | This file — what every file in the repo does. |
| `memory.md` | A dated debugging log (network/DDS discovery issues between the laptop and the robot). Historical notes, not living documentation. |
| `commands.txt` | Scratch notes / copy-paste command history used while developing the recording and localization workflow. Superseded by `guide.md` and `README.md` — kept as raw scratch reference. |
| `package.xml` | ROS 2 ament_python package manifest. Declares the package name (`rosbot_lane`), its `rclpy`/`sensor_msgs`/`std_msgs`/`geometry_msgs`/`cv_bridge` dependencies, and test dependencies (`ament_copyright`, `ament_flake8`, `ament_pep257`, `pytest`). |
| `setup.py` | Python packaging entry point for colcon. Registers the two `ros2 run`-able console scripts: `lane_keeping_node` and `straight_lane_node`. (`trajectory_follower_node.py` and `tf_relay.py` are run directly with `python3`, not registered here — see the Note below.) |
| `setup.cfg` | Boilerplate ament_python install-location config (`install_scripts=$base/lib/rosbot_lane`). |
| `resource/rosbot_lane` | Empty marker file required by `ament_index` so ROS 2 recognizes this as an installed package. |

> **Note on `setup.py`:** only `lane_keeping_node` and `straight_lane_node`
> are registered as console scripts. `trajectory_follower_node.py`,
> `tf_relay.py`, and everything under `config/*.py` are invoked as plain
> `python3 path/to/file.py` (as shown throughout `README.md`/`guide.md`),
> not `ros2 run rosbot_lane ...`. If you want `ros2 run` for those too,
> they'd need entries added to `setup.py`'s `console_scripts`.

## `test/`

Standard ament_python linter boilerplate, unrelated to robot behavior — run
via `colcon test`:

| File | Purpose |
|---|---|
| `test_copyright.py` | Checks source files carry a copyright header. |
| `test_flake8.py` | Runs flake8 style/lint checks. |
| `test_pep257.py` | Checks docstring style (PEP 257). |

## `config/`

Mix of ROS parameter files, recorded/generated data, and two standalone
Python scripts (not part of the ROS package build — run directly).

| File | Purpose |
|---|---|
| `amcl_params.yaml` | Parameters for `nav2_amcl` (localization) + the `map_server` + `lifecycle_manager` used in the "run on recorded path" step. Notably `scan_topic: /rosbot3/scan` (see README's version/compat note — older firmware published `/rosbot3/scan_filtered` instead). |
| `lane_params.yaml` | Parameters for `lane_keeping_node.py` — HSV segmentation thresholds, PD control gains, topic names (image/cmd_vel/scan), and a `trajectory.file` pointer (currently unused by that node). |
| `slam_params.yaml` | Parameters for `slam_toolbox`'s `online_async_launch.py`, used while recording a new map. |
| `slam_trajectory_recorder.py` | Standalone node. Polls `map → base_link` TF at 20 Hz while you teleop the robot during recording, and appends a point to `slam_trajectory.csv` every time the robot has moved ≥2 cm. Run with `python3 config/slam_trajectory_recorder.py`. |
| `smooth.py` | Standalone script (not a ROS node — no `rclpy`). Reads `config/slam_trajectory.csv`, **despikes** single-frame outlier points (a raw point whose neighbors are much closer to *each other* directly than to it — usually a one-frame SLAM scan-matching glitch, not real motion), splits what's left into segments at genuine direction-reversal points (dot product of consecutive direction vectors < -0.5), **merges any resulting segment shorter than 0.15m back into its neighbor** (a second safety net against noise producing spurious short "reverse" segments), fits a B-spline per real segment (`scipy.interpolate.splprep`), resamples to uniform 5 cm spacing, and writes `config/smoothed_trajectory.csv`. **Must be run from the repo root** (uses relative paths). Requires `pandas`, `numpy`, `scipy` — see README's venv note. |
| `slam_trajectory.csv` | Raw recorded path: one `x,y,theta` row per recorded point, in the `map` frame. Input to `smooth.py`. Overwritten each time you record a new path. |
| `smoothed_trajectory.csv` | Output of `smooth.py` — the uniform-spacing, spline-smoothed path. **This is the file `trajectory_follower_node.py` actually drives.** |
| `smoothed_trajectory.csv.bak` | Backup of a previous `smoothed_trajectory.csv`, kept from an earlier session as a rollback point. Safe to delete once you've confirmed the current smoothed file is good. |
| `track_map.yaml` | Map metadata for `nav2_map_server` — points at `track_map.pgm`, plus resolution/origin/occupancy thresholds. Passed as the `map:=` arg when launching AMCL. |
| `track_map.pgm` | The occupancy grid image itself (grayscale bitmap), produced by `map_saver_cli`. |
| `track_map.data` / `track_map.posegraph` | slam_toolbox's serialized pose-graph (from `SerializePoseGraph`). Lets you reload/continue the exact same SLAM mapping session later, independent of the `.pgm`/`.yaml` pair used for pure localization. |

## `rosbot_lane/` (the ROS package / Python module)

| File | Purpose |
|---|---|
| `__init__.py` | Marks `rosbot_lane` as a Python package. Empty. |
| `tf_relay.py` | Small always-needed node. Subscribes to the robot's namespaced `/rosbot3/tf` and `/rosbot3/tf_static`, and republishes them on the global `/tf` / `/tf_static` topics that AMCL, slam_toolbox, and the TF-based nodes in this repo expect. Must be running before SLAM, AMCL, the trajectory recorder, or the trajectory follower. |
| `trajectory_follower_node.py` | The main "drive the recorded path" node (pipeline 1). A segment-based pure-pursuit controller with a state machine (`WAITING_FOR_LOCALIZATION → ROTATE_TO_START → DRIVE_TO_START → ALIGN_AT_START → FOLLOWING_SEGMENT → ... → COMPLETE`, looping). Reads `config/smoothed_trajectory.csv`, gets pose from `map → base_link` TF, publishes `TwistStamped` to `/rosbot3/cmd_vel`. Also handles: forward obstacle stop/slow (via `/rosbot3/scan`), a scripted left-side overtake maneuver, hardcoded waypoint pause points, and a background-thread keyboard pause toggle. Depends on `core/pure_pursuit.py` and `core/trajectory.py` only. |
| `lane_keeping_node.py` | Camera-based lane-following node (pipeline 2). Subscribes to `/rosbot3/oak/rgb/image_raw`, segments white lane markings, fits lane lines, computes a steering error, and drives with a PD controller. Config-driven from `config/lane_params.yaml`. Built from the `core/` modules (segmentation, detection, geometry, control, logging, debug) — this is the "clean" composed version. Registered as a `ros2 run rosbot_lane lane_keeping_node` console script. |
| `straight_lane_node.py` | An alternate, self-contained, single-file version of camera-based lane keeping (v2, Hough-transform line fitting). Does **not** import from `core/` — all detection/control logic is inlined in this one ~650-line file. Registered as `ros2 run rosbot_lane straight_lane_node`. Appears to be an earlier/parallel implementation to `lane_keeping_node.py` rather than something it depends on. |
| `.curved_lane_node.py.kate-swp` | Leftover Kate editor swap file from editing a (now apparently deleted or never-committed) `curved_lane_node.py`. Not part of the build; safe to delete. |

### `rosbot_lane/core/` — shared library modules

Only used by `lane_keeping_node.py` and `trajectory_follower_node.py` (each
uses a different subset — see the dependency note below).

| File | Purpose |
|---|---|
| `config.py` | `load_config()` — reads `lane_params.yaml` into a typed `LaneConfig` dataclass (image size, HSV thresholds, PD gains, topic names, etc). Used only by `lane_keeping_node.py`. |
| `segmentation.py` | `segment_white()` — HSV-threshold + morphological cleanup to extract white lane-marking pixels from a BGR camera frame. Used only by `lane_keeping_node.py`. |
| `detection.py` | Hough-transform-based lane line fitting: sliding-window search, `detect_rl`/`detect_cl` (right line / center line), corridor masking, line/polynomial evaluation helpers. Used only by `lane_keeping_node.py`. |
| `geometry.py` | `compute_lane_error()` / `compute_lane_error_rl_only()` — converts detected lane line(s) into a lateral steering error in pixels. Used only by `lane_keeping_node.py`. |
| `control.py` | `LaneController` — simple PD controller turning a steering error into an angular velocity command. Used only by `lane_keeping_node.py`. |
| `logging.py` | `LaneLogger` — writes a CSV log of per-frame lane detection state (`lane_log.csv`) for offline debugging/tuning. Used only by `lane_keeping_node.py`. |
| `debug.py` | `draw_debug_frame()` and helpers — draws the fitted lines, detection points, and HUD text onto the camera frame for the `show_display` debug window. Used only by `lane_keeping_node.py`. |
| `pure_pursuit.py` | `PurePursuitController` / `PurePursuitConfig` — the geometric path-tracking controller (rotate-first for forward motion, direct heading correction for reverse). Used only by `trajectory_follower_node.py`. |
| `trajectory.py` | `Waypoint` / `Segment` / `Trajectory` — loads `smoothed_trajectory.csv`, auto-detects direction-flip points to split it into drivable segments, and provides closest-waypoint/lookahead lookups. Used only by `trajectory_follower_node.py`. |
| `obstacle_tracker.py` | A more elaborate state-machine-based obstacle tracker (`ObstacleState`: `NO_OBSTACLE → TRACKING → FOLLOWING → OVERTAKING → RETURNING_TO_PATH`). **Currently unused** — not imported by any node. `trajectory_follower_node.py` has its own simpler inline obstacle/overtake logic instead. Looks like an in-progress or superseded design; check with whoever wrote it before relying on it. |
| `obstacle.wav` / `pickup.wav` / `delivery.wav` | Audio cues played (fire-and-forget via `aplay`) by `trajectory_follower_node.py` — obstacle-detected sound, and two sounds alternated at scripted waypoint pause points. |

---

## Quick dependency map

```
trajectory_follower_node.py
 ├─ core/pure_pursuit.py
 └─ core/trajectory.py  →  reads config/smoothed_trajectory.csv

lane_keeping_node.py
 ├─ core/config.py       →  reads config/lane_params.yaml
 ├─ core/segmentation.py
 ├─ core/detection.py
 ├─ core/geometry.py
 ├─ core/control.py
 ├─ core/logging.py
 └─ core/debug.py

straight_lane_node.py     (self-contained, no core/ dependency)

tf_relay.py                (standalone, no dependencies on other repo files)

config/slam_trajectory_recorder.py   → writes config/slam_trajectory.csv
config/smooth.py                     → config/slam_trajectory.csv → config/smoothed_trajectory.csv

core/obstacle_tracker.py   (standalone module, currently unused by any node)
```
