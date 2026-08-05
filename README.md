# Autonomous Indoor Navigation using ROS2 for Quanser QCar

> **Autonomous Navigation • Mapping • Localization • Model Predictive Control • Obstacle Avoidance**

---

## ▶ Current state, open problems, handoff: [HANDOFF.md](HANDOFF.md)
## ▶ To actually drive the car: [RUNBOOK.md](RUNBOOK.md)

Exact three-terminal start sequence, the known-good configuration, how to add a
new route, and what every failure message means. Read that, not this file, if
your goal is to make the car move.

Two things below are out of date and will not work as written:

| This README says | Actual |
|---|---|
| `ros2 run qcar_science_night_pkg mpc_controller` | **`path_mpc`** |
| "Launching the MPC starts autonomous motion" | It starts **disabled**. `lidar_overtake` publishes `/motion_enable`, so the car does not move until that node runs. |
| `ros2 run qcar_science_night_pkg qcar2_trajectory_recorder` | **`trajectory_recorder`** |

---

# Quick Start — run the already-built implementation

Use this if the workspace is **already built** and the map and route already
exist on the car. Nothing here builds, maps, or records anything — for that,
see [RUNBOOK.md](RUNBOOK.md) §3–4. This is the known-good configuration that
ran continuous autonomous laps on 2026-08-03, tracking at 0.02–0.06 m.

### Prerequisites

| | |
|---|---|
| Target | `nvidia@192.168.0.53` |
| Workspace | `~/qcar_v2v_ws` (already built with `colcon`) |
| Map | `mapping_output/qcar_real_20260802-014755.pbstream` + `.yaml` |
| Route | `mapping_output/my_route_loop.npy` (777 pts, 38.80 m) |
| Localization | Cartographer pure-localization — **not** AMCL |

Confirm those exist before starting:

```bash
ls ~/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.{pbstream,yaml} \
   ~/qcar_v2v_ws/mapping_output/my_route_loop.npy
```

> **If you have just pulled new code, rebuild first** — the curvature corridor
> needs both nodes at the same version:
> ```bash
> cd ~/qcar_v2v_ws && colcon build --symlink-install \
>   --packages-select qcar_science_night_pkg && source install/setup.bash
> ```

### Step 0 — environment, in every terminal

```bash
cd ~/qcar_v2v_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
```

> Keep sessions alive across SSH drops, or an SSH hiccup kills the MPC
> mid-lap. `tmux` was **not installed** on the car as of 2026-08-03 and `apt`
> is broken there (do **not** run `apt --fix-broken` — it can pull or remove
> NVIDIA packages). If `tmux new -s car` fails, use `nohup … &` instead and
> stop nodes with `kill -INT` rather than `kill`.

### Terminal 1 — localization, hardware, LiDAR

```bash
ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
  state_filename:=/home/nvidia/qcar_v2v_ws/mapping_output/qcar_real_20260802-014755.pbstream \
  configuration_basename:=qcar2_2d_localization.lua \
  resolution:=0.05
```

**This does NOT start the camera.** `enable_camera` defaults to `false`
(`qcar2_launch.py`), and this launch includes that file without overriding it,
so the RGB-D node is never spawned. Add `enable_camera:=true` if you want it.
Localization needs only the LiDAR, so the known-good lap ran without it.

If you do enable it, the node is named **`RealsenseCamera`**, not `rgbd`, and
you must not also run `ros2 run qcar2_nodes rgbd` — two processes on one
RealSense fail to open the device.

Block until TF is up, or the MPC spams `TF unavailable`:

```bash
until timeout 3 ros2 run tf2_ros tf2_echo map base_link >/dev/null 2>&1; do sleep 2; done
ros2 topic hz /scan     # expect ~10 Hz
```

If `map -> base_link` reports *"not part of the same tree"*, Cartographer has
not relocalized. Put the car where mapping started and push it a metre by
hand — static scans give it nothing to match against.

### Terminal 2 — the controller

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

Wait for `Path/map validation: PASS`, `QCar MPC ready`, and
`Start alignment accepted`. **The car stays still — that is correct.**
`require_amcl_quality:=false` is mandatory: Cartographer publishes no
`/amcl_pose`, so that gate can never pass.

### Terminal 3 — arm it (this is what makes the car move)

**Clear the track. Hand on the E-stop.**

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
  -p front_narrow_half_width_m:=0.16 \
  -p min_front_narrow_points:=2 \
  -p front_stop_straight_m:=0.90 \
  -p emergency_stop_straight_m:=0.70 \
  -p front_stop_curve_m:=0.45 \
  -p emergency_stop_curve_m:=0.65
```

State goes `STARTUP_WAIT` → `DRIVE` and **the car moves**.

### Stopping

```bash
ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: false}"
```

### Confirm it is healthy

```bash
ros2 topic echo /path_curvature      # non-zero in a bend, sign flips between bends
```

In the `lidar_overtake` log, a good line looks like:

```
context=CURVE | state=DRIVE | obs=False | front=1.42 | narrow=-1.00 | nc=0 | kappa=+1.20
```

`kappa` stuck at `0.00` means `/path_curvature` is missing, the detection
corridor is straight, and **obstacle detection in curves is back to its old
blind behaviour** — check `path_mpc` is the redeployed build.

### If something goes wrong

| Symptom | Cause |
|---|---|
| Car never moves | Terminal 3 not running — it publishes `/motion_enable` |
| MPC spams `TF unavailable` | Started Terminal 2 before Cartographer relocalized |
| Car stops with **no message** in the MPC window | The safety layer stopped it silently — the reason is in the `lidar_overtake` window or `/drive_state` |
| Car halts mid-lap, MPC still alive | `lidar_overtake` died; the MPC needs `/drive_state` and `/avoidance_offset` under 0.7 s old |
| `Tracking safety stop: position_error=…` | Localization jump, see [HANDOFF.md](HANDOFF.md) §3.3 |

Full failure catalogue and tuning: [RUNBOOK.md](RUNBOOK.md) §2–3.

---

# Overview

This repository contains the complete software stack for autonomous indoor navigation on the **Quanser QCar** using **ROS2 Humble**.

The project was developed for indoor autonomous vehicle demonstrations and combines modern robotics algorithms with a modular ROS2 architecture to achieve reliable localization, smooth trajectory tracking, autonomous obstacle avoidance and emergency stopping.

Unlike conventional waypoint-following systems, this project records a trajectory using localization, processes it into a smooth reference path and follows it using a Model Predictive Controller (MPC). During execution, the vehicle continuously monitors its surroundings using LiDAR and performs autonomous lane changes whenever obstacles are detected.

The software has been designed with modularity in mind, allowing each subsystem to operate independently while communicating through standard ROS2 interfaces.

---

# Features

## Localization

* Cartographer SLAM mapping
* AMCL localization
* Extended Kalman Filter (EKF)
* Continuous pose estimation
* Low covariance localization

---

## Motion Planning

* Trajectory recording
* Automatic trajectory smoothing
* Uniform waypoint resampling
* Curvature computation
* Yaw generation
* Model Predictive Control (MPC)

---

## Perception

* RPLidar A2M12 obstacle detection
* Lane occupancy estimation
* Front obstacle detection
* Autonomous lane selection
* Intel RealSense D435i emergency stop

---

## Autonomous Behaviour

* Autonomous trajectory following
* Autonomous lane changing
* Return to original lane
* Mission completion detection
* Emergency stopping
* German voice notifications

---

# Hardware

| Component    | Description           |
| ------------ | --------------------- |
| Vehicle      | Quanser QCar          |
| LiDAR        | RPLidar A2M12         |
| Depth Camera | Intel RealSense D435i |
| Compute      | NVIDIA Jetson         |
| Sensors      | Wheel Encoders, IMU   |

---

# Software

| Software           | Version          |
| ------------------ | ---------------- |
| Ubuntu             | 22.04            |
| ROS2               | Humble Hawksbill |
| Python             | 3.10             |
| Cartographer       | ROS2             |
| Nav2               | Humble           |
| AMCL               | Nav2             |
| robot_localization | Humble           |
| NumPy              | Latest           |
| SciPy              | Latest           |
| Matplotlib         | Latest           |

---

# Repository Structure

```text
ros2_ws/

├── README.md
├── DEVELOPER_README.md
│
├── src/
│   ├── qcar_science_night_pkg/
│   │
│   ├── launch/
│   ├── config/
│   ├── resource/
│   └── qcar_science_night_pkg/
│
├── utils/
├── build/
├── install/
└── log/
```

---

# Software Architecture

The complete autonomous navigation pipeline is illustrated below.

```text
                      RPLidar
                         │
                         ▼
                Cartographer SLAM
                         │
                 Occupancy Grid Map
                         │
                         ▼
                      AMCL
                         │
                 map → base_link
                         │
                         ▼
              Model Predictive Control
                         │
                         ▼
                      QCar
```

Obstacle avoidance operates independently of localization.

```text
                    LaserScan
                        │
                        ▼
             LiDAR Sector Analyzer
                        │
                        ▼
             Obstacle Classification
                        │
                        ▼
             Lane Selection Logic
                        │
                        ▼
              MPC Reference Update
                        │
                        ▼
                  Steering Commands
```

Emergency stopping is handled by a dedicated safety layer.

```text
               Intel RealSense D435i
                         │
                         ▼
                Depth Emergency Node
                         │
                         ▼
                Emergency Stop Signal
                         │
                         ▼
                   Vehicle Controller
```

---

# Coordinate Frames

The system follows the standard ROS TF hierarchy.

```text
map
 │
 └── odom
      │
      └── base_link
            │
            ├── base_scan
            ├── lidar
            ├── camera_link
            └── wheel frames
```

Frame descriptions

| Frame       | Description               |
| ----------- | ------------------------- |
| map         | Global reference frame    |
| odom        | Continuous odometry frame |
| base_link   | Vehicle body frame        |
| base_scan   | LiDAR frame               |
| camera_link | RealSense frame           |

---

# ROS2 Nodes

The autonomous system is composed of multiple ROS2 nodes.

| Node                  | Purpose                    |
| --------------------- | -------------------------- |
| EKF Fusion            | Sensor fusion and odometry |
| Cartographer          | Mapping                    |
| Map Server            | Map publication            |
| AMCL                  | Localization               |
| Robot State Publisher | TF publication             |
| MPC Controller        | Path following             |
| LiDAR Sector Analyzer | Obstacle detection         |
| Depth Emergency Node  | Emergency stopping         |
| Trajectory Recorder   | Path recording             |
| Path Processor        | Trajectory smoothing       |

Each node is designed to operate independently while communicating through ROS2 topics and services.

---

# Typical Workflow

The recommended workflow for operating the system is:

```text
Create Map
      │
      ▼
Save Map
      │
      ▼
Start EKF
      │
      ▼
Start AMCL
      │
      ▼
Set Initial Pose
      │
      ▼
Verify Localization
      │
      ▼
Launch MPC
      │
      ▼
Autonomous Navigation
```
# Installation

## Clone the Repository

```bash
cd ~/ros2_ws/src
git clone <repository_url>
```

## Build the Workspace

```bash
cd ~/ros2_ws
colcon build 
source install/setup.bash
```

To automatically source the workspace on every terminal:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

# Required Dependencies

The following ROS2 packages are required:

- Cartographer ROS
- Navigation2
- robot_localization
- robot_state_publisher
- RViz2
- tf2
- NumPy
- SciPy
- Matplotlib

---

# Mapping

A new occupancy grid map can be generated using Cartographer.

Start the mapping system:

```bash
ros2 launch qcar_science_night_pkg qcar2_cartographer_original_launch.py
```

Drive the QCar through the complete environment while ensuring that all navigable areas are observed by the LiDAR. For best results:

- Drive slowly and smoothly.
- Avoid sudden steering inputs.
- Visit all corridors and intersections.
- Return close to the starting position before saving the map.

---

## Saving the Map

Save the Cartographer pose graph:

```bash
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState \
"{filename: '/home/nvidia/demo_map.pbstream'}"
```

Export the occupancy grid for localization:

```bash
ros2 run nav2_map_server map_saver_cli \
-f /home/nvidia/demo_map
```

The following files will be generated:

```
demo_map.pbstream
demo_map.pgm
demo_map.yaml
```

---

# Localization

Localization is performed using AMCL together with the Extended Kalman Filter (EKF).

Launch the localization stack:

```bash
ros2 launch qcar_science_night_pkg science_night_slam.launch.py
```

Once the map has loaded:

1. Open RViz.
2. Select **2D Pose Estimate**.
3. Click on the vehicle's current position on the map.
4. Move the vehicle slightly until the localization converges.

Verify localization:

```bash
ros2 topic echo /amcl_pose
```

Recommended covariance values:

| Parameter | Recommended |
|-----------|-------------|
| x variance | < 0.02 |
| y variance | < 0.02 |
| yaw variance | < 0.02 |

You can also verify that the localization transform is stable:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

---

# Trajectory Recording

Reference trajectories are recorded after localization has converged.

Start the trajectory recorder:

```bash
ros2 run qcar_science_night_pkg qcar2_trajectory_recorder
```

Drive the desired route once.

A CSV file containing the recorded trajectory will be generated.

Example:

```
qcar_trajectory.csv
```

---

# Trajectory Processing

The recorded trajectory is converted into a smooth reference path suitable for the MPC controller.

Processing includes:

- Removal of stationary points
- Cubic B-Spline fitting
- Uniform waypoint resampling
- Heading computation
- Curvature computation

Generate the processed trajectory:

```bash
python3 smooth_path.py \
--input qcar_trajectory.csv \
--output recorded_path_amcl_final.npy
```

The resulting trajectory contains:

```
x
y
yaw
curvature
```

This processed trajectory is used directly by the MPC controller.

---

# Autonomous Driving

**Localization only estimates the vehicle pose and does not command the vehicle to move.**

After localization has successfully converged, autonomous driving is started by launching the **Model Predictive Controller (MPC)**.

## Before Starting MPC

Ensure that:

- Hardware drivers are running.
- EKF localization is running.
- AMCL localization has converged.
- The map has been loaded.
- A reference trajectory (`.npy`) is available.
- The transform between `map` and `base_link` is stable.

Verify localization:

```bash
ros2 topic echo /amcl_pose
```

Verify the TF tree:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

Once localization is stable, launch the MPC controller:

```bash
ros2 run qcar_science_night_pkg mpc_controller
```

Launching the MPC controller starts autonomous vehicle motion.

During operation the controller will:

1. Load the processed reference trajectory.
2. Read the vehicle pose from AMCL.
3. Compute steering and velocity commands using Model Predictive Control.
4. Track the reference trajectory.
5. Continuously monitor for obstacles.
6. Perform autonomous lane changes when safe.
7. Return to the original lane after overtaking.
8. Stop immediately if an emergency condition is detected.
9. Stop automatically once the final waypoint has been reached.

> **Important:** The QCar will remain stationary until the MPC controller is started, even if localization is active.

---

# Obstacle Avoidance

Obstacle avoidance is performed using the onboard LiDAR.

The system continuously monitors:

- Front lane
- Left lane
- Right lane
- Emergency zone

Lane changes are only initiated when:

- An obstacle is detected within the configured activation distance.
- The target lane is clear.
- The adjacent lane has been verified to be free.

If no safe maneuver exists, the vehicle performs a controlled stop.

---

# Emergency Stop

An independent emergency stop system monitors the forward region using the Intel RealSense D435i.

If an object suddenly enters the emergency zone:

- Vehicle commands are cancelled.
- The MPC controller is interrupted.
- The QCar stops immediately.

This safety layer operates independently of obstacle avoidance.

---

# Demonstration Procedure

The recommended startup sequence is:

1. Power on the QCar.
2. Start hardware drivers.
3. Launch the EKF.
4. Launch the localization stack.
5. Set the initial pose in RViz.
6. Wait until AMCL converges.
7. Verify the `map → base_link` transform.
8. Launch the MPC controller.
9. The vehicle begins autonomous navigation.

---

# Troubleshooting

## Map does not appear

```bash
ros2 topic echo /map
```

---

## Localization does not converge

```bash
ros2 topic echo /amcl_pose
```

Move the vehicle slightly to provide additional LiDAR observations.

---

## Vehicle does not move

Verify:

- MPC controller is running.
- Localization has converged.
- The processed trajectory file exists.
- No emergency stop is active.

---

## Unexpected Vehicle Stop

Check:

```bash
ros2 topic echo /depth_emergency_stop
```

Also verify the LiDAR obstacle detection status.

---

# Future Work

Future improvements include:

- Dynamic obstacle tracking
- Velocity planning
- Multi-obstacle overtaking
- Automatic localization routine
- Mission scheduler
- Traffic sign recognition
- Fleet coordination
- Web-based monitoring interface

---

# Acknowledgements

This project integrates several established ROS2 frameworks, including Cartographer, Navigation2, AMCL and robot_localization, together with custom-developed software for autonomous navigation on the Quanser QCar platform.