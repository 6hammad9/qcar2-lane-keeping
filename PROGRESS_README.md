# QCar Physical Deployment — Progress Log & Demo Guide

> **2026-08-01 safety update:** the fixed `(0, 0, yaw=0)` initial-pose command
> below is not valid for every trajectory, and the committed old trajectory is
> incompatible with `track_map_new`. Use
> [`QCAR_STEERING_LOCALIZATION_FIX.md`](QCAR_STEERING_LOCALIZATION_FIX.md) for
> the current validation, launch, steering-test, run, and forced-stop sequence.

**Last updated:** 2026-07-09
**Status:** Localization confirmed working (visually + numerically). Trajectory recording is the next step.

---

## 1. What We Were Trying To Do

Get the QCar's existing ROS2 Humble software stack running on the **physical car** (not simulation), following the project's documented workflow:

```
Hardware/software setup
      │
      ▼
Cartographer mapping (drive manually, build a map of the room)
      │
      ▼
Save map (.pbstream + .pgm/.yaml)
      │
      ▼
AMCL localization (car figures out where it is on the saved map)
      │
      ▼
Trajectory recording (drive the track once, record the path)
      │
      ▼
Trajectory processing (smooth the recorded path)
      │
      ▼
Launch MPC controller → fully autonomous driving
```

We are currently **at the localization stage**, confirmed working both numerically (AMCL covariance below target thresholds) and visually (RViz shows the live LiDAR scan correctly overlapping the pre-built map).

---

## 2. Current Approach

- **Everything hardware-facing runs natively** on the QCar's own Ubuntu 20.04 / ROS2 Humble install — motors, LiDAR, camera, EKF, Cartographer, AMCL, and the MPC controller all run directly on the car's onboard Jetson computer, *not* inside Docker.
- **Docker (Isaac-ROS container) is used only for visualization** — specifically to run RViz2 so we can *see* the map, live LiDAR scan, and the car's estimated pose. It is not part of the control/driving pipeline.
- **`ROS_DOMAIN_ID=42`** is used for every terminal session on the QCar. This isolates the car's ROS2 traffic from other devices on the same WiFi network (see Section 4, Issue 3, for why this matters).
- The car's SSH IP is currently **`192.168.0.53`**, user **`nvidia`**.

---

## 3. What "Localization" Actually Does (plain explanation)

Two different things are involved, and they're easy to confuse:

| | The Map (black outline in RViz) | The Live Scan (white lines in RViz) |
|---|---|---|
| What it is | A fixed, saved floor plan of the room, built once by driving around with Cartographer | What the LiDAR sees *right now*, updated continuously |
| When it's created | Once, during the mapping step | Every scan cycle, in real time |
| Does it change? | No — stays fixed once saved | Yes — constantly updates as the car moves |

**AMCL's job**: given the fixed map and the live scan, figure out *where on that map* the car currently is and which way it's facing. It does this by testing many candidate positions/orientations and finding the one where the live scan best matches the map's walls. When the white live-scan lines visually line up with the black map walls in RViz, that means AMCL has correctly determined the car's real-world position — this is what lets the car know "where it is" without GPS.

---

## 4. Problems We Hit & How We Fixed Them

### Issue 1 — Duplicate nodes in the launch file
`qcar2_cartographer_original_launch.py` both included `qcar2_launch.py` (which already starts `nav2_qcar2_converter` and `fixed_lidar_frame`) **and** separately re-declared those same two nodes itself. Fixed by removing the duplicate `Node(...)` blocks and their entries in the `LaunchDescription([...])` list.

### Issue 2 — `/cmd_vel` vs `/cmd_vel_nav`
`nav2_qcar2_converter` (the node that actually talks to the motors) subscribes to **`/cmd_vel_nav`**, not the standard `/cmd_vel`. Manual teleop driving must remap accordingly:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_nav
```

### Issue 3 — Every process crashing with `std::bad_alloc`
The entire node graph (Cartographer, hardware interface, LiDAR, camera, everything) was crashing simultaneously, a few seconds after startup, every single time.

**Root cause (confirmed via Apport crash report + `gdb` backtrace):** a corrupted DDS discovery packet deserialization, almost certainly caused by ROS2 domain collision — this laptop (ROS2 Jazzy) and the QCar (ROS2 Humble) were both on the default `ROS_DOMAIN_ID=0` on the same WiFi network. The version mismatch in `rmw_dds_common` message definitions between Jazzy and Humble corrupted the discovery protocol, crashing every participant on the domain at once.

**Fix:** isolate the QCar onto its own domain:
```bash
export ROS_DOMAIN_ID=42
```
This must be set in **every terminal** used for QCar work, including inside the Docker container.

### Issue 4 — Stale map references
Three different files pointed at three different, outdated maps (`track_map.yaml`, `my_map_03.yaml`, `full_track_map.yaml`). Since the physical track layout had changed since those were made, we remapped from scratch and updated both `science_night_slam.launch.py` and `amcl_config.yaml` to point at the fresh map (`track_map_new.yaml`).

### Issue 5 — RViz visualization (Map display QoS mismatch)
Adding the Map display "by display type" left it stuck on "No map received" even though `map_server` was actively publishing. Cause: QoS mismatch — `map_server` publishes with `Durability: Transient Local`, but RViz's default Map display QoS didn't match. Fixed by setting the Map display's **Durability Policy → Transient Local** manually, or by adding the display via **"By topic"** instead of "By display type" (which auto-negotiates the correct QoS).

---

## 5. Full Demo Commands (run in this order)

Open **three separate SSH terminals** to the car for a live demo. In every terminal, start with:
```bash
ssh nvidia@192.168.0.53
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
```

### Terminal 1 — Localization stack (map + AMCL)
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch qcar_science_night_pkg science_night_slam.launch.py
```
Wait for it to settle — you'll see `AMCL cannot publish a pose or update the transform. Please set the initial pose...`. This is expected; it means AMCL is alive and waiting.

### Terminal 2 — Set the initial pose
Only needed once per session, right after Terminal 1 is up:
```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: 0.0, y: 0.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    covariance: [0.25, 0, 0, 0, 0, 0,  0, 0.25, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0.06]
  }
}"
```
(Assumes the car starts at roughly the same position/orientation as when mapping began. Adjust x/y/yaw if starting elsewhere.)

Then drive the car a short distance with some turning motion (not just straight) to help AMCL converge — see Terminal 3 below. Check convergence with:
```bash
ros2 topic echo /amcl_pose --once
```
Target: x, y, and yaw covariance all below ~0.02.

### Terminal 3 — Manual driving (teleop)
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_nav
```
Controls: `i` forward, `,` back, `j`/`l` turn, `k` stop, `q`/`z` speed up/down.

### (Optional) RViz visualization — separate terminal, needs X11 forwarding
On your laptop, connect with `-X`:
```bash
ssh -X nvidia@192.168.0.53
```
Then:
```bash
cd ~/Documents/ACC_Development/isaac_ros_common
export ROS_DOMAIN_ID=42
./scripts/run_dev.sh /home/nvidia/Documents/ACC_Development/Development
```
Inside the container:
```bash
export ROS_DOMAIN_ID=42
rviz2
```
In RViz: set **Fixed Frame → map**, then **Add → By topic** and add `/map` (Map), `/scan` (LaserScan), and `/amcl_pose` (PoseWithCovariance).

---

## 6. Mapping From Scratch (if the track layout changes again)

```bash
cd ~/ros2_ws
source install/setup.bash
export ROS_DOMAIN_ID=42
ros2 launch qcar2_nodes qcar2_cartographer_original_launch.py
```
Drive the full track slowly and smoothly (teleop, same as above but remapped to `/cmd_vel_nav`), covering every corridor, ending back near the start.

Save the map (**do this before stopping Cartographer**):
```bash
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '/home/nvidia/ros2_ws/track_map_new.pbstream'}"
ros2 run nav2_map_server map_saver_cli -f /home/nvidia/ros2_ws/track_map_new
```
Then stop Cartographer (`Ctrl+C`), and re-launch the localization stack (Terminal 1 above) to use the fresh map.

---

## 7. Next Steps

```
✅ Physical hardware driving confirmed
✅ Mapping (Cartographer) confirmed
✅ Localization (AMCL) confirmed — numerically converged AND visually verified in RViz
      │
      ▼
🔲 Trajectory Recording — drive the track once cleanly with the recorder running,
     capturing the reference path for the MPC controller
      │
      ▼
🔲 Trajectory Processing — run the recorded path through the smoothing script
     to produce the final .npy reference trajectory (x, y, yaw, curvature)
      │
      ▼
🔲 Launch the MPC Controller — car drives the recorded path autonomously,
     with LiDAR-based obstacle avoidance and emergency stop active
```

**Immediate next action:** confirm the trajectory recorder node exists and run it during a clean drive of the track:
```bash
ros2 pkg executables qcar_science_night_pkg | grep trajectory
ros2 run qcar_science_night_pkg qcar2_trajectory_recorder
```
