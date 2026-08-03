# QCar2 / ROSbot V2V simulation

This is the canonical two-vehicle simulation for the half-map experiment. It
uses the occupancy map from the physical track, one shared closed route, a
dynamic Ackermann QCar2 model, and a kinematic ROSbot traffic proxy that sends
the same UDP V2V packet format as the real ROSbot.

The two route files are intentionally identical in position and heading:

- `assets/smoothed_trajectory.csv` drives the simulated ROSbot.
- `assets/qcar_half_map_centerline.npy` is the QCar MPC reference.

The current validated route contains 526 points, is approximately 26.30 m
around, has approximately 0.05 m spacing including its closed seam, and stays
in known free map space. `run_mpc.sh` runs `validate_sim_geometry.py` before
starting the controller; a failed route, map, lane corridor, or steering check
blocks MPC startup.

The ROSbot's visually perfect tracking is by construction: its pose is
interpolated directly on the canonical route. It is useful as repeatable V2V
traffic, but it is not evidence that a ROSbot controller outperforms QCar MPC.

## Lane meaning

All left/right directions below are relative to the direction of travel, not
fixed map axes.

| Road feature | Offset left of canonical route | Meaning |
|---|---:|---|
| Right solid edge | -0.225 m | Outside edge of the normal lane |
| Canonical route | 0.00 m | Center of the normal/right lane |
| Dashed divider | +0.21 m | Boundary between the two lanes |
| Overtake center | +0.47 m | Center of the left passing lane |
| Left solid edge | +0.695 m | Outside edge of the passing lane |

Both vehicles normally travel inside the right lane on the `0.00 m` route;
neither is supposed to drive on the dashed line. During `OVERTAKE_LEFT`, QCar
crosses the divider toward the `+0.47 m` left-lane reference. During
`RETURN_RIGHT`, it blends back to the canonical right-lane center.

The Gazebo controller requests up to `+0.52 m` during the pass. This is a
measured Ackermann tracking calibration: the simulated body settles about
`0.06--0.07 m` inside that reference, placing the physical vehicle near the
`+0.47 m` lane centre. The `+0.52 m` reference corridor is separately checked
against the occupancy map before MPC starts.

QCar motion is primarily map-localized waypoint MPC. The front camera is also
active by default, but only supplies a small bounded centering correction in
the normal lane when both painted boundaries are valid. Stale or ambiguous
vision disables that correction. Camera output does not replace localization,
MPC, LiDAR safety, or the explicit overtake offset. Use `--no-lane` to test
without the vision correction.

## Recommended V2V acceptance run

Run these commands in WSL:

```bash
cd /mnt/e/WSL/Ubuntu-24.04/qcar-izhan

# Start the short, deterministic moving-lead overtake scenario.
bash sim/wsl/start_all.sh --v2v-demo

# Process status. Every listed component should be UP.
bash sim/wsl/status_all.sh

# Live QCar/ROSbot decisions, received gap, and commanded speeds.
bash sim/wsl/watch_demo_live.sh

# Stop safely when finished.
bash sim/wsl/stop_all.sh
```

`--v2v-demo` configures:

- QCar start at waypoint 419, the first curvature-approved passing waypoint.
- ROSbot start at waypoint 436, approximately 0.85 m centre-to-centre ahead
  (about 0.46 m of bumper clearance). QCar's following governor holds until
  the moving lead reaches the LiDAR-approved passing distance.
- ROSbot cruises continuously at 0.03 m/s and broadcasts its moving horizon;
  it has no scheduled stop and loops the closed route. This demo speed is
  required to finish the measured catch-up plus both lane changes on
  the available 2.60 m straight; the normal non-demo speed remains 0.07 m/s.
- QCar is faster (up to 0.15 m/s normally and 0.10 m/s while maneuvering), so
  it can catch and pass the slower traffic on waypoints 416-468.
- QCar uses the validated closed route with a two-lap target.

ROSbot is launched after the QCar safety and MPC components, preventing it
from drifting out of the straight while those components initialize. Motion
uses `/clock`, so a low Gazebo real-time factor changes wall-clock duration but
not scenario geometry or simulated timing.

Flags can be combined:

```bash
bash sim/wsl/start_all.sh --v2v-demo --headless
bash sim/wsl/start_all.sh --v2v-demo --no-lane
bash sim/wsl/start_all.sh --no-mpc
```

The ordinary command without `--v2v-demo` starts QCar at waypoint 0 and
ROSbot at waypoint 60. QCar performs one non-looping traversal by default;
ROSbot loops continuously at 0.07 m/s. This is a longer follow scenario,
whereas `--v2v-demo` is the fast moving-lead acceptance setup.

Scenario values may be overridden before starting, for example:

```bash
ROSBOT_SIM_SPEED=0.025 bash sim/wsl/start_all.sh --v2v-demo
```

Other supported environment overrides are `QCAR_SIM_START_IDX`,
`ROSBOT_SIM_START_IDX`, `ROSBOT_SIM_SPEED`, `ROSBOT_SIM_STOP_DISTANCE`,
`ROSBOT_SIM_SLOW_DISTANCE`, `ROSBOT_SIM_CLEAR_DISTANCE`,
`QCAR_SIM_LOOP_PATH`, and `QCAR_SIM_TARGET_LAPS`. Keep both vehicles on the
canonical route and choose a validated straight if the test is expected to
overtake. The physical-style defaults progressively slow ROSbot below 0.90 m,
reserve a hard stop for less than 0.40 m, and release that latch above 0.50 m.

## Expected behavior stages

The exact transition time depends on localization, LiDAR and MPC conditions;
do not judge it only by elapsed wall time.

| Stage | Expected result |
|---|---|
| `STARTUP_WAIT` | Motion remains disabled until LiDAR is received and the safety chain is armed. |
| `DRIVE` / follow | QCar remains in the normal/right lane. With a fresh moving lead below 3 m, the V2V governor can only reduce speed; its nominal follow gap is 1.20 m. |
| V2V hold | At or below the 0.70 m stop gap, or when the computed cap is nearly zero, QCar publishes a direct zero command rather than relying on a soft MPC cost. |
| `WAIT_FOR_CLEAR` | QCar stays stopped if the maneuver is too close, the left lane is occupied, the route preview is too curved, or another required condition is not confirmed. |
| `OVERTAKE_LEFT` | On a permitted straight with the left lane clear, QCar blends toward the +0.47 m passing-lane center over a configured 0.90 m lane-change distance. |
| Pass confirmation | A V2V-triggered pass is not complete merely because LiDAR loses the front return. The signed V2V gap must show the known ROSbot behind, measured QCar progress must be sufficient, yaw must be stable, and the return lane/route preview must be safe. |
| `RETURN_RIGHT` | QCar blends back to offset 0. If the right/original lane is blocked, it remains in the passing lane rather than steering into that obstacle. |
| `DRIVE` | Only after measured return progress does the controller resume normal right-lane driving. |

The coordination is asymmetric: ROSbot shares its current state and predicted
horizon; QCar uses those data in its governor and MPC. In the recommended
moving-lead demo, ROSbot maintains its slow cruise and does not optimize or
change lanes in response to QCar. Its independent local proximity governor
may reduce longitudinal speed when QCar is physically ahead and close, and
reserves zero speed for the 0.40 m emergency region. That sensor reflex does
not consume QCar's V2V horizon and does not make the MPC symmetric.

That reflex now mirrors `trajectory_follower_node` geometry exactly: a front
region one lane wide (`±0.20 m`, from `_lane_width / 2`) reporting a *surface*
range, not centre-to-centre. It previously used a `±0.35 m` band and a
centre-to-centre distance, so the simulated ROSbot braked for a QCar out in
the passing lane that the physical robot ignores — which is what made it crawl
through every overtake.

The road visual is medium-grey PBR asphalt with white boundaries. Occupancy
map wall boxes remain active collision/LiDAR geometry, but their giant visual
boxes are transparent so they cannot cover the operator view or camera image.
For a non-headless run, the launcher selects a close chase camera on QCar2.

## Safety expectations

- Gazebo LiDAR is remapped from `/qcar2/lidar/scan` to `/scan` with the
  simulation's 0-degree forward mounting. The physical QCar default remains
  180 degrees.
- A missing/stale LiDAR behavior heartbeat, LiDAR timeout after arming, invalid
  path/map, bad start alignment, poor localization covariance, missing TF,
  excessive tracking error, solver failure, or rejected predicted horizon
  produces a zero command.
- An unknown obstacle closer than 0.40 m in the forward collision corridor
  has emergency-stop authority. The only narrow exception is the identified
  V2V lead during an already committed pass when fresh relative geometry and
  the physical scan agree that it is safely offset.
- The predicted ROSbot horizon is protected by a hard elliptical DCBF
  constraint with zero allowed barrier slack, followed by a separate predicted
  clearance check before command publication.
- Following and V2V-assisted overtake injection require fresh, localized data
  with ROSbot on the shared path. A stale link disables those cooperative
  decisions; it is not by itself proof that the road is clear. Physical LiDAR
  behavior remains authoritative, while fresh predicted geometry can still be
  checked by the MPC safety barrier.
- Overtake and return completion use measured route progress, not a count of
  sensor callbacks, so repeated scans while stationary cannot finish a pass.
- `cmd_bridge.py` publishes zero if MPC commands become stale.
- `stop_all.sh` publishes zero first, then stops the recorded process groups.
  Do not reset by teleporting the QCar body: `reset_qcar.sh` performs a full
  supervised stop/start so odometry and `map -> odom` remain consistent.

These checks make failures stop conservatively, but Gazebo results do not by
themselves certify hardware safety. Begin physical tests at low speed with a
manual emergency stop available.

## Watching the run

Process status only tells whether components are alive. Use the ROS topics and
supervised logs to see behavior:

```bash
source sim/wsl/common.sh
setup_ros

ros2 topic echo /drive_state
ros2 topic echo /motion_enable
ros2 topic echo /v2v/stats
ros2 topic echo /v2v/gap
ros2 topic echo /lane_center_valid
```

For a positive `/v2v/gap`, ROSbot is ahead; after QCar has passed it, the gap
becomes negative. `NaN` means the relative path gap is unknown.

Logs live under `/tmp/qcar_v2v_sim_$UID/logs/`:

```bash
tail -F /tmp/qcar_v2v_sim_$(id -u)/logs/mpc.log
tail -F /tmp/qcar_v2v_sim_$(id -u)/logs/lidar.log
tail -F /tmp/qcar_v2v_sim_$(id -u)/logs/rosbot.log
```

Useful evidence in the logs includes `v2v_cap`, `V2V hold`, `v2v=INJ`,
`pass_ok`, `OVERTAKE_LEFT`, `RETURN_RIGHT`, tracking error, and published
offset. If QCar remains stopped, first check for `STARTUP_WAIT`,
`LIDAR_TIMEOUT`, path-validation errors, start-alignment errors, or missing TF
instead of bypassing `/motion_enable`.

Validate the current assets without starting Gazebo:

```bash
python3 sim/validate_sim_geometry.py
python3 sim/test_sim_geometry.py
```

## Implementation boundary

The local SDF contains occupancy-derived walls, the two-lane road artwork, and
the ROSbot proxy. Static obstacle boxes are not included in the default V2V
world; `obstacles.yaml` is an optional separate sensor-test scenario.

The ROSbot proxy is a self-contained primitive four-wheel visual approximation
with a recognizable chassis, wheels, sensor mast, LiDAR puck and front camera.
It is not an official Husarion model and it has no simulated drivetrain,
odometry, scan, wheel joints or controller plugin. `sim_rosbot.py` moves that
single body directly along the canonical route and publishes the matching V2V
state; this keeps the visible pose, transmitted pose and experiment timing in
one deterministic process.

A local model audit found no complete ROSbot 2 or ROSbot 3 simulation package
in `opta2`, `opta2-sami_ahmed`, `/home/hammad/rosbot_ws`, the local Gazebo model
cache, or the installed ROS shares. The only installed Husarion description is
`husarion_components_description`, which provides sensor/component fragments
but no ROSbot base, chassis/wheel model, differential-drive joints or Gazebo
control plugin. Those missing assets are the exact blocker to using the real
robot model without fetching and integrating an additional package.

`opta2-sami_ahmed` does contain the current physical ROSbot 3 trajectory
follower. It expects `map -> base_link`, `/rosbot3/scan`, and publishes
`TwistStamped` on `/rosbot3/cmd_vel`; it also follows that repository's own CSV
and has independent obstacle/overtake behavior. It is intentionally not wired
into this asymmetric simulation because it is not a Gazebo vehicle model, its
route is not the generated canonical QCar route, and its autonomous overtake
would change the experiment from "ROSbot broadcasts, QCar decides." Integrate
a complete ROSbot simulation description plus its TF/odom/scan bridge first if
controller-in-the-loop ROSbot dynamics are required.

QCar2's URDF, meshes, Ackermann plugin and ROS-Gazebo bridge configuration are
read from `/home/hammad/rosbot_ws` by default. This repository launches its
own local world directly and does not copy or modify that external workspace.
The supervised launcher isolates ROS discovery to localhost on domain 0 unless
`SIM_ROS_DOMAIN_ID` is explicitly set.
