# V2V: QCar ↔ ROSbot Cooperative Driving

Asymmetric vehicle-to-vehicle communication between the ROSbot and the QCar:

- **ROSbot** broadcasts its pose, speed, and predicted future trajectory at
  10 Hz. Its trajectory follower is **completely unchanged** — the broadcaster
  is a separate, standalone node.
- **QCar** receives, validates, and reacts: it **slows**, **follows**,
  **stops**, or **overtakes** the ROSbot. Only the QCar optimizes; the ROSbot
  never reacts to the QCar.

Architecturally this mirrors the IDEAM paper (Shu et al., IEEE T-ITS 2025):
the surrounding vehicle becomes a predicted obstacle constraining the ego
MPC through a discrete control barrier function (DCBF), while a behavioral
layer decides lane changes. Two deliberate improvements over the paper's
setting: the ROSbot's prediction is *exact* (it follows a recorded path — no
IDM guessing), and the DCBF carries per-stage slack so the solver can never
go infeasible (paper §V-B: slack variables + constraining only the first
`N_dho < N` stages).

---

## 1. Design decisions and why

### UDP link, not DDS

This lab's Wi-Fi has **twice** crashed the entire ROS graph via corrupted
Fast DDS discovery when mixed ROS 2 distros shared a domain (see
`PROGRESS_README.md` issue 3, and again on 2026-08-01: every node died with
`std::bad_alloc` within a second). A DDS bridge between the Humble QCar and
the ROSbot would recreate exactly that hazard.

Instead, **both robots keep their ROS graphs private**
(`ROS_LOCALHOST_ONLY=1`, or a unique `ROS_DOMAIN_ID` per robot — both work)
and V2V rides one raw UDP socket that DDS discovery cannot touch. Bonus: no
custom `.msg` package needs to be compiled identically on two different ROS
distros. Packet loss is harmless — every datagram is a complete state
refresh, and freshness is judged by **arrival time**, so the robots do not
need synchronized clocks.

### Fail-safe contract (the most important property)

| Condition | QCar behavior |
|---|---|
| Receiver not running / link down / packets stale | **Identical to the pre-V2V stack.** `/v2v/alive` false, DCBF fed a 50 m placeholder (trivially satisfied), governor off, LiDAR fusion off. |
| Packet malformed / wrong vehicle / NaN / out of range | Dropped and counted; never applied. |
| ROSbot not localized | Heartbeats marked `loc: 0` → treated as *no data*, never as "clear". |
| Fresh data | QCar can only become **more** cautious — except overtakes, which still require the physical LiDAR to confirm the left lane is clear. |

Safety priority order is unchanged: depth e-stop > LiDAR emergency/state
machine > V2V governor/DCBF > MPC tracking.

**Both halves are opt-in together.** `path_mpc`'s `enable_v2v` and
`lidar_overtake`'s `v2v_fusion_enable` both default to `false`. The LiDAR half
is what *starts* a V2V pass; the MPC half is the DCBF keep-out and the speed
governor that make one safe. Enabling only the first is fail-open, so the
defaults now match — set both, or neither.

**Threshold coupling that must hold.** `v2v_return_clearance_m` >
`v2v_ellipse_a`. `RETURN_RIGHT` drives the lateral offset to zero, so a merge
authorized inside the MPC's longitudinal keep-out is one the MPC's own barrier
then refuses: the post-solve shield publishes zero velocity with the car still
straddling the lane while the vehicle behind keeps closing. The node validates
its own floor (0.60 m); `test_return_clearance_must_clear_the_mpc_safety_ellipse`
pins the invariant.

### One map frame

Gap and lane logic assume both robots localize against **the same map of the
track** (transform = identity). If the maps are ever separate again, measure
two shared landmarks in both frames and set `frame_tx/ty/tyaw` in
`config/v2v_params.yaml` — everything else is unchanged.

---

## 2. What was added / changed

**New files**

| File | Runs on | Purpose |
|---|---|---|
| `opta2/rosbot_lane/rosbot_v2v_broadcaster.py` | ROSbot | Self-contained sender: TF pose → speed estimate → path-based prediction → UDP at 10 Hz. Copy this ONE file to the ROSbot. |
| `src/.../v2v_common.py` | QCar | Pure logic: wire format, validation, SE2, path projector. No ROS imports — unit-tested offline. |
| `src/.../v2v_receiver_node.py` | QCar | UDP → validation → frame transform → projection → `/v2v/*` topics + CSV log. |
| `config/v2v_params.yaml` | QCar | Receiver parameters (all optional). |
| `utils/v2v_fake_rosbot.py` | anywhere | Replays a trajectory over UDP in the real wire format — bench-test the whole QCar side with no ROSbot. |
| `utils/v2v_selftest.py` | anywhere | Offline test of all V2V math against the real trajectory. **33/33 passing.** |

**Modified files (QCar)**

- `path_mpc_node.py` —
  1. *DCBF safety layer*: rotated elliptical keep-out (a=0.55 m along the
     ROSbot's heading, b=0.40 m lateral) around the ROSbot's predicted pose
     at each of the first `N_dho = 12` MPC stages, enforced as
     `h(k+1) ≥ (1-γ)·h(k)` with γ=0.35 and per-stage slack (quadratic
     penalty 400). Slack means a marginal solve gets expensive instead of
     infeasible — no phantom hard-stops from solver failures.
  2. *Speed governor*: when fresh V2V shows the ROSbot ahead **on our path**
     within 3 m, `target_v` is capped by
     `min( √(2·0.5·(gap−0.70)), v_rosbot + 0.5·(gap−1.20) )` —
     smooth slow-down, speed-matched following at ~1.2 m, full stop ~0.7–1.0 m
     behind a stopped ROSbot. Active only in `DRIVE` state; overtake states
     keep their existing 0.3 m/s logic.
  3. New topics consumed: `/v2v/alive`, `/v2v/gap`, `/v2v/on_path`,
     `/v2v/rosbot_speed`, `/v2v/predicted_path`.
  4. Tracking log gains `v2v_active,v2v_gap,v2v_cap` columns.
- `lidar_overtake_node.py` — *early-warning fusion*: a **stopped** ROSbot on
  our path within 1.5 m is injected as a virtual front obstacle, so the
  overtake state machine can decide ~5 s earlier than the 0.9 m LiDAR box.
  A **moving** ROSbot is deliberately not injected (the governor follows it
  instead of freezing in WAIT). Injection is strictly additive — it can set
  `obstacle_ahead` / shorten `front_min` but never clear anything, and
  left/right/emergency checks stay LiDAR-only.
- `setup.py` — `v2v_receiver` console script.

**Not changed:** the ROSbot's `trajectory_follower_node.py` and everything
else in `rosbot_lane`; the QCar's depth e-stop, sound, lane-centering, EKF,
AMCL, hardware nodes.

---

## 3. Deployment

### QCar (192.168.0.53)

```bash
# copy the updated package and rebuild
scp -r src nvidia@192.168.0.53:~/ros2_ws/        # or your workspace
ssh nvidia@192.168.0.53
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42        # or ROS_LOCALHOST_ONLY=1 — both fine for V2V
colcon build --symlink-install --packages-select qcar_science_night_pkg
source install/setup.bash
```

Run order (each terminal: `source` + same `ROS_DOMAIN_ID`/`LOCALHOST` env):

```bash
# T1: localization stack (unchanged)
ros2 launch qcar_science_night_pkg science_night_slam.launch.py
# T2: set /initialpose, verify /amcl_pose covariance < 0.02 (unchanged)

# T3: V2V receiver
ros2 run qcar_science_night_pkg v2v_receiver

# T4: MPC (after localization converges) — V2V is OPT-IN via parameter:
ros2 run qcar_science_night_pkg path_mpc --ros-args -p enable_v2v:=true
```

Without `enable_v2v:=true` the MPC ignores all V2V input (its DCBF stays on
the inactive placeholder) — that is the default, so a plain `path_mpc` run
behaves exactly like the pre-V2V controller.

Firewall note: the receiver binds UDP 47100; Jetson Ubuntu ships without
ufw enabled, but if `sudo ufw status` says active:
`sudo ufw allow 47100/udp`.

### ROSbot

```bash
# copy ONE file
scp opta2/rosbot_lane/rosbot_v2v_broadcaster.py husarion@<rosbot-ip>:~/

# on the robot, after tf_relay + SLAM/AMCL are up (unchanged workflow):
python3 ~/rosbot_v2v_broadcaster.py --ros-args \
  -p target_ip:=192.168.0.53 \
  -p trajectory_csv:=/home/sharjeel-ahmad/Documents/rosbot_ws/src/rosbot_lane/config/smoothed_trajectory.csv
```

The follower runs exactly as before — the broadcaster only reads TF.

### Verify the link (before any driving)

```bash
# QCar:
ros2 topic echo /v2v/stats      # rx counting up, age_s ~0.1, parse_errors 0
ros2 topic echo /v2v/alive      # true
ros2 topic echo /v2v/gap        # sensible along-path meters
```

---

## 4. Bench test WITHOUT the ROSbot (do this first)

Everything QCar-side can be validated with the fake broadcaster — car on a
stand or e-stopped, wheels free:

```bash
# T1: receiver
ros2 run qcar_science_night_pkg v2v_receiver
# T2: virtual ROSbot 200 waypoints ahead on the QCar's own path, 0.25 m/s,
#     pausing 10 s every 20 s (exercises follow AND stop AND overtake-inject):
python3 utils/v2v_fake_rosbot.py \
  --path ~/ros2_ws/recorded_path_amcl_final_long.npy \
  --target 127.0.0.1 --speed 0.25 --start-idx 200 \
  --pause-every 20 --pause-for 10
# T3: watch
ros2 topic echo /v2v/gap
```

Then start the MPC with the car held safe and watch its log line:
`v2v=ON, v2v_gap=..., v2v_cap=...` — the cap should shrink as the virtual
gap closes, hit 0 when the fake pauses inside 1 m, and release afterwards.

Kill the fake mid-run: within 0.6 s `/v2v/alive` must go false and the MPC
line must return to `v2v=off` with normal speeds — that is the fail-safe
contract working.

Offline (no ROS at all): `python3 utils/v2v_selftest.py` → 33 checks.

---

## 5. Expected behaviors on track

| Scenario | What happens |
|---|---|
| ROSbot far ahead / behind / other lane | Nothing. Normal driving. |
| ROSbot moving ahead, same lane, gap < 3 m | Governor caps speed; QCar settles ~1.2 m behind, matching speed. DCBF active as backstop. |
| ROSbot stops (waypoint pause) | Prediction collapses to stationary; QCar brakes smoothly, stops ~0.7–1.0 m behind. |
| ROSbot stopped, straight section, left lane clear (LiDAR) | Injected obstacle at ≤1.5 m → state machine goes `OVERTAKE_LEFT` early → 0.55 m offset pass → `RETURN_RIGHT`. DCBF keeps ≥ ellipse clearance throughout. |
| ROSbot stopped, curve or left blocked | QCar waits behind (governor hold / `WAIT_FOR_CLEAR`), resumes when clear or ROSbot moves. |
| Wi-Fi drops mid-approach | ≤ 0.6 s later V2V goes stale → governor releases, DCBF idles — but the LiDAR stack still sees the physical ROSbot exactly as it did before V2V existed. |

## 6. Evaluation / experiments

Logs produced every run:

- `~/ros2_ws/v2v_rx_log.csv` (receiver): per-packet arrival gaps (link
  quality), seq drops, gap/lateral trace.
- `~/ros2_ws/mpc_tracking_log.csv` (MPC): now includes
  `v2v_active,v2v_gap,v2v_cap` alongside speed/tracking columns.

Suggested experiment matrix (each: one lap, log both CSVs):

1. **Baseline** — ROSbot on track, V2V receiver OFF (LiDAR-only behavior).
2. **V2V follow** — receiver ON, overtaking disabled
   (`v2v_fusion_enable=False`): measure approach smoothness, standoff gap.
3. **V2V overtake** — full stack: measure decision distance (gap at
   `OVERTAKE_LEFT` entry) vs. baseline's 0.9 m, and minimum clearance
   during the pass.
4. **Link-loss injection** — kill the broadcaster mid-follow: verify
   fallback within `stale_sec`.

Metrics to report: packet rate & loss (`/v2v/stats`), decision distance,
min inter-vehicle gap, speed profile smoothness (jerk), time-to-pass.

## 7. Tuning quick reference (all in-code, top of each file)

| Knob | Where | Default | Effect |
|---|---|---|---|
| `v2v_stop_gap` | path_mpc | 0.70 m | standoff behind stopped ROSbot |
| `v2v_follow_gap` | path_mpc | 1.20 m | steady following distance |
| `v2v_slow_start` | path_mpc | 3.0 m | governor engagement range |
| `v2v_ellipse_a/b` | path_mpc | 0.55/0.40 m | DCBF keep-out size |
| `v2v_ndho` | path_mpc | 12 | constrained stages (solve cost ↑ with it) |
| `v2v_detect_range` | lidar_overtake | 1.5 m | early-inject range (stopped ROSbot) |
| `v2v_return_clearance_m` | lidar_overtake | 0.80 m | signed lead required before `RETURN_RIGHT` — **must exceed `v2v_ellipse_a`** |
| `stale_sec` | receiver | 0.6 s | link-loss fallback latency |
| `lane_half_width` | receiver | 0.35 m | "in our lane" threshold |

## 7b. Verified in simulation (2026-08-01)

The full chain ran closed-loop in Gazebo against the real map — see
`sim/SIM_README.md`. Headline results: 1024 packets with **0 parse errors
and 0 sequence drops**, MPC driving at 0.008-0.034 m tracking error with
`v2v=ON`, governor engaging correctly below 3 m.

**One real defect was found and fixed there**: `PathProjector.closest_idx()`
used a global nearest-waypoint search, and because the recorded path passes
within **7 mm of itself** (waypoints 534 and 220), the projection could flip
across the loop for a single frame — reporting a 2.8 m gap when the true gap
was 13.6 m, and engaging the speed governor for a vehicle 13 m away. Now
fixed with a windowed search seeded by the previous index (global re-acquire
only past `reacquire_dist`), verified over 591 samples: max frame-to-frame
index jump fell from ~200 (10 m) to **3 (0.15 m)**, zero flips. A regression
test guards it; the suite is **36/36** on both trajectories.

This is the class of bug that only appears with a real trajectory on a real
map, which is exactly why the sim exists. Run it before any hardware
session after changing paths or maps.

**A second finding applies directly to the physical car, independent of
V2V.** The QCar could not track the ROSbot-recorded path: that path was
recorded by a differential-drive robot which can pivot in place, and demands
a 0.456 m turning radius, while the QCar at the `max_steer=0.50` default has
a 0.469 m minimum — geometrically impossible. Worse, once steering
saturates the MPC *raises* speed to chase yaw rate (yaw rate = v/L*tan(δ)),
so it commanded 0.30 m/s against a 0.15 curve target in 94% of samples.
Setting `max_steer:=0.58` (the true hardware limit) and capping
`max_speed:=0.15` took worst tracking error from **0.474 m to 0.024 m**.
Verify both parameters before the next track session — see
`sim/SIM_README.md` for the full measurements.

## 8. Known limits (honest list)

- The DCBF adds 12 nonlinear constraints + 12 variables to the IPOPT solve.
  Expected cost on the Jetson is small, but **watch the control-loop rate on
  first hardware run**; if it degrades, drop `v2v_ndho` to 8.
- Along-path gap ≈ straight-line distance only at short range/gentle
  curvature — the ranges involved (≤3 m) keep the error negligible, but the
  fused `front_min` is an approximation, not a measurement.
- Prediction ages up to one packet period (100 ms) plus Wi-Fi latency
  (~20–60 ms measured) before the MPC consumes it: ≤ ~7 cm of ROSbot motion
  at 0.4 m/s, well inside the ellipse margins.
- The wire format is unauthenticated JSON. Validation rejects malformed and
  wrong-ID packets, but anyone on the LAN could forge valid-looking ones.
  Fine for a closed lab network; do not use on an open network.
