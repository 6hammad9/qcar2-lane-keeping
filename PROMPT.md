# Session start prompt — QCar / ROSbot V2V

Paste this into a fresh session. It is written for an agent with no prior
context. Read `HANDOFF.md` and `RUNBOOK.md` after it.

---

## The task

Two robots share an indoor track. A **QCar** (Quanser, ROS 2 Humble) drives a
recorded 38.8 m loop under MPC. A **ROSbot** (Husarion, ROS 2 Jazzy) sits in
its path. The goal:

> The ROSbot stops the QCar. The QCar passes it. The QCar signals the pass is
> complete. The ROSbot resumes.

Reference paper for the decision/planning design:
**Shu, Zhou & Zhang, "Agile Decision-Making and Safety-Critical Motion
Planning for Emergency Autonomous Vehicles", IEEE T-ITS 26(9):13750-13766,
2025.** Reference implementation:
https://github.com/YimingShu-teay/IDEAM (note the hyphen — the URL printed in
the PDF is missing it and 404s).

## Access

Ask the user for the passwords; they will paste them. Do not assume the ones
in any old transcript are still valid — they should have been rotated.

| | |
|---|---|
| QCar | `nvidia@192.168.0.53`, workspace `~/qcar_v2v_ws` |
| ROSbot | `husarion@192.168.0.110`, V2V files in `~/v2v/` |
| Laptop | `192.168.0.76`, repo at `e:\WSL\Ubuntu-24.04\qcar-izhan` |

There is no key-based SSH. `paramiko` works well for scripted access; a
helper pattern is in the git history. `gh` is not installed.

**Safety.** The QCar drives. Never arm it unless the user says they are
standing beside it. Stage every run: localization, then MPC (does not move),
then confirm alignment, then arm.

## How to stop the car

```bash
pkill -x lidar_overtake
```

**`ros2 topic pub --once /motion_enable ... false` does NOT stop it.**
`lidar_overtake` republishes that topic every cycle and overwrites the
one-shot within ~100 ms. This was the documented procedure and it is wrong.
The physical E-stop is the real stop.

## Traps that cost real time

- `pkill -f <pattern>` over SSH matches the pattern inside your own command
  line and kills your shell. Use `pkill -x` (exact process name).
- Linux truncates process names to 15 chars, so `pgrep -x cartographer_node`
  never matches a running Cartographer. Read the launch log instead.
- `~/.bashrc` on the QCar sets `ROS_DOMAIN_ID=1` and sources `~/ros2_ws`,
  **not** `~/qcar_v2v_ws`. Any shell that does not override this runs old
  code on the wrong domain. Always export `ROS_DOMAIN_ID=42` and
  `ROS_LOCALHOST_ONLY=1` and source `~/qcar_v2v_ws/install/setup.bash`.
- Python nodes load code at process start. Deploying a file does nothing
  until you restart the node.
- The route **crosses itself**; branches run 0.02–0.26 m apart. Any
  nearest-waypoint search must be windowed or it hops branches.
- `tmux` is not installed and `apt` is broken. Do not run `apt --fix-broken`.
- RViz cannot be run from the user's WSL: it is Jazzy, the car is Humble, and
  mixing distros on one DDS domain kills nodes with `std::bad_alloc`. Render
  images instead (see `utils/lidar_probe.py` and the capture patterns in git
  history), or use the MJPEG viewers below.

## What is working (verified on hardware, 2026-08-06)

- **Driving**: 89.9 m continuous (2.3 laps), zero tracking faults, loop seam
  crossed repeatedly. Alignment accepts at 2–3 cm.
- **LiDAR sees the ROSbot**: 76 returns at 0.46 m — but only because a **card
  collar** is taped to the ROSbot at 13–20 cm. The QCar's scan plane is at
  **0.16152 m** (URDF `body_lidar_joint`, confirmed on all 13 copies on the
  car). Without the card the ROSbot returns **literally zero** points: the
  beam passes through the gap between its chassis top and its own LiDAR.
  Anything not crossing 16 cm is invisible. Do not remove the card.
- **Camera**: 20.9 Hz. It was never broken — `enable_camera` defaults to
  `false` and the cartographer launch never overrode it. Pass
  `enable_camera:=true`, and note the node is called `RealsenseCamera`.
- **V2V transport**: raw UDP to port 47100 at 10 Hz, 279 packets proven
  QCar-side. Deliberately **not** DDS — the two robots run different ROS 2
  distros and sharing a domain crashes both.
- **MJPEG viewers**: `python3 camera_web_view.py --topic X --port N`, then
  browse `http://192.168.0.53:8080` (raw) / `:8081` (`/lane_debug`).

## What is NOT working

1. **The overtake does not complete.** The car detects the ROSbot, stops in
   `WAIT_FOR_CLEAR`, and never commands a lateral offset. Two causes, one
   fixed by parameters (see README quick start: bigger boxes, higher
   curvature limits), one still open:
   **the return to lane is gated on `overtake_allowed`**, which is false on
   most of this curved route, so after passing the car stays in the wrong
   lane far too long. Fix: give the return its own, looser curvature test —
   the swept-return corridor already checks the real swept volume, so gating
   additionally on full pass-permission double-counts.

2. **IDEAM lane-probing is written but not wired.**
   `qcar_science_night_pkg/ideam.py` implements the LK/LP/LC state selection,
   the risk gap (Eq. 20), the ellipse barrier (Eq. 25–27) and probe speed,
   with 19 tests. **It is not connected to `OvertakeStateMachine`.** Wiring it
   is the highest-value next step: a curve currently forbids the *whole*
   manoeuvre, so the car freezes; LP would let it keep probing forward. The
   paper measures 26.22 m of extra progress at 40 s from probing.
   LSGM/C-DFS is deliberately not implemented — it ranks six vehicle groups
   across three lanes of traffic that does not exist here.

3. **Lane centering perception is unreliable.** The watchdog bug is fixed
   (it was measuring our own throughput, not camera liveness), so the node
   now reports continuously — but *what* it reports is often wrong. Captured
   frames show it fitting "lane lines" to a wall/floor junction, to the
   ROSbot, and to a person's feet, while reporting `valid=True` at ~85% of
   full offset authority. Root causes: the HSV gate `[0,0,170]-[180,45,255]`
   means "anything pale", the ROI still contains radiators and walls, and
   there is no plausibility or obstacle masking.
   **Recommendation: leave it disabled.** Its whole authority is ±4 cm on
   straights while the MPC already tracks at 0.02–0.06 m. Fixing it is a
   perception rebuild for negligible gain.

4. **ROSbot is not localized** (`loc=0` in every V2V packet). It runs no SLAM
   and no AMCL — only EKF — so it cannot report a map pose, and every V2V gate
   needs `on_path` and a signed gap. nav2 is installable
   (`ros-jazzy-nav2-amcl`, the robot is online) but is not installed. Beyond
   that, the two robots localize in **different maps**, so poses are not
   comparable without a shared map or a measured map→map transform.

5. **V2V collision-avoidance gap.** `should_inject_slow_v2v_lead()` opens with
   `if lead_blocked is not True: return False`, so a ROSbot merely sitting in
   the path is never injected as an obstacle. That gate was written to answer
   "should I overtake?", where demanding a blocked report is right, but it is
   also the only path by which V2V can say "something is there". Split those
   two questions.

6. **Loop seam.** The route was closed with a synthesised 1.07 m bridge over
   ground the car never drove. Its curvature previews at **3.728** against a
   0.40 default limit and there is a wall at 0.55 m. Never stage an overtake
   there. Proper fix is re-recording the route (`RUNBOOK.md` §4).

## Suggested order

1. Wire `ideam.constraint_state()` into `OvertakeStateMachine` so
   `WAIT_FOR_CLEAR` becomes lane-probing. Test, deploy, run with the user
   beside the car.
2. Fix the return gate so the car comes back promptly.
3. Then V2V: install nav2 on the ROSbot, shared map, split the
   collision-avoidance path from pass authorisation.

Do not touch lane centering unless asked.

## Repo

Work is pushed to `https://github.com/6hammad9/qcar2-lane-keeping` on branch
`fix/lidar-curve-corridor`. It is an **orphan branch** — the original repo has
~2.4 GB of committed VS Code caches (`browse.vc.db`, three ~1 GB revisions)
and a 185 MB rosbag, all over GitHub's 100 MB file limit, so its history
cannot be pushed. The ROSbot package is included under `rosbot_opta2/`.

Run the tests before and after any change:

```bash
PYTHONPATH=src/qcar_science_night_pkg python3 -m pytest \
  src/qcar_science_night_pkg/test sim/test_sim_rosbot_safety.py \
  sim/test_sim_geometry.py -q \
  --ignore=src/qcar_science_night_pkg/test/test_flake8.py \
  --ignore=src/qcar_science_night_pkg/test/test_pep257.py \
  --ignore=src/qcar_science_night_pkg/test/test_copyright.py
```

154 passing as of 2026-08-06.
