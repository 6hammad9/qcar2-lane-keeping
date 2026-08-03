#!/usr/bin/env python3
"""Simulated ROSbot: drives the `sim_rosbot` model in Gazebo along the REAL
recorded trajectory while broadcasting REAL V2V UDP packets.

One process does both, so what the QCar's simulated LiDAR physically sees and
what V2V claims are always identical — the same consistency the real robots
have. Exercises the full QCar chain: receiver, speed governor, DCBF, LiDAR
fusion, early overtake.

    python3 sim_rosbot.py \
        --csv assets/smoothed_trajectory.csv \
        --world sami_track \
        --target 127.0.0.1 \
        --speed 0.07

Pose updates go through the ros_gz SetEntityPose service bridge if available:
    ros2 run ros_gz_bridge parameter_bridge \
        '/world/sami_track/set_pose@ros_gz_interfaces/srv/SetEntityPose'
otherwise it falls back to the `gz service` CLI automatically.
"""

import argparse
import bisect
import csv
from dataclasses import dataclass
import json
import math
import shutil
import socket
import subprocess
import time

SCHEMA_VERSION = 2
MAX_PACKET_BYTES = 1400
COMMAND_HOLD = "H"
COMMAND_PROCEED = "P"
MAX_COMMAND_TTL_SEC = 5.0


# ---------------------------------------------------------------- path
def load_csv(path):
    pts = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = "theta" if "theta" in row else "yaw"
            pts.append((float(row["x"]), float(row["y"]), float(row[key])))
    return pts


def cumulative(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(
            pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return cum


def curvature_at_arc(pts, cum, arc, loop, span=0.25):
    """Local path curvature [1/m] as |dtheta| / ds over a short span.

    Used to mirror the follower's curve slowdown. A span rather than a single
    segment keeps the estimate from being dominated by waypoint quantisation
    on a 5 cm path.
    """
    total = cum[-1]
    if total <= 0.0:
        return 0.0
    a0, a1 = arc, arc + span
    if loop:
        a0 %= total
        a1 %= total
    else:
        a0 = min(max(a0, 0.0), total)
        a1 = min(max(a1, 0.0), total)
    _, _, th0 = pose_at_arc(pts, cum, a0, loop)
    _, _, th1 = pose_at_arc(pts, cum, a1, loop)
    dtheta = abs(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
    return dtheta / span


def pose_at_arc(pts, cum, arc, loop):
    total = cum[-1]
    if loop and total > 0:
        arc %= total
    else:
        arc = min(max(arc, 0.0), total)
    hi = bisect.bisect_left(cum, arc)
    if hi <= 0:
        return pts[0]
    if hi >= len(pts):
        return pts[-1]

    lo = hi - 1
    segment = cum[hi] - cum[lo]
    alpha = 0.0 if segment <= 1e-12 else (arc - cum[lo]) / segment
    x = pts[lo][0] + alpha * (pts[hi][0] - pts[lo][0])
    y = pts[lo][1] + alpha * (pts[hi][1] - pts[lo][1])
    # Interpolate the shortest angular distance, including across +/-pi.
    dyaw = math.atan2(
        math.sin(pts[hi][2] - pts[lo][2]),
        math.cos(pts[hi][2] - pts[lo][2]),
    )
    yaw = math.atan2(
        math.sin(pts[lo][2] + alpha * dyaw),
        math.cos(pts[lo][2] + alpha * dyaw),
    )
    return x, y, yaw


def configured_pause_trigger(enabled, pause_index, pause_arc, cumulative_arc):
    """Resolve optional scripted fault injection, never normal traffic flow."""
    if not enabled:
        return None
    if pause_index is not None:
        return cumulative_arc[pause_index]
    return pause_arc


# ---------------------------------------------------------------- v2v
def pack(vehicle_id, seq, stamp, x, y, yaw, v, hdt, predicted,
         blocked=None, blocked_distance=None, detour_intent=False):
    predicted = [(round(a, 3), round(b, 3), round(c, 3))
                 for a, b, c in predicted]
    while True:
        msg = {"s": SCHEMA_VERSION, "id": vehicle_id, "q": seq,
               "t": round(stamp, 3), "loc": 1,
               "x": round(x, 3), "y": round(y, 3), "th": round(yaw, 3),
               "v": round(v, 3), "ms": "M" if v > 0 else "S",
               "hdt": round(hdt, 3),
               "p": [c for pt in predicted for c in pt]}
        if blocked is not None:
            msg["bl"] = 1 if blocked else 0
            msg["di"] = 1 if detour_intent else 0
            if blocked_distance is not None and math.isfinite(
                blocked_distance
            ):
                msg["bd"] = round(float(blocked_distance), 3)
        data = json.dumps(msg, separators=(",", ":")).encode()
        if len(data) <= MAX_PACKET_BYTES or not predicted:
            return data
        predicted = predicted[: max(1, len(predicted) - 4)]


# ------------------------------------------------------- obstacle sensing
class ObstacleField:
    """Static obstacles the simulated ROSbot can be blocked by.

    The prop is teleported and carries no LiDAR, so its front-lane distance
    is computed geometrically from the same `obstacles.yaml` the world was
    generated from. The lane box matches the physical follower exactly
    (`x in [0, lane_max_lookahead]`, `|y| <= lane_width / 2`), so "blocked"
    here means what it means on hardware.
    """

    def __init__(self, obstacles, lane_half_width=0.20, lookahead=2.5):
        self.obstacles = list(obstacles)
        self.lane_half_width = float(lane_half_width)
        self.lookahead = float(lookahead)

    @classmethod
    def from_yaml(cls, path, **kwargs):
        if not path:
            return cls([], **kwargs)
        entries = []
        try:
            import yaml
            with open(path) as fh:
                loaded = yaml.safe_load(fh) or []
        except Exception as e:
            print(f"[sim-rosbot] no obstacle file ({path}: {e}); "
                  "reporting lane CLEAR")
            return cls([], **kwargs)
        for item in loaded:
            try:
                entries.append((
                    float(item["x"]), float(item["y"]),
                    float(item.get("sx", 0.2)), float(item.get("sy", 0.2)),
                    str(item.get("name", "obstacle")),
                ))
            except (TypeError, KeyError, ValueError):
                continue
        print(f"[sim-rosbot] {len(entries)} obstacle(s) loaded from {path}")
        return cls(entries, **kwargs)

    def distance_ahead(self, x, y, yaw):
        """Nearest forward distance to an obstacle surface in our lane."""
        closest = float("inf")
        for ox, oy, sx, sy, _name in self.obstacles:
            dx = ox - x
            dy = oy - y
            fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
            lat = -math.sin(yaw) * dx + math.cos(yaw) * dy
            # Half-extent of the box, approximated by its larger half-side so
            # a rotated prop is never reported further away than it is.
            reach = 0.5 * max(sx, sy)
            if fwd <= 0.0 or abs(lat) > self.lane_half_width + reach:
                continue
            surface = fwd - reach
            if 0.0 <= surface < self.lookahead:
                closest = min(closest, surface)
        return closest


class HoldListener:
    """QCar -> ROSbot command receiver for the simulated prop.

    Mirrors `rosbot_v2v_gate` on hardware: a hold is a bounded lease, not a
    latch, so a dead link frees the prop after ttl instead of stranding it.
    """

    def __init__(self, port, vehicle_id, enabled=True):
        self.vehicle_id = vehicle_id
        self.hold_until = None
        self.last_seq = -1
        self.rx = 0
        self.sock = None
        if not enabled:
            return
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", int(port)))
            self.sock.setblocking(False)
            print(f"[sim-rosbot] listening for QCar commands on udp/{port}")
        except OSError as e:
            self.sock = None
            print(f"[sim-rosbot] command listener unavailable ({e}); "
                  "the prop will NOT be sequenced by QCar")

    def poll(self):
        """Drain pending datagrams; returns True while a hold is in force."""
        if self.sock is not None:
            while True:
                try:
                    data, _addr = self.sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
                try:
                    msg = json.loads(data.decode())
                    if msg.get("to") != self.vehicle_id:
                        continue
                    word = msg.get("c")
                    if word not in (COMMAND_HOLD, COMMAND_PROCEED):
                        continue
                    seq = int(msg.get("q", -1))
                    ttl = float(msg.get("ttl", 0.0))
                    if not 0.0 < ttl <= MAX_COMMAND_TTL_SEC:
                        continue
                except (ValueError, TypeError, AttributeError):
                    continue
                if seq <= self.last_seq and self.last_seq - seq < 1000:
                    continue
                self.last_seq = seq
                self.rx += 1
                self.hold_until = (
                    time.monotonic() + ttl if word == COMMAND_HOLD else None
                )

        if self.hold_until is None:
            return False
        if time.monotonic() >= self.hold_until:
            self.hold_until = None
            return False
        return True


# ---------------------------------------------------------------- gazebo
class PoseSetter:
    """SetEntityPose via the ros_gz service bridge, gz CLI as fallback."""

    def __init__(self, world, entity):
        self.world = world
        self.entity = entity
        self.mode = None
        self.node = None
        self.client = None
        self.gz_bin = shutil.which("gz")
        self._pending = []
        self._pose_future = None
        self._spin_stop = False

        try:
            import rclpy
            from rclpy.parameter import Parameter
            from ros_gz_interfaces.srv import SetEntityPose  # noqa: F401
            rclpy.init()
            from rclpy.node import Node as RclNode
            self.node = RclNode(
                "sim_rosbot_driver",
                parameter_overrides=[
                    Parameter("use_sim_time", Parameter.Type.BOOL, True)
                ],
            )
            self.client = self.node.create_client(
                SetEntityPose, f"/world/{world}/set_pose"
            )
            if self.client.wait_for_service(timeout_sec=3.0):
                self.mode = "bridge"
                self._SetEntityPose = SetEntityPose
                print("[sim-rosbot] pose via ros_gz service bridge")
            else:
                print("[sim-rosbot] bridge service not found "
                      "(start parameter_bridge for /world/"
                      f"{world}/set_pose) — falling back to gz CLI")
        except Exception as e:  # rclpy or ros_gz_interfaces missing
            print(f"[sim-rosbot] no ROS pose bridge ({e})")

        # One executor services /clock, odometry and service responses.  It is
        # needed even in CLI pose mode because vehicle motion is based on
        # Gazebo time, never on host wall time.
        if self.node is not None:
            import threading
            threading.Thread(target=self._spin_loop, daemon=True).start()

        if self.mode is None:
            if self.gz_bin is None:
                raise RuntimeError(
                    "Neither ros_gz bridge nor `gz` CLI available — "
                    "cannot move the model."
                )
            self.mode = "cli"
            print("[sim-rosbot] pose via gz CLI (slower)")

    def _spin_loop(self):
        import rclpy
        while rclpy.ok() and not self._spin_stop:
            try:
                rclpy.spin_once(self.node, timeout_sec=0.05)
            except Exception:
                break

    def set_pose(self, x, y, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        if self.mode == "bridge":
            # Do not accumulate an unbounded queue when Gazebo runs slower
            # than wall time.  The next tick always carries the newest pose.
            if self._pose_future is not None and not self._pose_future.done():
                return
            req = self._SetEntityPose.Request()
            req.entity.name = self.entity
            req.entity.type = 2  # MODEL
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.position.z = 0.11
            req.pose.orientation.z = qz
            req.pose.orientation.w = qw
            # Fire and forget; the spin thread reaps the response.
            self._pose_future = self.client.call_async(req)
        else:
            req = (f'name: "{self.entity}", '
                   f'position: {{x: {x:.4f}, y: {y:.4f}, z: 0.11}}, '
                   f'orientation: {{z: {qz:.6f}, w: {qw:.6f}}}')
            # Fire and forget. A blocking subprocess.run here spawns a ruby
            # process per tick and throttled the whole loop to ~2.6 Hz, which
            # pushed packet age past the receiver's 0.6 s stale threshold and
            # made /v2v/alive flap. Pose updates must never gate the V2V rate.
            self._reap()
            try:
                self._pending.append(subprocess.Popen(
                    [self.gz_bin, "service", "-s",
                     f"/world/{self.world}/set_pose",
                     "--reqtype", "gz.msgs.Pose",
                     "--reptype", "gz.msgs.Boolean",
                     "--timeout", "100", "--req", req],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            except OSError:
                pass

    def sim_time(self):
        if self.node is None:
            raise RuntimeError("ROS node unavailable; cannot read /clock")
        return self.node.get_clock().now().nanoseconds * 1e-9

    def wait_for_clock(self, timeout_sec=20.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            stamp = self.sim_time()
            if stamp > 0.0:
                print(f"[sim-rosbot] Gazebo clock ready at {stamp:.3f} s")
                return stamp
            time.sleep(0.05)
        raise RuntimeError(
            "No Gazebo /clock received. Verify the ros_gz clock bridge and "
            "that the world is running."
        )

    def _reap(self):
        """Drop finished CLI children; cap the backlog if Gazebo is slow."""
        self._pending = [p for p in self._pending if p.poll() is None]
        while len(self._pending) > 5:
            p = self._pending.pop(0)
            try:
                p.kill()
            except OSError:
                pass

    def close(self):
        self._spin_stop = True
        if self.node is not None:
            import rclpy
            try:
                self.node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass


class QCarWatcher:
    """Track QCar pose for the simulated ROSbot's proximity response.

    The physical ROSbot slows from 0.90 m and only halts inside 0.40 m;
    without an equivalent here the teleported prop drives through the QCar.
    Uses the bridged /model/qcar2/odometry when rclpy is available (the QCar
    spawns at the map origin, so odom ~= map).
    """

    def __init__(
        self,
        world,
        node,
        model="qcar2",
        lane_half_width=0.20,
        qcar_body_radius=0.20,
    ):
        """`node` must be an already-spinning rclpy node (PoseSetter's).
        Adding the subscription there keeps everything on one spin loop.

        The two geometry defaults mirror the physical follower rather than
        being chosen here.  ``lane_half_width`` is
        ``trajectory_follower_node._lane_width / 2`` (0.40 / 2): that node's
        front region is ``y in [-0.20, +0.20]``, so anything further out is
        in the other lane and does not slow the robot at all.
        ``qcar_body_radius`` converts this centre-to-centre measurement into
        the surface range a LiDAR actually returns, which is what
        ``_closest_in_region`` reports and what the 0.90/0.40 m thresholds
        were tuned against.
        """
        self.world = world
        self.model = model
        self.lane_half_width = float(lane_half_width)
        self.qcar_body_radius = float(qcar_body_radius)
        self.x = None
        self.y = None
        self.last_update_wall = None
        if node is None:
            print("[sim-rosbot] no ROS node (gz CLI mode); prop will NOT "
                  "stop for the QCar")
            return
        try:
            from nav_msgs.msg import Odometry
            node.create_subscription(
                Odometry, f"/model/{model}/odometry", self._cb, 10
            )
            print("[sim-rosbot] watching QCar via /model/%s/odometry" % model)
        except Exception as e:
            print(f"[sim-rosbot] QCar watch unavailable ({e}); "
                  "prop will NOT stop for the QCar")

    def _cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.last_update_wall = time.monotonic()

    def fresh(self, timeout_sec):
        return bool(
            self.last_update_wall is not None
            and 0.0 <= time.monotonic() - self.last_update_wall < timeout_sec
        )

    def distance_ahead(self, x, y, yaw):
        """Surface range to the QCar ahead, or inf if not ahead/unknown.

        Mirrors ``trajectory_follower_node._front_dist``: a rectangular front
        region one lane wide, reporting the distance to the nearest *surface*.
        A QCar out in the passing lane is outside the region and produces no
        slowdown at all, exactly as on hardware.
        """
        if self.x is None:
            return float("inf")
        dx = self.x - x
        dy = self.y - y
        fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
        lat = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if fwd <= 0.0 or abs(lat) > self.lane_half_width:
            return float("inf")
        return max(0.0, math.hypot(dx, dy) - self.qcar_body_radius)


@dataclass(frozen=True)
class ProximityDecision:
    """Longitudinal command produced by the ROSbot proximity governor."""

    speed_scale: float
    hard_stop: bool
    state: str


class FollowerProximityGovernor:
    """Mirror the physical ROSbot's progressive obstacle speed policy.

    The current ``opta2`` trajectory follower cruises above 0.90 m, ramps its
    speed linearly between 0.90 m and 0.40 m, and uses a hard-stop latch below
    0.40 m until the obstacle is beyond 0.50 m.  This governor deliberately
    changes longitudinal speed only: it does not consume a V2V horizon and it
    never requests a passing lane, preserving the asymmetric experiment.

    Loss of observation while a hard stop is latched is fail-safe.  A stale
    observation during normal driving cannot manufacture an obstacle and
    therefore leaves the requested scale at one.
    """

    def __init__(self, stop_distance, slow_distance, clear_distance):
        self.stop_distance = float(stop_distance)
        self.slow_distance = float(slow_distance)
        self.clear_distance = float(clear_distance)
        if self.stop_distance <= 0.0:
            raise ValueError("stop_distance must be positive")
        if self.slow_distance <= self.stop_distance:
            raise ValueError("slow_distance must exceed stop_distance")
        if self.clear_distance <= self.stop_distance:
            raise ValueError("clear_distance must exceed stop_distance")
        if self.clear_distance >= self.slow_distance:
            raise ValueError("clear_distance must be below slow_distance")
        self.hard_stop = False

    def update(self, distance_ahead, observation_fresh=True):
        if not observation_fresh:
            if self.hard_stop:
                return ProximityDecision(0.0, True, "QCAR_STALE_HOLD")
            return ProximityDecision(1.0, False, "QCAR_UNKNOWN_CRUISE")

        distance = float(distance_ahead)
        if math.isnan(distance):
            if self.hard_stop:
                return ProximityDecision(0.0, True, "QCAR_STALE_HOLD")
            return ProximityDecision(1.0, False, "QCAR_UNKNOWN_CRUISE")

        if self.hard_stop:
            # Match the physical controller: resume only when comfortably
            # beyond the clear-distance threshold, not while oscillating on
            # its boundary.
            if distance > self.clear_distance:
                self.hard_stop = False
            else:
                return ProximityDecision(0.0, True, "HOLD_QCAR_AHEAD")

        if distance < self.stop_distance:
            self.hard_stop = True
            return ProximityDecision(0.0, True, "HOLD_QCAR_AHEAD")

        if distance < self.slow_distance:
            band = self.slow_distance - self.stop_distance
            scale = (distance - self.stop_distance) / band
            return ProximityDecision(
                min(1.0, max(0.0, scale)), False, "SLOW_QCAR_AHEAD"
            )

        return ProximityDecision(1.0, False, "CRUISE")


class FollowerStopLatch:
    """Compatibility wrapper for older focused tests and imports.

    New simulation code should use :class:`FollowerProximityGovernor` so the
    slowdown band is represented instead of reducing the behavior to a bool.
    """

    def __init__(self, stop_distance, clear_distance):
        self.stop_distance = float(stop_distance)
        self.clear_distance = float(clear_distance)
        if self.stop_distance <= 0.0:
            raise ValueError("stop_distance must be positive")
        if self.clear_distance <= self.stop_distance:
            raise ValueError("clear_distance must exceed stop_distance")
        self._blocked = False

    @property
    def blocked(self):
        return self._blocked

    def update(self, distance_ahead, observation_fresh=True):
        if not observation_fresh:
            return self._blocked
        distance = float(distance_ahead)
        if math.isnan(distance):
            return self._blocked
        if self._blocked:
            if distance >= self.clear_distance:
                self._blocked = False
        elif distance <= self.stop_distance:
            self._blocked = True
        return self._blocked


class BehaviorReporter:
    """Publish and record the ROSbot follower's verifiable state/command."""

    def __init__(self, node, log_path):
        self.state_pub = None
        self.command_pub = None
        self.speed_scale_pub = None
        self.hard_stop_pub = None
        self.last_summary = None
        self.log_fh = open(log_path, "w", newline="") if log_path else None
        self.log_writer = csv.writer(self.log_fh) if self.log_fh else None
        if self.log_writer:
            self.log_writer.writerow([
                "sim_time",
                "state",
                "base_speed_mps",
                "command_speed_mps",
                "proximity_speed_scale",
                "hard_stop_requested",
                "qcar_distance_ahead_m",
                "qcar_observation_fresh",
                "scheduled_pause",
                "path_finished",
                "overtake_enabled",
            ])
            self.log_fh.flush()
            print(f"[sim-rosbot] behavior CSV: {log_path}")

        if node is not None:
            try:
                from std_msgs.msg import Bool, Float32, String
                self._String = String
                self._Bool = Bool
                self._Float32 = Float32
                self.state_pub = node.create_publisher(
                    String, "/v2v/rosbot/follower_state", 10
                )
                self.command_pub = node.create_publisher(
                    String, "/v2v/rosbot/command", 10
                )
                self.speed_scale_pub = node.create_publisher(
                    Float32, "/v2v/rosbot/proximity_speed_scale", 10
                )
                self.hard_stop_pub = node.create_publisher(
                    Bool, "/v2v/rosbot/hard_stop", 10
                )
            except Exception as exc:
                print(f"[sim-rosbot] behavior topics unavailable ({exc})")

    def publish(
        self,
        *,
        sim_time,
        state,
        base_speed,
        command_speed,
        speed_scale,
        hard_stop,
        qcar_distance,
        qcar_fresh,
        scheduled_pause,
        finished,
    ):
        finite_distance = math.isfinite(qcar_distance)
        state_data = {
            "role": "ROSBOT_NON_OVERTAKING_FOLLOWER",
            "state": str(state),
            "qcar_observation_fresh": bool(qcar_fresh),
            "qcar_distance_ahead_m": (
                round(float(qcar_distance), 3) if finite_distance else None
            ),
            "proximity_speed_scale": round(float(speed_scale), 3),
            "hard_stop_requested": bool(hard_stop),
            "v2v_horizon_optimization": False,
            "overtake_enabled": False,
        }
        command_data = {
            "base_speed_mps": round(float(base_speed), 3),
            "speed_mps": round(float(command_speed), 3),
            "proximity_speed_scale": round(float(speed_scale), 3),
            "lateral_offset_m": 0.0,
            "hard_stop_requested": bool(hard_stop),
            "stop_requested": bool(hard_stop) or float(command_speed) <= 0.0,
            "state": str(state),
        }
        state_json = json.dumps(state_data, separators=(",", ":"))
        command_json = json.dumps(command_data, separators=(",", ":"))
        if self.state_pub is not None:
            self.state_pub.publish(self._String(data=state_json))
        if self.command_pub is not None:
            self.command_pub.publish(self._String(data=command_json))
        if self.speed_scale_pub is not None:
            self.speed_scale_pub.publish(self._Float32(data=float(speed_scale)))
        if self.hard_stop_pub is not None:
            self.hard_stop_pub.publish(self._Bool(data=bool(hard_stop)))
        if self.log_writer:
            self.log_writer.writerow([
                f"{sim_time:.3f}",
                state,
                f"{base_speed:.3f}",
                f"{command_speed:.3f}",
                f"{speed_scale:.3f}",
                int(bool(hard_stop)),
                f"{qcar_distance:.3f}" if finite_distance else "",
                int(bool(qcar_fresh)),
                int(bool(scheduled_pause)),
                int(bool(finished)),
                0,
            ])
            self.log_fh.flush()

        summary = (
            state,
            round(float(command_speed), 3),
            round(float(speed_scale), 2),
            bool(qcar_fresh),
        )
        if summary != self.last_summary:
            print(f"[sim-rosbot] FOLLOWER_STATE {state_json} CMD {command_json}")
            self.last_summary = summary

    def close(self):
        if self.log_fh is not None:
            self.log_fh.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--world", default="sami_track")
    ap.add_argument("--entity", default="sim_rosbot")
    ap.add_argument("--target", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--vehicle-id", default="rosbot3")
    # Matches the physical follower's PurePursuitConfig: cruise 0.40 m/s,
    # slowed to 0.12 in curves. The ROSbot is the FASTER of the two robots
    # (QCar cruises 0.15), so the common encounter is it catching the QCar
    # from behind, not the other way round.
    ap.add_argument("--speed", type=float, default=0.40)
    ap.add_argument("--curve-speed", type=float, default=0.12)
    ap.add_argument("--curve-threshold", type=float, default=0.8,
                    help="path curvature [1/m] above which curve-speed applies")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--hdt", type=float, default=0.08)
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument(
        "--start-gap-m", type=float, default=0.0,
        help="additional along-path start gap after --start-idx",
    )
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument(
        "--pause-every", type=float, default=0.0,
        help="fault-injection pause interval; requires "
             "--enable-scripted-pauses",
    )
    pause_group = ap.add_mutually_exclusive_group()
    pause_group.add_argument(
        "--pause-at-index", type=int,
        help="pause exactly once on reaching this path waypoint",
    )
    pause_group.add_argument(
        "--pause-at-arc", type=float,
        help="pause exactly once on reaching this along-path distance [m]",
    )
    ap.add_argument(
        "--repeat-pause-every-lap", action="store_true",
        help="repeat --pause-at-index/--pause-at-arc on subsequent laps",
    )
    ap.add_argument("--pause-for", type=float, default=10.0)
    ap.add_argument(
        "--enable-scripted-pauses", action="store_true",
        help="enable deterministic waypoint/time stops for fault-injection; "
             "normal V2V demonstrations keep the slower ROSbot moving",
    )
    ap.add_argument("--no-v2v", action="store_true",
                    help="move the model but send no packets (baseline runs)")
    ap.add_argument("--stop-distance", type=float, default=0.40,
                    help="emergency stop distance used by the physical "
                         "ROSbot follower (0 disables proximity control)")
    ap.add_argument("--slow-distance", type=float, default=0.90,
                    help="start linearly slowing for a QCar ahead")
    ap.add_argument("--clear-distance", type=float, default=0.50,
                    help="release a hard stop only beyond this distance")
    ap.add_argument("--qcar-watch-timeout", type=float, default=1.0,
                    help="maximum wall-time age of QCar odometry [s]")
    ap.add_argument("--behavior-log", default="/tmp/rosbot_v2v_behavior.csv",
                    help="CSV follower evidence path; empty disables")
    ap.add_argument("--obstacles", default="",
                    help="obstacles.yaml the world was generated from; "
                         "enables the blocked/detour-intent report")
    ap.add_argument("--command-port", type=int, default=47101,
                    help="UDP port for QCar hold/proceed commands")
    ap.add_argument("--no-commands", action="store_true",
                    help="ignore QCar commands (uncoordinated baseline)")
    ap.add_argument("--detour-warn-time", type=float, default=0.5,
                    help="seconds blocked before announcing detour intent")
    args = ap.parse_args()

    if args.speed < 0.0 or args.rate <= 0.0 or args.pause_for < 0.0:
        ap.error("speed must be non-negative; rate positive; pause-for non-negative")
    if args.start_gap_m < 0.0:
        ap.error("--start-gap-m must be non-negative")
    if args.qcar_watch_timeout <= 0.0:
        ap.error("--qcar-watch-timeout must be positive")
    if args.stop_distance > 0.0:
        if args.slow_distance <= args.stop_distance:
            ap.error("--slow-distance must exceed --stop-distance")
        if args.clear_distance <= args.stop_distance:
            ap.error("--clear-distance must exceed --stop-distance")
        if args.clear_distance >= args.slow_distance:
            ap.error("--clear-distance must be below --slow-distance")
    if (args.enable_scripted_pauses and args.repeat_pause_every_lap
            and args.no_loop):
        ap.error("--repeat-pause-every-lap requires looping")

    pts = load_csv(args.csv)
    if len(pts) < 2:
        ap.error("trajectory must contain at least two points")
    cum = cumulative(pts)
    total = cum[-1]
    loop = not args.no_loop
    if not 0 <= args.start_idx < len(pts):
        ap.error(f"--start-idx must be in [0, {len(pts) - 1}]")
    if args.pause_at_index is not None and not (
        0 <= args.pause_at_index < len(pts)
    ):
        ap.error(f"--pause-at-index must be in [0, {len(pts) - 1}]")
    if args.pause_at_arc is not None and not 0.0 <= args.pause_at_arc <= total:
        ap.error(f"--pause-at-arc must be in [0, {total:.3f}]")

    print(f"[sim-rosbot] {len(pts)} pts, {cum[-1]:.1f} m | "
          f"v2v={'OFF' if args.no_v2v else f'-> {args.target}:{args.port}'}")

    setter = PoseSetter(args.world, args.entity)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # The REAL ROSbot follower progressively slows for obstacles and reserves
    # its 0.40 m halt for emergencies (trajectory_follower_node
    # `_check_obstacle`). Without this the prop drives straight through QCar
    # and shows a collision that cannot happen on hardware.
    watcher = (
        QCarWatcher(args.world, setter.node) if args.stop_distance > 0 else None
    )
    follower = (
        FollowerProximityGovernor(
            args.stop_distance, args.slow_distance, args.clear_distance
        )
        if watcher is not None else None
    )
    reporter = BehaviorReporter(setter.node, args.behavior_log)
    proximity = ProximityDecision(1.0, False, "CRUISE")

    obstacles = ObstacleField.from_yaml(args.obstacles)
    commands = HoldListener(
        args.command_port, args.vehicle_id, enabled=not args.no_commands
    )
    blocked_since = None
    was_held = False

    arc = cum[args.start_idx] + args.start_gap_m
    if not loop:
        arc = min(arc, total)

    pause_trigger = configured_pause_trigger(
        args.enable_scripted_pauses,
        args.pause_at_index,
        args.pause_at_arc,
        cum,
    )
    if not args.enable_scripted_pauses and (
        args.pause_every > 0.0
        or args.pause_at_index is not None
        or args.pause_at_arc is not None
    ):
        print(
            "[sim-rosbot] scripted pause arguments ignored: the normal "
            "scenario uses a continuously moving slower ROSbot; add "
            "--enable-scripted-pauses only for explicit fault injection"
        )
    if pause_trigger is not None and pause_trigger < arc:
        if loop:
            pause_trigger += total
        else:
            ap.error("one-shot pause point is behind the non-looping start")

    seq = 0
    wall_period = 1.0 / args.rate
    drive_timer = 0.0
    paused_until = None
    last_v = 0.0

    try:
        last_sim_time = setter.wait_for_clock()
        print(
            f"[sim-rosbot] start arc={arc:.2f} m "
            + (
                f"scheduled pause arc={pause_trigger:.2f} m"
                if pause_trigger is not None
                else "no fixed pause"
            )
        )
        while True:
            sim_now = setter.sim_time()
            if sim_now < last_sim_time:
                # A world reset invalidates elapsed time but must not create a
                # huge negative or positive path jump.
                print("[sim-rosbot] Gazebo clock reset; holding this tick")
                last_v = 0.0
                last_sim_time = sim_now
            sim_dt = max(0.0, sim_now - last_sim_time)
            last_sim_time = sim_now

            # Integrate the speed that was commanded during the elapsed
            # Gazebo interval.  Slow real-time factor no longer speeds this
            # prop up relative to QCar physics.
            arc += last_v * sim_dt
            if not loop:
                arc = min(arc, total)
            if last_v > 0.0:
                drive_timer += sim_dt

            paused = paused_until is not None and sim_now < paused_until
            if paused_until is not None and not paused:
                paused_until = None
                print(f"[sim-rosbot] scheduled pause complete at {sim_now:.2f}s")

            # Deterministic scenario trigger: tied to path progress, not to
            # process startup phase or host wall time.
            if (
                not paused
                and pause_trigger is not None
                and arc >= pause_trigger
            ):
                paused_until = sim_now + args.pause_for
                paused = True
                print(
                    f"[sim-rosbot] pausing {args.pause_for:.1f} Gazebo s "
                    f"at arc {arc:.2f} m"
                )
                if args.repeat_pause_every_lap:
                    pause_trigger += total
                else:
                    pause_trigger = None

            # Optional periodic scenario also uses accumulated Gazebo driving
            # time, so its phase is reproducible at every real-time factor.
            if (
                not paused
                and not proximity.hard_stop
                and args.enable_scripted_pauses
                and args.pause_every > 0.0
                and drive_timer >= args.pause_every
            ):
                drive_timer = 0.0
                paused_until = sim_now + args.pause_for
                paused = True
                print(
                    f"[sim-rosbot] periodic pause {args.pause_for:.1f} "
                    f"Gazebo s at arc {arc:.2f} m"
                )

            x, y, yaw = pose_at_arc(pts, cum, arc, loop)

            # Progressive obstacle response, mirroring the real follower:
            # cruise -> linear slowdown -> emergency hold with hysteresis.
            qcar_distance = float("inf")
            qcar_fresh = False
            if watcher is not None:
                qcar_fresh = watcher.fresh(args.qcar_watch_timeout)
                if qcar_fresh:
                    qcar_distance = watcher.distance_ahead(x, y, yaw)
                was_hard_stop = proximity.hard_stop
                proximity = follower.update(qcar_distance, qcar_fresh)
                if proximity.hard_stop and not was_hard_stop:
                    print(f"[sim-rosbot] QCar {qcar_distance:.2f} m ahead; "
                          "emergency hold (physical follower behavior)")
                elif was_hard_stop and not proximity.hard_stop:
                    print(f"[sim-rosbot] QCar clear ({qcar_distance:.2f} m); "
                          "resuming")

            # Our own obstacle situation -- what we tell QCar about.  The
            # default generated world has no obstacle props, so an empty
            # obstacle field is positive knowledge that this simulated lane is
            # clear, not an old-schema/unknown report.  Hardware broadcasters
            # that cannot sense/report blockage can still omit the field.
            if obstacles.obstacles:
                obstacle_dist = obstacles.distance_ahead(x, y, yaw)
                blocked = obstacle_dist < args.slow_distance
                if not blocked:
                    blocked_since = None
                elif blocked_since is None:
                    blocked_since = sim_now
                detour_intent = bool(
                    blocked
                    and blocked_since is not None
                    and (sim_now - blocked_since) >= args.detour_warn_time
                )
                blocked_distance = (
                    obstacle_dist if math.isfinite(obstacle_dist) else None
                )
            else:
                blocked, blocked_distance, detour_intent = False, None, False
                obstacle_dist = float("inf")

            # QCar's instruction. It outranks our own cruise because only
            # QCar can see the conflict: our detour check cannot detect a
            # vehicle overtaking us from behind.
            held = commands.poll()
            if held and not was_held:
                print("[sim-rosbot] QCar HOLD — waiting for it to pass")
            elif was_held and not held:
                print("[sim-rosbot] QCar released — resuming")
            was_held = held

            finished = not loop and arc >= total
            # A real obstacle stops us too: without this the prop drives
            # through the very box it is telling QCar about.
            obstacle_stop = obstacle_dist < args.stop_distance
            # Curve slowdown, mirroring PurePursuitConfig.curve_speed.
            kappa = curvature_at_arc(pts, cum, arc, loop)
            cruise = (
                args.curve_speed
                if kappa > args.curve_threshold
                else args.speed
            )
            v = (
                0.0
                if (paused or proximity.hard_stop or finished or held
                    or obstacle_stop)
                else cruise * proximity.speed_scale
            )

            if finished:
                follower_state = "PATH_FINISHED"
            elif held:
                follower_state = "V2V_HOLD_FOR_QCAR"
            elif obstacle_stop:
                follower_state = "HOLD_OBSTACLE_AHEAD"
            elif detour_intent:
                follower_state = "DETOUR_INTENT"
            elif paused:
                follower_state = "SCHEDULED_STOP"
            else:
                follower_state = proximity.state

            reporter.publish(
                sim_time=sim_now,
                state=follower_state,
                base_speed=args.speed,
                command_speed=v,
                speed_scale=proximity.speed_scale,
                hard_stop=proximity.hard_stop,
                qcar_distance=qcar_distance,
                qcar_fresh=qcar_fresh,
                scheduled_pause=paused,
                finished=finished,
            )

            setter.set_pose(x, y, yaw)

            if not args.no_v2v:
                predicted = [
                    pose_at_arc(pts, cum, arc + v * args.hdt * k, loop)
                    for k in range(args.horizon)
                ]
                sock.sendto(
                    pack(args.vehicle_id, seq, sim_now, x, y, yaw, v,
                         args.hdt, predicted,
                         blocked=blocked, blocked_distance=blocked_distance,
                         detour_intent=detour_intent),
                    (args.target, args.port),
                )
            seq += 1
            last_v = v
            if seq % 50 == 0:
                print(f"[sim-rosbot] seq={seq} arc={arc:.1f} "
                      f"pos=({x:.2f},{y:.2f}) v={v}")
            time.sleep(wall_period)
    except KeyboardInterrupt:
        print(f"\n[sim-rosbot] stopped after {seq} ticks")
    finally:
        reporter.close()
        setter.close()


if __name__ == "__main__":
    main()
