#!/usr/bin/env python3
"""V2V broadcaster — runs on the ROSbot. The ROSbot's controller is untouched.

Broadcasts the ROSbot's current pose, speed, and predicted future trajectory
over UDP to the QCar at 10 Hz. The prediction rolls the robot forward along
its own recorded reference path at the current measured speed — unlike the
constant-velocity models used for uncooperative traffic, the ROSbot KNOWS its
future: it is following a recorded trajectory. When the follower pauses
(waypoint pause, obstacle hold), the measured speed collapses to zero and the
prediction automatically becomes "stationary" — no coupling to the follower
node is needed.

Self-contained on purpose: no imports from rosbot_lane, so this single file
can be copied to the robot next to any checkout and run directly:

    # Terminal on the ROSbot (after tf_relay + SLAM/AMCL are up):
    python3 rosbot_v2v_broadcaster.py --ros-args \
        -p target_ip:=192.168.0.53 \
        -p trajectory_csv:=/path/to/smoothed_trajectory.csv

Why UDP, not DDS: this lab's Wi-Fi has twice crashed Fast DDS discovery when
mixed ROS 2 distros shared a domain (every node dies with std::bad_alloc).
Both robots keep their ROS graphs private (ROS_LOCALHOST_ONLY=1 or a unique
ROS_DOMAIN_ID) and V2V rides this raw socket instead. UDP loss is harmless:
every datagram is a complete state refresh.

If the trajectory CSV is missing, falls back to a constant-velocity
prediction. If TF is unavailable (SLAM/AMCL down), sends heartbeat packets
marked not-localized — the QCar treats those as "no data", never as "clear".
"""

import csv
import json
import math
import os
import socket
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

SCHEMA_VERSION = 2
MAX_PACKET_BYTES = 1400


class RosbotV2VBroadcaster(Node):

    def __init__(self):
        super().__init__("rosbot_v2v_broadcaster")

        self.declare_parameter("target_ip", "192.168.0.53")
        self.declare_parameter("target_port", 47100)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("vehicle_id", "rosbot3")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter(
            "trajectory_csv",
            os.path.expanduser(
                "~/Documents/rosbot_ws/src/rosbot_lane/config/"
                "smoothed_trajectory.csv"
            ),
        )
        self.declare_parameter("horizon_steps", 26)   # incl. stage 0
        self.declare_parameter("horizon_dt", 0.08)    # matches QCar MPC dt
        self.declare_parameter("speed_alpha", 0.4)    # speed EMA blend
        self.declare_parameter("speed_deadband", 0.02)

        # ---- Obstacle report (schema 2) ----
        # Every value below mirrors trajectory_follower_node so this node
        # reports what that node is ABOUT to do without importing it or
        # modifying it. They must be kept in step with it by hand; that is
        # the price of the follower staying untouched.
        self.declare_parameter("scan_topic", "/rosbot3/scan")
        self.declare_parameter("lane_width", 0.40)          # _lane_width
        self.declare_parameter("lane_max_lookahead", 2.5)   # _lane_max_lookahead
        self.declare_parameter("obstacle_slow_distance", 0.90)
        self.declare_parameter("detour_lateral_offset", 0.33)
        self.declare_parameter("detour_check_forward", 2.0)
        # The follower commits to its detour after 1.5 s of slowdown
        # (_overtake_trigger_time). Announcing intent at that same instant
        # would give QCar zero time to react, so warn earlier: QCar needs the
        # warning BEFORE the swerve, not with it.
        self.declare_parameter("detour_warn_time", 0.5)

        gp = lambda name: self.get_parameter(name).value
        self.target = (str(gp("target_ip")), int(gp("target_port")))
        self.vehicle_id = str(gp("vehicle_id"))
        self.map_frame = str(gp("map_frame"))
        self.base_frame = str(gp("base_frame"))
        self.horizon_steps = int(gp("horizon_steps"))
        self.horizon_dt = float(gp("horizon_dt"))
        self.speed_alpha = float(gp("speed_alpha"))
        self.speed_deadband = float(gp("speed_deadband"))
        rate_hz = float(gp("rate_hz"))

        self.lane_half_width = float(gp("lane_width")) / 2.0
        self.lane_max_lookahead = float(gp("lane_max_lookahead"))
        self.obstacle_slow_distance = float(gp("obstacle_slow_distance"))
        self.detour_lateral_offset = float(gp("detour_lateral_offset"))
        self.detour_check_forward = float(gp("detour_check_forward"))
        self.detour_warn_time = float(gp("detour_warn_time"))

        self.latest_scan = None
        self.front_dist = float("inf")
        self.blocked_since = None       # monotonic time blockage began
        self.last_report = (None, None, False)   # cached for diagnostics

        # Recorded reference path (x, y, theta per row). Optional.
        self.path_xy = []
        self.path_yaw = []
        self.cum_dist = [0.0]
        csv_file = str(gp("trajectory_csv"))
        try:
            with open(csv_file) as fh:
                for row in csv.DictReader(fh):
                    self.path_xy.append((float(row["x"]), float(row["y"])))
                    self.path_yaw.append(float(row["theta"]))
            for i in range(1, len(self.path_xy)):
                step = math.hypot(
                    self.path_xy[i][0] - self.path_xy[i - 1][0],
                    self.path_xy[i][1] - self.path_xy[i - 1][1],
                )
                self.cum_dist.append(self.cum_dist[-1] + step)
            self.get_logger().info(
                f"Path loaded: {csv_file} ({len(self.path_xy)} pts, "
                f"{self.cum_dist[-1]:.1f} m)"
            )
        except (OSError, KeyError, ValueError) as e:
            self.path_xy = []
            self.get_logger().warn(
                f"No path ({csv_file}: {e}) — using constant-velocity "
                "prediction."
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            LaserScan, str(gp("scan_topic")), self._scan_cb, 10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

        self.seq = 0
        self.tx_count = 0
        self.tx_errors = 0
        self.prev_pose = None            # (x, y, t_monotonic)
        self.speed_ema = 0.0

        # Local debug topics (loopback DDS only — nothing leaves the robot).
        self.debug_pred_pub = self.create_publisher(Path, "/v2v/tx_predicted", 10)
        self.debug_pose_pub = self.create_publisher(
            PoseStamped, "/v2v/tx_pose", 10
        )
        self.stats_pub = self.create_publisher(String, "/v2v/tx_stats", 10)

        self.create_timer(1.0 / rate_hz, self._tick)
        self.create_timer(1.0, self._publish_stats)

        self.get_logger().info(
            f"V2V broadcaster up: -> udp://{self.target[0]}:{self.target[1]} "
            f"at {rate_hz:.0f} Hz as '{self.vehicle_id}'"
        )

    # ------------------------------------------------------------------
    def _lookup_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return t.transform.translation.x, t.transform.translation.y, yaw

    # ------------------------------------------------------------------
    # Obstacle report. Deliberately a re-implementation of
    # trajectory_follower_node._closest_in_region rather than an import: that
    # node must stay untouched, and this file must stay copy-and-run.
    @staticmethod
    def _closest_in_region(msg, x_min, x_max, y_min, y_max):
        """Nearest forward distance inside a robot-frame rectangle, else inf.

        The LiDAR is mounted backwards, hence the negated projections — same
        convention as the follower.
        """
        closest = float("inf")
        for i, dist in enumerate(msg.ranges):
            if dist < msg.range_min or dist > msg.range_max:
                continue
            if math.isnan(dist) or math.isinf(dist):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            x = -dist * math.cos(angle)
            y = -dist * math.sin(angle)
            if x_min < x < x_max and y_min < y < y_max:
                if x < closest:
                    closest = x
        return closest

    def _scan_cb(self, msg):
        self.latest_scan = msg
        self.front_dist = self._closest_in_region(
            msg,
            0.0, self.lane_max_lookahead,
            -self.lane_half_width, self.lane_half_width,
        )

    def _detour_lane_is_feasible(self):
        """Mirror of the follower's _is_overtake_feasible."""
        if self.latest_scan is None:
            return False
        offset = self.detour_lateral_offset
        closest = self._closest_in_region(
            self.latest_scan,
            -0.2, self.detour_check_forward,
            offset - self.lane_half_width, offset + self.lane_half_width,
        )
        return closest == float("inf")

    def _obstacle_report(self):
        """(blocked, distance, detour_intent) for this tick.

        Without a scan we report *unknown* (None) rather than clear: the QCar
        must never read a missing sensor as an empty lane.
        """
        if self.latest_scan is None:
            self.blocked_since = None
            return None, None, False

        now = time.monotonic()
        blocked = self.front_dist < self.obstacle_slow_distance
        if not blocked:
            self.blocked_since = None
            return False, None, False

        if self.blocked_since is None:
            self.blocked_since = now
        intent = (
            (now - self.blocked_since) >= self.detour_warn_time
            and self._detour_lane_is_feasible()
        )
        return True, self.front_dist, intent

    def _update_speed(self, x, y):
        now = time.monotonic()
        if self.prev_pose is None:
            self.prev_pose = (x, y, now)
            return 0.0
        px, py, pt = self.prev_pose
        dt = now - pt
        self.prev_pose = (x, y, now)
        if dt <= 1e-3:
            return self.speed_ema
        raw = math.hypot(x - px, y - py) / dt
        self.speed_ema = (
            (1.0 - self.speed_alpha) * self.speed_ema + self.speed_alpha * raw
        )
        if self.speed_ema < self.speed_deadband:
            return 0.0
        return self.speed_ema

    def _closest_path_idx(self, x, y):
        best_i, best_d = 0, float("inf")
        for i, (px, py) in enumerate(self.path_xy):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def _pose_at_arc(self, start_idx, arc):
        """Waypoint ~arc meters further along the path from start_idx.
        Holds the final waypoint once the path runs out (conservative:
        predicts the robot parked at the path end)."""
        target = self.cum_dist[start_idx] + arc
        i = start_idx
        while i < len(self.path_xy) - 1 and self.cum_dist[i] < target:
            i += 1
        return self.path_xy[i][0], self.path_xy[i][1], self.path_yaw[i]

    def _predict(self, x, y, yaw, v):
        """Predicted (x, y, yaw) per stage; stage 0 = actual current pose."""
        n = self.horizon_steps
        if v <= 0.0:
            return [(x, y, yaw)] * n

        if not self.path_xy:
            # Constant-velocity fallback along the current heading.
            return [
                (
                    x + v * self.horizon_dt * k * math.cos(yaw),
                    y + v * self.horizon_dt * k * math.sin(yaw),
                    yaw,
                )
                for k in range(n)
            ]

        idx = self._closest_path_idx(x, y)
        # Residual between the robot and the path (nonzero mid-overtake or
        # with tracking error); blend it out linearly over the horizon so the
        # prediction starts where the robot IS and rejoins the path.
        rx = x - self.path_xy[idx][0]
        ry = y - self.path_xy[idx][1]

        pred = [(x, y, yaw)]
        for k in range(1, n):
            px, py, pyaw = self._pose_at_arc(idx, v * self.horizon_dt * k)
            fade = 1.0 - k / (n - 1)
            pred.append((px + rx * fade, py + ry * fade, pyaw))
        return pred

    # ------------------------------------------------------------------
    def _tick(self):
        pose = self._lookup_pose()
        stamp = time.time()
        blocked, blocked_distance, detour_intent = self._obstacle_report()
        self.last_report = (blocked, blocked_distance, detour_intent)

        if pose is None:
            packet = self._pack(
                localized=False, x=0.0, y=0.0, yaw=0.0, v=0.0,
                moving=False, predicted=[], stamp=stamp,
                blocked=blocked, blocked_distance=blocked_distance,
                detour_intent=detour_intent,
            )
        else:
            x, y, yaw = pose
            v = self._update_speed(x, y)
            predicted = self._predict(x, y, yaw, v)
            packet = self._pack(
                localized=True, x=x, y=y, yaw=yaw, v=v,
                moving=v > 0.0, predicted=predicted, stamp=stamp,
                blocked=blocked, blocked_distance=blocked_distance,
                detour_intent=detour_intent,
            )
            self._publish_debug(x, y, yaw, predicted)

        try:
            self.sock.sendto(packet, self.target)
            self.tx_count += 1
        except OSError as e:
            self.tx_errors += 1
            self.get_logger().warn(
                f"UDP send failed: {e}", throttle_duration_sec=5.0
            )
        self.seq += 1

    def _pack(self, localized, x, y, yaw, v, moving, predicted, stamp,
              blocked=None, blocked_distance=None, detour_intent=False):
        predicted = [
            (round(px, 3), round(py, 3), round(pyaw, 3))
            for px, py, pyaw in predicted
        ]
        while True:
            msg = {
                "s": SCHEMA_VERSION,
                "id": self.vehicle_id,
                "q": self.seq,
                "t": round(stamp, 3),
                "loc": 1 if localized else 0,
                "x": round(x, 3),
                "y": round(y, 3),
                "th": round(yaw, 3),
                "v": round(v, 3),
                "ms": "M" if moving else "S",
                "hdt": round(self.horizon_dt, 3),
                "p": [c for pt in predicted for c in pt],
            }
            # Omitted entirely when unknown: the receiver decodes a missing
            # 'bl' as "cannot tell", which is never read as "clear".
            if blocked is not None:
                msg["bl"] = 1 if blocked else 0
                msg["di"] = 1 if detour_intent else 0
                if blocked_distance is not None and math.isfinite(
                    blocked_distance
                ):
                    msg["bd"] = round(float(blocked_distance), 3)
            data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
            if len(data) <= MAX_PACKET_BYTES or not predicted:
                return data
            predicted = predicted[: max(1, len(predicted) - 4)]

    # ------------------------------------------------------------------
    @staticmethod
    def _yaw_to_quat(yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def _publish_debug(self, x, y, yaw, predicted):
        now = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        _, _, qz, qw = self._yaw_to_quat(yaw)
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.debug_pose_pub.publish(pose_msg)

        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = self.map_frame
        for px, py, pyaw in predicted:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = px
            ps.pose.position.y = py
            _, _, qz, qw = self._yaw_to_quat(pyaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)
        self.debug_pred_pub.publish(path_msg)

    def _publish_stats(self):
        # Read the tick's cached report; _obstacle_report advances the
        # blocked-duration timer and must not be driven from diagnostics.
        blocked, blocked_distance, detour_intent = self.last_report
        self.stats_pub.publish(
            String(
                data=json.dumps(
                    {
                        "tx": self.tx_count,
                        "tx_errors": self.tx_errors,
                        "seq": self.seq,
                        "speed": round(self.speed_ema, 3),
                        "blocked": blocked,
                        "blocked_distance": (
                            None if blocked_distance is None
                            else round(float(blocked_distance), 3)
                        ),
                        "detour_intent": detour_intent,
                    }
                )
            )
        )


def main():
    rclpy.init()
    node = RosbotV2VBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
