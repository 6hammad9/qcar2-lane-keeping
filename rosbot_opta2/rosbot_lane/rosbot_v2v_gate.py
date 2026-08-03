#!/usr/bin/env python3
"""V2V command gate — runs on the ROSbot. The ROSbot's controller is untouched.

Receives HOLD/PROCEED instructions from the QCar over UDP and gates the
trajectory follower's velocity command accordingly. This is the ROSbot half of
the asymmetric coordination: the QCar runs the only optimizer and decides; this
node does not plan, negotiate, or hold an opinion. It relays or zeroes.

Why it exists
-------------
A blocked ROSbot waits 1.5 s and then detours LEFT around the obstacle
(`trajectory_follower_node._is_overtake_feasible` / `_generate_detour`). That
feasibility check inspects the passing lane from only 0.2 m behind the robot,
so a QCar overtaking from further back is invisible to it and both vehicles
claim the same lane. The QCar cannot solve this alone — the swerve is not
predictable from the ROSbot's broadcast horizon — so the QCar holds the ROSbot
for the length of its pass and releases it afterwards. The ROSbot then performs
its detour into a lane the QCar has already left.

Wiring (launch-level, so the follower's source stays unmodified)
---------------------------------------------------------------
Remap the follower's output and let this node own the real topic:

    ros2 run rosbot_lane trajectory_follower_node --ros-args \
        -r /rosbot3/cmd_vel:=/rosbot3/cmd_vel_raw
    python3 rosbot_v2v_gate.py --ros-args -p bind_port:=47101

A held robot still *plans* normally: the follower keeps running, still sees the
obstacle, and may well generate its detour while stopped. That is intended —
the detour is generated from the pose it is standing at, so it stays valid and
simply executes on release.

Fail-safe
---------
A hold is a bounded lease, never a latch. Each command carries a ttl and this
node resumes passthrough once it expires, so a link failure mid-pass costs at
most ttl seconds of standing still instead of stranding the robot forever.
With no command ever received, this node is a pure passthrough and the ROSbot
behaves exactly as it does without V2V.
"""

import json
import math
import os
import socket
import sys
import threading
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
MAX_PACKET_BYTES = 1400
MAX_COMMAND_TTL_SEC = 5.0
COMMAND_HOLD = "H"
COMMAND_PROCEED = "P"


class CommandError(ValueError):
    """Raised when an incoming command datagram fails validation."""


def parse_command(data, expected_target=None, expected_issuer=None):
    """Validate and decode one QCar command datagram.

    Kept byte-identical in behaviour to
    ``qcar_science_night_pkg.v2v_common.parse_command``; duplicated rather
    than imported so this file stays copy-and-run on the robot. An
    unrecognised command word is an error, never a default, so no corrupted
    byte can be read as "proceed".
    """
    if len(data) > MAX_PACKET_BYTES:
        raise CommandError("oversized command datagram")
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CommandError(f"not valid JSON: {e}")
    if not isinstance(msg, dict):
        raise CommandError("not a JSON object")
    if msg.get("s") not in SUPPORTED_SCHEMA_VERSIONS:
        raise CommandError(f"unsupported schema {msg.get('s')!r}")

    issuer = msg.get("id")
    target = msg.get("to")
    if not isinstance(issuer, str) or not issuer:
        raise CommandError("missing issuing vehicle id")
    if not isinstance(target, str) or not target:
        raise CommandError("missing target vehicle id")
    if expected_issuer is not None and issuer != expected_issuer:
        raise CommandError(f"unexpected issuer {issuer!r}")
    if expected_target is not None and target != expected_target:
        raise CommandError(f"command not addressed to us ({target!r})")

    word = msg.get("c")
    if word not in (COMMAND_HOLD, COMMAND_PROCEED):
        raise CommandError(f"unknown command {word!r}")

    seq = msg.get("q")
    ttl = msg.get("ttl")
    for name, value in (("q", seq), ("ttl", ttl)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CommandError(f"field {name!r} not numeric")
        if not math.isfinite(float(value)):
            raise CommandError(f"field {name!r} not finite")
    if not 0.0 < float(ttl) <= MAX_COMMAND_TTL_SEC:
        raise CommandError(f"ttl {ttl} out of range")

    return {
        "id": issuer,
        "target": target,
        "seq": int(seq),
        "hold": word == COMMAND_HOLD,
        "ttl_sec": float(ttl),
    }


class RosbotV2VGate(Node):

    def __init__(self):
        super().__init__("rosbot_v2v_gate")

        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("bind_port", 47101)
        self.declare_parameter("vehicle_id", "rosbot3")
        self.declare_parameter("expected_issuer", "qcar2")
        self.declare_parameter("input_topic", "/rosbot3/cmd_vel_raw")
        self.declare_parameter("output_topic", "/rosbot3/cmd_vel")
        # TwistStamped matches trajectory_follower_node's publisher.
        self.declare_parameter("use_twist_stamped", True)

        gp = lambda name: self.get_parameter(name).value
        self.vehicle_id = str(gp("vehicle_id"))
        self.expected_issuer = str(gp("expected_issuer"))
        stamped = bool(gp("use_twist_stamped"))
        self.msg_type = TwistStamped if stamped else Twist

        self.lock = threading.Lock()
        self.hold_until = None       # monotonic deadline, None = free
        self.last_seq = -1
        self.rx_count = 0
        self.parse_errors = 0
        self.hold_events = 0
        self.gated_messages = 0
        self.was_holding = False

        self.cmd_pub = self.create_publisher(
            self.msg_type, str(gp("output_topic")), 10
        )
        self.create_subscription(
            self.msg_type, str(gp("input_topic")), self._cmd_cb, 10
        )

        self.holding_pub = self.create_publisher(Bool, "/v2v/rx_holding", 10)
        self.stats_pub = self.create_publisher(String, "/v2v/rx_stats", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((str(gp("bind_ip")), int(gp("bind_port"))))
        self.sock.settimeout(0.2)

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        self.create_timer(0.1, self._publish_holding)
        self.create_timer(1.0, self._publish_stats)

        self.get_logger().info(
            f"V2V gate up: udp://{gp('bind_ip')}:{gp('bind_port')} "
            f"as '{self.vehicle_id}', {gp('input_topic')} -> "
            f"{gp('output_topic')}"
        )

    # ------------------------------------------------------------------
    def _rx_loop(self):
        while rclpy.ok():
            try:
                data, _addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                cmd = parse_command(
                    data,
                    expected_target=self.vehicle_id,
                    expected_issuer=self.expected_issuer,
                )
            except CommandError as e:
                with self.lock:
                    self.parse_errors += 1
                self.get_logger().warn(
                    f"Dropped V2V command: {e}", throttle_duration_sec=2.0
                )
                continue

            with self.lock:
                self.rx_count += 1
                # Reordered UDP must not resurrect a superseded instruction.
                # A large backwards jump means the QCar restarted, so resync.
                if (
                    cmd["seq"] <= self.last_seq
                    and self.last_seq - cmd["seq"] < 1000
                ):
                    continue
                self.last_seq = cmd["seq"]
                if cmd["hold"]:
                    self.hold_until = time.monotonic() + cmd["ttl_sec"]
                else:
                    self.hold_until = None

    def _is_holding(self):
        with self.lock:
            deadline = self.hold_until
        if deadline is None:
            return False
        if time.monotonic() >= deadline:
            # Lease expired. Resuming is the fail-safe: a dead link must not
            # leave the robot parked in a live lane indefinitely.
            with self.lock:
                if self.hold_until is not None and (
                    time.monotonic() >= self.hold_until
                ):
                    self.hold_until = None
            return False
        return True

    def _cmd_cb(self, msg):
        if not self._is_holding():
            self.cmd_pub.publish(msg)
            return

        with self.lock:
            self.gated_messages += 1

        stop = self.msg_type()
        if self.msg_type is TwistStamped:
            stop.header = msg.header
            stop.twist.linear.x = 0.0
            stop.twist.angular.z = 0.0
        else:
            stop.linear.x = 0.0
            stop.angular.z = 0.0
        self.cmd_pub.publish(stop)

    def _publish_holding(self):
        holding = self._is_holding()
        if holding and not self.was_holding:
            with self.lock:
                self.hold_events += 1
            self.get_logger().info("V2V hold engaged — QCar is passing")
        elif self.was_holding and not holding:
            self.get_logger().info("V2V hold released — resuming")
        self.was_holding = holding
        self.holding_pub.publish(Bool(data=bool(holding)))

    def _publish_stats(self):
        with self.lock:
            stats = {
                "rx": self.rx_count,
                "parse_errors": self.parse_errors,
                "hold_events": self.hold_events,
                "gated_messages": self.gated_messages,
                "holding": self.was_holding,
            }
        self.stats_pub.publish(String(data=json.dumps(stats)))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    rclpy.init()
    node = RosbotV2VGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
