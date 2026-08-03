#!/usr/bin/env python3
"""Small live dashboard for the asymmetric QCar/ROSbot V2V demo."""

import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String


class DemoMonitor(Node):
    def __init__(self):
        super().__init__("v2v_live_monitor")
        self.values = {
            "v2v_alive": False,
            "on_path": False,
            "gap": None,
            "rosbot_speed": None,
            "idx": None,
            "drive_state": "WAITING",
            "motion": False,
            "offset": None,
            "qcar_behavior": {},
            "qcar_command": {},
            "rosbot_state": {},
            "rosbot_command": {},
        }
        self.updated = {}

        self._sub(Bool, "/v2v/alive", "v2v_alive")
        self._sub(Bool, "/v2v/on_path", "on_path")
        self._sub(Float32, "/v2v/gap", "gap")
        self._sub(Float32, "/v2v/rosbot_speed", "rosbot_speed")
        self._sub(Int32, "/current_path_idx", "idx")
        self._sub(String, "/drive_state", "drive_state")
        self._sub(Bool, "/motion_enable", "motion")
        self._sub(Float32, "/avoidance_offset", "offset")
        self._json_sub("/v2v/qcar/behavior_state", "qcar_behavior")
        self._json_sub("/v2v/qcar/command", "qcar_command")
        self._json_sub("/v2v/rosbot/follower_state", "rosbot_state")
        self._json_sub("/v2v/rosbot/command", "rosbot_command")
        self.create_timer(0.25, self.render)

    def _sub(self, msg_type, topic, key):
        def callback(msg):
            self.values[key] = msg.data
            self.updated[key] = time.monotonic()

        self.create_subscription(msg_type, topic, callback, 10)

    def _json_sub(self, topic, key):
        def callback(msg):
            try:
                self.values[key] = json.loads(msg.data)
            except (TypeError, ValueError):
                self.values[key] = {"raw": msg.data}
            self.updated[key] = time.monotonic()

        self.create_subscription(String, topic, callback, 10)

    @staticmethod
    def _number(value, suffix=""):
        if value is None:
            return "--"
        return f"{float(value):.3f}{suffix}"

    def _age(self, key):
        stamp = self.updated.get(key)
        return "never" if stamp is None else f"{time.monotonic() - stamp:.1f}s"

    def render(self):
        v = self.values
        qb = v["qcar_behavior"]
        qc = v["qcar_command"]
        rb = v["rosbot_state"]
        rc = v["rosbot_command"]

        lines = [
            "QCAR + ROSBOT ASYMMETRIC V2V -- LIVE",
            "=" * 57,
            (
                f"LINK     alive={str(bool(v['v2v_alive'])):<5} "
                f"same_path={str(bool(v['on_path'])):<5} "
                f"gap={self._number(v['gap'], ' m'):<10} "
                f"rx_age={self._age('v2v_alive')}"
            ),
            (
                f"ROSBOT   state={rb.get('state', 'WAITING'):<20} "
                f"cmd_v={self._number(rc.get('speed_mps', v['rosbot_speed']), ' m/s')}"
            ),
            (
                "         proximity speed_scale="
                f"{self._number(rc.get('proximity_speed_scale'), ''):<7} "
                "V2V horizon optimization=OFF"
            ),
            (
                f"QCAR     idx={str(v['idx']):<5} "
                f"state={str(v['drive_state']):<18} "
                f"motion={str(bool(v['motion'])):<5}"
            ),
            (
                f"         lead_mode={qb.get('lead_mode', 'WAITING'):<27} "
                f"offset={self._number(qc.get('avoidance_offset_m', v['offset']), ' m')}"
            ),
            "-" * 57,
            "Expected: moving ROSbot reports its pose/horizon. QCar slows,",
            "follows, or passes using V2V + LiDAR. ROSbot uses only its",
            "local slow/stop safety; this remains deliberately asymmetric.",
            "",
            "Ctrl+C closes this monitor only; use sim/wsl/force_stop.sh",
            "for an immediate vehicle stop.",
        ]
        print("\033[2J\033[H" + "\n".join(lines), end="", flush=True)


def main():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    rclpy.init()
    node = DemoMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # SIGINT may already have shut down the default context. Guarding the
        # second call keeps Ctrl+C / ``timeout -s INT`` from ending an
        # otherwise healthy dashboard with an RCLError traceback.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
