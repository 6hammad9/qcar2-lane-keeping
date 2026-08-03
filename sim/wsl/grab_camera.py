#!/usr/bin/env python3
"""Save one QCar camera frame and a LiDAR sector summary.

Usage: python3 grab_camera.py /tmp/qcar_view.png [minimum_path_index]
"""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int32


class Grab(Node):
    def __init__(self, out, minimum_path_index=None):
        super().__init__("grab_camera")
        self.out = out
        self.got_img = False
        self.got_scan = False
        self.minimum_path_index = minimum_path_index
        self.current_path_index = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, "/qcar2/front_camera/image",
                                 self.cb_img, qos)
        self.create_subscription(LaserScan, "/qcar2/lidar/scan",
                                 self.cb_scan, qos)
        self.create_subscription(Int32, "/current_path_idx",
                                 self.cb_index, 10)

    def ready(self):
        return (
            self.minimum_path_index is None
            or (
                self.current_path_index is not None
                and self.current_path_index >= self.minimum_path_index
            )
        )

    def cb_index(self, msg):
        self.current_path_index = int(msg.data)

    def cb_img(self, msg):
        if self.got_img or not self.ready():
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            if msg.encoding in ("rgb8", "bgr8"):
                img = arr.reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    img = img[:, :, ::-1]
            elif msg.encoding == "rgba8":
                img = arr.reshape(msg.height, msg.width, 4)[:, :, :3]
            else:
                print(f"encoding {msg.encoding} unsupported")
                return
            import cv2
            cv2.imwrite(self.out, img[:, :, ::-1])
            print(f"SAVED {self.out}  {msg.width}x{msg.height} {msg.encoding}")
            self.got_img = True
        except Exception as e:
            print(f"image error: {e}")

    def cb_scan(self, msg):
        if self.got_scan or not self.ready():
            return
        r = np.array(msg.ranges)
        finite = r[np.isfinite(r)]
        if len(finite):
            print(f"LIDAR {len(r)} beams, min={finite.min():.2f} m "
                  f"max={finite.max():.2f} m, range_max={msg.range_max:.1f}")
            angles = msg.angle_min + np.arange(len(r)) * msg.angle_increment
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            valid = np.isfinite(r) & (r > 0.05) & (r < 2.0) & (x > 0.0)
            sectors = {
                "front": valid & (x >= 0.10) & (x <= 0.90)
                & (y >= -0.215) & (y <= 0.215),
                "left": valid & (x >= 0.10) & (x <= 0.70)
                & (y >= 0.215) & (y <= 0.645),
                "right": valid & (x >= 0.10) & (x <= 0.70)
                & (y >= -0.645) & (y <= -0.215),
            }
            for name, mask in sectors.items():
                if np.any(mask):
                    print(
                        f"  {name}: n={int(mask.sum())}, "
                        f"r={r[mask].min():.2f}..{r[mask].max():.2f}, "
                        f"x={x[mask].min():.2f}..{x[mask].max():.2f}, "
                        f"y={y[mask].min():.2f}..{y[mask].max():.2f}"
                    )
                    if name == "right":
                        indices = np.flatnonzero(mask)
                        indices = indices[np.argsort(r[indices])[:8]]
                        samples = ", ".join(
                            f"({x[i]:.2f},{y[i]:.2f},{r[i]:.2f})"
                            for i in indices
                        )
                        print(f"    closest xy/r: {samples}")
                else:
                    print(f"  {name}: clear")
            if self.current_path_index is not None:
                print(f"  path_index={self.current_path_index}")
        self.got_scan = True


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qcar_view.png"
    minimum_path_index = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rclpy.init()
    node = Grab(out, minimum_path_index)
    # With an index trigger this may intentionally cover a complete approach.
    iterations = 2400 if minimum_path_index is not None else 200
    for _ in range(iterations):
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.got_img and node.got_scan:
            break
    if not node.got_img:
        print("NO IMAGE RECEIVED")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
