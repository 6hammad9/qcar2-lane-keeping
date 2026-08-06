#!/usr/bin/env python3
"""Show what the LiDAR actually returns in front of the car.

Answers the one question the lidar_overtake log cannot: is the object
producing returns at all? If it is, this prints where they land in vehicle
coordinates, so a miss can be attributed to geometry (wrong lateral offset),
range (beyond the active stop distance) or the scan plane (nothing there at
all, i.e. the beam is passing over or under the object).

Run it alongside everything else -- it only subscribes.

    ros2 run qcar_science_night_pkg lidar_probe        # if installed
    python3 utils/lidar_probe.py                       # or directly

Options mirror the lidar_overtake parameters so the boxes match what the
real node is using:

    python3 utils/lidar_probe.py --ros-args \
      -p front_offset_deg:=180.0 \
      -p lane_width_m:=0.43 \
      -p emergency_half_width_m:=0.12 \
      -p front_narrow_half_width_m:=0.16

Reading the output:

    nearest returns    empty  -> nothing in front at that height at all.
                                The beam is missing the object; no parameter
                                can fix that. Measure the mount height
                                (tf2_echo base_link lidar) against the
                                object, or raise the object.
    narrow n=0         but wide n>0 -> the object is outside the corridor
                                laterally. Check kappa, or widen
                                front_narrow_half_width_m.
    narrow n>0         but the car does not stop -> range. Compare the
                                printed distance against front_stop_curve_m
                                (0.45 m default), which is the ONLY limit
                                that applies while context=CURVE.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class LidarProbe(Node):

    def __init__(self):
        super().__init__("lidar_probe")

        self.declare_parameter("front_offset_deg", 180.0)
        self.declare_parameter("lane_width_m", 0.43)
        self.declare_parameter("emergency_half_width_m", 0.12)
        self.declare_parameter("front_narrow_half_width_m", 0.16)
        self.declare_parameter("report_range_m", 1.5)
        self.declare_parameter("path_curvature", 0.0)
        self.declare_parameter("period_s", 1.0)

        gp = self.get_parameter
        self.front_offset = math.radians(
            float(gp("front_offset_deg").value)
        )
        self.half_lane = float(gp("lane_width_m").value) / 2.0
        self.emergency_half = float(gp("emergency_half_width_m").value)
        self.narrow_half = float(gp("front_narrow_half_width_m").value)
        self.report_range = float(gp("report_range_m").value)
        self.curvature = float(gp("path_curvature").value)

        self.latest = None
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(float(gp("period_s").value), self.report)

        self.get_logger().info(
            f"lidar_probe: front_offset={math.degrees(self.front_offset):.0f} deg, "
            f"lane_half={self.half_lane:.3f}, "
            f"narrow_half={self.narrow_half:.3f}, "
            f"emg_half={self.emergency_half:.3f}, "
            f"kappa={self.curvature:+.2f}"
        )

    def scan_callback(self, msg):
        self.latest = msg

    def to_vehicle_xy(self, msg):
        """Same transform lidar_sector_analyzer uses, so boxes agree."""
        points = []

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if not 0.03 < r < 4.0:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            rel = math.atan2(
                math.sin(angle - self.front_offset),
                math.cos(angle - self.front_offset),
            )

            x = r * math.cos(rel)
            y = r * math.sin(rel)
            if x <= 0.0:
                continue

            points.append((x, y, r))

        return points

    def corridor_center(self, x):
        return 0.5 * self.curvature * x * x

    def report(self):
        if self.latest is None:
            self.get_logger().warn("no /scan yet")
            return

        points = self.to_vehicle_xy(self.latest)
        forward = [p for p in points if p[2] <= self.report_range]
        forward.sort(key=lambda p: p[2])

        if not forward:
            self.get_logger().warn(
                f"NOTHING within {self.report_range:.2f} m in front. "
                "Either the space really is clear, or the scan plane is "
                "missing the object -- check the mount height against the "
                "object height."
            )
            return

        wide = [p for p in forward if abs(p[1]) <= self.half_lane]
        narrow = [
            p for p in forward
            if abs(p[1] - self.corridor_center(p[0])) <= self.narrow_half
        ]
        emg = [
            p for p in forward
            if abs(p[1] - self.corridor_center(p[0])) <= self.emergency_half
        ]

        def nearest(group):
            return f"{min(p[2] for p in group):.2f}" if group else "  -  "

        lines = [
            f"total={len(forward):3d} <{self.report_range:.2f}m | "
            f"wide n={len(wide):3d} min={nearest(wide)} | "
            f"NARROW n={len(narrow):3d} min={nearest(narrow)} | "
            f"emg n={len(emg):3d} min={nearest(emg)}",
            "  nearest returns (vehicle frame, y>0 = left):",
        ]

        for x, y, r in forward[:8]:
            dy = y - self.corridor_center(x)
            tag = "NARROW" if abs(dy) <= self.narrow_half else "      "
            lines.append(
                f"    x={x:+.3f}  y={y:+.3f}  r={r:.3f}  "
                f"dy_corridor={dy:+.3f}  {tag}"
            )

        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = LidarProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
