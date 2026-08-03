#!/usr/bin/env python3
"""
QCar trajectory recorder.

Records the QCar pose from TF into a CSV file.

Default TF recorded:

    map -> base_link

This works with EKF + AMCL if your TF tree is:

    map  -> odom       from AMCL / localization
    odom -> base_link  from EKF / odometry

CSV output columns:

    t, tf_t, x, y, theta, yaw_deg, reason
"""

import csv
import math
import os
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener, TransformException


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    """Convert ROS timestamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class QCarTrajectoryRecorderNode(Node):
    """Record QCar trajectory from TF."""

    def __init__(self):
        super().__init__("qcar_trajectory_recorder_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter(
            "trajectory_file",
            "/home/nvidia/ros2_ws/src/qcar_science_night_pkg/config/qcar_trajectory_01.csv",
        )

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("source_frame", "base_link")

        # TF polling rate: 0.05 s = 20 Hz
        self.declare_parameter("poll_period", 0.05)

        # Record a new point after the QCar moves this distance.
        self.declare_parameter("min_record_distance", 0.02)

        # Also record even if stationary after this many seconds.
        # This helps capture parked start/end poses.
        self.declare_parameter("max_record_interval", 0.5)

        # TF lookup timeout.
        self.declare_parameter("tf_timeout", 0.2)

        # Save one final point when Ctrl+C is pressed.
        self.declare_parameter("save_final_pose", True)

        # Print log every N saved points.
        self.declare_parameter("log_every_n", 25)

        self._trajectory_file = (
            self.get_parameter("trajectory_file")
            .get_parameter_value()
            .string_value
        )

        self._target_frame = (
            self.get_parameter("target_frame")
            .get_parameter_value()
            .string_value
        )

        self._source_frame = (
            self.get_parameter("source_frame")
            .get_parameter_value()
            .string_value
        )

        self._poll_period = (
            self.get_parameter("poll_period")
            .get_parameter_value()
            .double_value
        )

        self._min_record_distance = (
            self.get_parameter("min_record_distance")
            .get_parameter_value()
            .double_value
        )

        self._max_record_interval = (
            self.get_parameter("max_record_interval")
            .get_parameter_value()
            .double_value
        )

        self._tf_timeout = (
            self.get_parameter("tf_timeout")
            .get_parameter_value()
            .double_value
        )

        self._save_final_pose = (
            self.get_parameter("save_final_pose")
            .get_parameter_value()
            .bool_value
        )

        self._log_every_n = (
            self.get_parameter("log_every_n")
            .get_parameter_value()
            .integer_value
        )

        if self._poll_period <= 0.0:
            raise ValueError("poll_period must be > 0")

        if self._min_record_distance < 0.0:
            raise ValueError("min_record_distance must be >= 0")

        if self._max_record_interval < 0.0:
            raise ValueError("max_record_interval must be >= 0")

        if self._tf_timeout < 0.0:
            raise ValueError("tf_timeout must be >= 0")

        if self._log_every_n < 1:
            self._log_every_n = 1

        # --------------------------------------------------
        # TF listener
        # --------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --------------------------------------------------
        # Current pose
        # --------------------------------------------------
        self._current_t: Optional[float] = None
        self._current_tf_t: Optional[float] = None
        self._current_x: Optional[float] = None
        self._current_y: Optional[float] = None
        self._current_theta: Optional[float] = None

        # --------------------------------------------------
        # Last recorded pose
        # --------------------------------------------------
        self._last_recorded_t: Optional[float] = None
        self._last_recorded_x: Optional[float] = None
        self._last_recorded_y: Optional[float] = None

        self._point_count = 0
        self._closed = False

        # --------------------------------------------------
        # CSV setup
        # --------------------------------------------------
        trajectory_dir = os.path.dirname(os.path.abspath(self._trajectory_file))
        os.makedirs(trajectory_dir, exist_ok=True)

        self._traj_file = open(self._trajectory_file, "w", newline="")
        self._traj_writer = csv.writer(self._traj_file)

        self._traj_writer.writerow(
            [
                "t",
                "tf_t",
                "x",
                "y",
                "theta",
                "yaw_deg",
                "reason",
            ]
        )
        self._traj_file.flush()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.create_timer(self._poll_period, self._tf_callback)

        self.get_logger().info("QCar trajectory recorder started")
        self.get_logger().info(f"Saving trajectory to: {self._trajectory_file}")
        self.get_logger().info(
            f"Recording TF transform: {self._target_frame} -> {self._source_frame}"
        )
        self.get_logger().info(
            f"min_record_distance={self._min_record_distance:.3f} m, "
            f"max_record_interval={self._max_record_interval:.3f} s, "
            f"poll_period={self._poll_period:.3f} s"
        )
        self.get_logger().info("Drive the QCar manually. Press Ctrl+C to stop.")

    def _now_sec(self) -> float:
        """Current node time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _tf_callback(self):
        """Read TF and record trajectory point if needed."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame,
                self._source_frame,
                Time(),
                timeout=Duration(seconds=self._tf_timeout),
            )

        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed for {self._target_frame} -> "
                f"{self._source_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        tr = transform.transform.translation
        q = transform.transform.rotation

        self._current_t = self._now_sec()
        self._current_tf_t = stamp_to_sec(transform.header.stamp)
        self._current_x = float(tr.x)
        self._current_y = float(tr.y)
        self._current_theta = quat_to_yaw(q.x, q.y, q.z, q.w)

        self._record_position_if_needed()

    def _record_position_if_needed(self):
        """Record first point, moved-distance points, and stationary time samples."""
        if self._current_x is None or self._current_y is None:
            return

        # First point
        if self._last_recorded_x is None or self._last_recorded_y is None:
            self._save_point(reason="first")
            return

        dx = self._current_x - self._last_recorded_x
        dy = self._current_y - self._last_recorded_y
        distance = math.hypot(dx, dy)

        time_since_last = 0.0
        if self._last_recorded_t is not None and self._current_t is not None:
            time_since_last = self._current_t - self._last_recorded_t

        # Save if moved enough
        if distance >= self._min_record_distance:
            self._save_point(reason="distance")
            return

        # Save occasional stationary samples
        if (
            self._max_record_interval > 0.0
            and self._last_recorded_t is not None
            and time_since_last >= self._max_record_interval
        ):
            self._save_point(reason="time")
            return

    def _save_point(self, reason: str):
        """Save current pose to CSV."""
        if (
            self._current_t is None
            or self._current_tf_t is None
            or self._current_x is None
            or self._current_y is None
            or self._current_theta is None
        ):
            return

        yaw_deg = math.degrees(self._current_theta)

        self._traj_writer.writerow(
            [
                f"{self._current_t:.6f}",
                f"{self._current_tf_t:.6f}",
                f"{self._current_x:.6f}",
                f"{self._current_y:.6f}",
                f"{self._current_theta:.6f}",
                f"{yaw_deg:.3f}",
                reason,
            ]
        )
        self._traj_file.flush()

        self._last_recorded_t = self._current_t
        self._last_recorded_x = self._current_x
        self._last_recorded_y = self._current_y

        self._point_count += 1

        if self._point_count == 1 or self._point_count % self._log_every_n == 0:
            self.get_logger().info(
                f"Point {self._point_count}: "
                f"x={self._current_x:.3f}, "
                f"y={self._current_y:.3f}, "
                f"theta={yaw_deg:.1f} deg, "
                f"reason={reason}"
            )

    def close(self):
        """Save final pose and close CSV file."""
        if self._closed:
            return

        if self._save_final_pose and self._current_x is not None:
            self._save_point(reason="final")

        self._traj_file.flush()
        self._traj_file.close()
        self._closed = True

        self.get_logger().info(
            f"Saved {self._point_count} points to {self._trajectory_file}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = QCarTrajectoryRecorderNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received. Stopping QCar trajectory recorder.")

    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()