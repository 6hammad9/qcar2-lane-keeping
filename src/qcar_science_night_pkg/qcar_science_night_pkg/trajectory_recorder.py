#!/usr/bin/env python3

"""Record a map-frame QCar trajectory directly to an MPC-loadable NPY."""

import csv
import math
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_to_yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


def resample_trajectory(points, spacing):
    """Return uniformly spaced x/y points with tangent-consistent yaw."""

    trajectory = np.asarray(points, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) < 3:
        raise ValueError("At least three Nx2/Nx3 points are required")
    if spacing <= 0.0 or not np.all(np.isfinite(trajectory)):
        raise ValueError("Spacing must be positive and points must be finite")

    segment_lengths = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-5))
    xy = trajectory[keep, :2]
    if len(xy) < 3:
        raise ValueError("Trajectory has too few distinct points")

    cumulative = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)),
    ))
    total_length = float(cumulative[-1])
    if total_length < 2.0 * spacing:
        raise ValueError("Trajectory is too short to resample")

    # linspace avoids a short final segment while keeping spacing within one
    # sample interval of the requested value.
    segment_count = max(2, int(round(total_length / spacing)))
    sample_distance = np.linspace(0.0, total_length, segment_count + 1)
    x = np.interp(sample_distance, cumulative, xy[:, 0])
    y = np.interp(sample_distance, cumulative, xy[:, 1])
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    return np.column_stack((x, y, yaw))


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__("qcar_trajectory_recorder")
        workspace = os.path.expanduser("~/ros2_ws")
        self.declare_parameter(
            "output_file", os.path.join(workspace, "recorded_path_current.npy")
        )
        self.declare_parameter("min_distance", 0.03)
        self.declare_parameter("output_spacing", 0.03)
        self.declare_parameter("sample_period", 0.05)
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("source_frame", "base_link")

        self.output_file = os.path.abspath(os.path.expanduser(
            self.get_parameter("output_file").value
        ))
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.output_spacing = float(self.get_parameter("output_spacing").value)
        sample_period = float(self.get_parameter("sample_period").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)

        if (
            self.min_distance <= 0.0
            or self.output_spacing <= 0.0
            or sample_period <= 0.0
        ):
            raise ValueError(
                "min_distance, output_spacing and sample_period must be positive"
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.points = []
        self.timer = self.create_timer(sample_period, self._sample)
        self.saved = False

        self.get_logger().info(
            f"Recording {self.target_frame} -> {self.source_frame} every "
            f"{self.min_distance:.3f} m to {self.output_file}"
        )

    def _sample(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, Time()
            )
        except TransformException as error:
            self.get_logger().warn(
                f"Waiting for {self.target_frame} -> {self.source_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return

        translation = transform.transform.translation
        yaw = quaternion_to_yaw(transform.transform.rotation)
        point = [float(translation.x), float(translation.y), float(yaw)]

        if self.points:
            distance = math.hypot(
                point[0] - self.points[-1][0], point[1] - self.points[-1][1]
            )
            if distance < self.min_distance:
                return

        self.points.append(point)
        if len(self.points) == 1 or len(self.points) % 50 == 0:
            self.get_logger().info(
                f"Recorded {len(self.points)} points; latest="
                f"({point[0]:.3f}, {point[1]:.3f}, {math.degrees(yaw):.1f} deg)"
            )

    def save(self):
        if self.saved:
            return
        self.saved = True
        if len(self.points) < 3:
            self.get_logger().error("Not enough valid points; trajectory was not saved")
            return

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        raw_trajectory = np.asarray(self.points, dtype=float)
        raw_file = os.path.splitext(self.output_file)[0] + "_raw.npy"
        np.save(raw_file, raw_trajectory)
        try:
            trajectory = resample_trajectory(raw_trajectory, self.output_spacing)
        except ValueError as error:
            self.get_logger().error(
                f"Raw points saved to {raw_file}, but resampling failed: {error}"
            )
            return
        np.save(self.output_file, trajectory)

        csv_file = os.path.splitext(self.output_file)[0] + ".csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "yaw"])
            writer.writerows(trajectory)

        gap = float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]))
        yaw_gap = math.atan2(
            math.sin(trajectory[-1, 2] - trajectory[0, 2]),
            math.cos(trajectory[-1, 2] - trajectory[0, 2]),
        )
        self.get_logger().info(
            f"Saved {len(trajectory)} uniformly spaced points to "
            f"{self.output_file} (raw={raw_file}); "
            f"end/start gap={gap:.3f} m, "
            f"yaw gap={math.degrees(yaw_gap):.1f} deg. "
            "Run validate_path_map before enabling motion."
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
