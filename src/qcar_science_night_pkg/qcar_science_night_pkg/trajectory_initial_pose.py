#!/usr/bin/env python3

"""Publish an AMCL initial pose taken from the selected trajectory."""

import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class TrajectoryInitialPose(Node):
    def __init__(self):
        super().__init__("trajectory_initial_pose")

        self.declare_parameter(
            "trajectory_file",
            os.path.expanduser("~/ros2_ws/recorded_path_amcl_final_long.npy"),
        )
        self.declare_parameter("waypoint_index", 0)
        self.declare_parameter("position_variance", 0.25)
        self.declare_parameter("yaw_variance", 0.0685)
        self.declare_parameter("wait_for_odom", True)
        self.declare_parameter("timeout_sec", 30.0)

        trajectory_file = os.path.abspath(os.path.expanduser(
            self.get_parameter("trajectory_file").value
        ))
        trajectory = np.load(trajectory_file)
        if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) == 0:
            raise RuntimeError("Initial-pose trajectory must be a non-empty Nx3/Nx4 array")
        if not np.all(np.isfinite(trajectory[:, :3])):
            raise RuntimeError("Initial-pose trajectory contains non-finite values")

        requested_index = int(self.get_parameter("waypoint_index").value)
        if requested_index < 0 or requested_index >= len(trajectory):
            raise IndexError(
                f"waypoint_index {requested_index} is outside [0, {len(trajectory) - 1}]"
            )
        self.index = requested_index
        self.x = float(trajectory[self.index, 0])
        self.y = float(trajectory[self.index, 1])
        self.yaw = float(trajectory[self.index, 2])
        self.position_variance = float(self.get_parameter("position_variance").value)
        self.yaw_variance = float(self.get_parameter("yaw_variance").value)
        self.wait_for_odom = bool(self.get_parameter("wait_for_odom").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        if (
            self.position_variance <= 0.0
            or self.yaw_variance <= 0.0
            or self.timeout_sec <= 0.0
        ):
            raise ValueError("pose variances and timeout_sec must be positive")

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.started_at = time.monotonic()
        self.done = False
        self.success = False
        self.timer = self.create_timer(0.25, self._try_publish)

        self.get_logger().warn(
            "Place the physical QCar at the selected waypoint before using this "
            f"pose: file={trajectory_file}, idx={self.index}, "
            f"x={self.x:.3f}, y={self.y:.3f}, "
            f"yaw={math.degrees(self.yaw):.1f} deg"
        )

    def _try_publish(self):
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.timeout_sec:
            self.get_logger().error(
                "Timed out waiting for AMCL subscription and odom -> base_link"
            )
            self.done = True
            self.timer.cancel()
            return

        if self.publisher.get_subscription_count() < 1:
            return

        odom_transform = None
        if self.wait_for_odom:
            try:
                odom_transform = self.tf_buffer.lookup_transform(
                    "odom", "base_link", Time()
                )
            except TransformException:
                return

        message = PoseWithCovarianceStamped()
        # Stamp at a transform that is already in the buffer.  Using "now"
        # here can put the initial pose a few milliseconds ahead of odometry
        # and trigger AMCL's extrapolation-into-the-future warning.
        if odom_transform is not None and hasattr(odom_transform, "header"):
            message.header.stamp = odom_transform.header.stamp
        else:
            message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        message.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        message.pose.covariance[0] = self.position_variance
        message.pose.covariance[7] = self.position_variance
        message.pose.covariance[35] = self.yaw_variance
        self.publisher.publish(message)

        self.get_logger().info(
            f"Published /initialpose from waypoint {self.index}: "
            f"({self.x:.3f}, {self.y:.3f}, {math.degrees(self.yaw):.1f} deg)"
        )
        self.success = True
        self.done = True
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryInitialPose()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.25)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        success = node.success
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
