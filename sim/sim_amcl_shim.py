#!/usr/bin/env python3
"""Sim stand-in for AMCL.

The MPC gates motion on /amcl_pose covariance (require_amcl_quality). In
simulation there is no AMCL — localization is ground truth from Gazebo — so
this republishes the map->base_link TF as a PoseWithCovarianceStamped with
a small, converged covariance.

This keeps the MPC's real safety gate ACTIVE (rather than switching it off
with require_amcl_quality:=false) and exercises the same code path the
physical car uses.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener


class SimAmclShim(Node):
    def __init__(self):
        super().__init__("sim_amcl_shim")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("variance", 0.002)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.var = float(self.get_parameter("variance").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, "/amcl_pose", 10
        )
        self.warned = False
        self.create_timer(
            1.0 / float(self.get_parameter("rate_hz").value), self.tick
        )
        self.get_logger().info(
            f"sim AMCL shim: {self.map_frame}->{self.base_frame} "
            f"-> /amcl_pose (variance {self.var})"
        )

    def tick(self):
        try:
            t = self.buf.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception as e:
            if not self.warned:
                self.get_logger().warn(f"TF not ready: {e}")
                self.warned = True
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = t.transform.translation.x
        msg.pose.pose.position.y = t.transform.translation.y
        msg.pose.pose.orientation = t.transform.rotation

        cov = [0.0] * 36
        cov[0] = self.var       # x
        cov[7] = self.var       # y
        cov[35] = self.var      # yaw
        msg.pose.covariance = cov
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SimAmclShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
