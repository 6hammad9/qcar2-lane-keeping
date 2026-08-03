#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class AmclNudge(Node):
    def __init__(self):
        super().__init__("amcl_nudge_node")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)

        self.speed = 0.1
        self.duration = 5.0

        self.steer_amp = 0.18      # radians, about 10 degrees
        self.wiggle_hz = 0.7

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().warn("AMCL snake/wiggle nudge started.")

    def loop(self):
        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds * 1e-9

        cmd = Twist()

        if elapsed < self.duration:
            cmd.linear.x = self.speed
            cmd.angular.z = self.steer_amp * math.sin(
                2.0 * math.pi * self.wiggle_hz * elapsed
            )
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        if elapsed >= self.duration + 0.2:
            self.get_logger().warn("AMCL nudge complete.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = AmclNudge()
    rclpy.spin(node)


if __name__ == "__main__":
    main()