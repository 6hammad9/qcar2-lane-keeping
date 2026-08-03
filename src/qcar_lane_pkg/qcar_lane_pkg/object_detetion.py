#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class LidarObstacleDetector(Node):

    def __init__(self):
        super().__init__('lidar_obstacle_detector')

        self.declare_parameter('safe_distance', 0.45)
        self.declare_parameter('warning_distance', 0.75)
        self.declare_parameter('front_angle_deg', 45.0)
        self.declare_parameter('minimum_points', 2)

        # 180 deg because your LiDAR front/back is reversed
        self.declare_parameter('front_offset_deg', 180.0)

        self.safe_distance = self.get_parameter('safe_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.front_angle_deg = self.get_parameter('front_angle_deg').value
        self.minimum_points = self.get_parameter('minimum_points').value
        self.front_offset_deg = self.get_parameter('front_offset_deg').value

        self.motion_pub = self.create_publisher(Bool, '/motion_enable', 10)
        self.distance_pub = self.create_publisher(Float32, '/obstacle_distance', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info("LiDAR obstacle detector started")

    def scan_callback(self, msg):

        front_angle = math.radians(self.front_angle_deg)
        front_offset = math.radians(self.front_offset_deg)

        front_points = []

        for i, r in enumerate(msg.ranges):

            if np.isnan(r) or np.isinf(r):
                continue

            if not (0.05 < r < 2.0):
                continue

            angle = msg.angle_min + i * msg.angle_increment

            relative_angle = math.atan2(
                math.sin(angle - front_offset),
                math.cos(angle - front_offset)
            )

            if abs(relative_angle) <= front_angle:
                front_points.append(r)

        motion = True
        obstacle_distance = -1.0

        if len(front_points) > 0:

            front_points = np.array(front_points)

            obstacle_distance = float(np.min(front_points))

            close_points = int(np.sum(front_points < self.safe_distance))

            if close_points >= self.minimum_points:
                motion = False

        motion_msg = Bool()
        motion_msg.data = motion

        dist_msg = Float32()
        dist_msg.data = obstacle_distance

        self.motion_pub.publish(motion_msg)
        self.distance_pub.publish(dist_msg)

        state = "DRIVE" if motion else "STOP"

        self.get_logger().info(
            f"{state} | dist={obstacle_distance:.2f} m | points={len(front_points)}"
        )


def main():
    rclpy.init()
    node = LidarObstacleDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()