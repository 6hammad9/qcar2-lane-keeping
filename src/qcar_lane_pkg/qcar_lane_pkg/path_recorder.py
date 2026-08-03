#!/usr/bin/env python3
"""
Path recorder — drive the car manually and this node saves the path to CSV.
Usage:
    ros2 run qcar_lane_pkg path_recorder_node
    # or directly:
    python3 path_recorder.py

Drive the car along the track. Press Ctrl+C when done.
The path is saved to OUTPUT_FILE.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import signal
import sys
from nav_msgs.msg import Odometry

OUTPUT_FILE    = '/home/nvidia/recorded_path_clean.csv'
MIN_DIST       = 0.05   # metres between saved waypoints (filters duplicates)


class PathRecorder(Node):

    def __init__(self):
        super().__init__('path_recorder')

        self.waypoints = []   # list of (x, y, yaw)
        self.last_x    = None
        self.last_y    = None

        self.sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        signal.signal(signal.SIGINT,  self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

        self.get_logger().info(
            f'Recording path from /odom → {OUTPUT_FILE}\n'
            f'Drive the car now. Press Ctrl+C to stop and save.')

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Only save if we've moved MIN_DIST since last waypoint
        if self.last_x is not None:
            dist = math.hypot(x - self.last_x, y - self.last_y)
            if dist < MIN_DIST:
                return

        self.waypoints.append((x, y, yaw))
        self.last_x = x
        self.last_y = y

        if len(self.waypoints) % 20 == 0:
            self.get_logger().info(f'Recorded {len(self.waypoints)} waypoints...')

    def shutdown_handler(self, signum, frame):
        self.save()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def save(self):
        if not self.waypoints:
            self.get_logger().warn('No waypoints recorded — nothing saved.')
            return

        arr = np.array(self.waypoints)
        np.savetxt(OUTPUT_FILE, arr, delimiter=',',
                   header='x,y,yaw', comments='')
        self.get_logger().info(
            f'Saved {len(self.waypoints)} waypoints to {OUTPUT_FILE}\n'
            f'First: x={arr[0,0]:.4f} y={arr[0,1]:.4f} yaw={math.degrees(arr[0,2]):.1f}°\n'
            f'Last:  x={arr[-1,0]:.4f} y={arr[-1,1]:.4f} yaw={math.degrees(arr[-1,2]):.1f}°')


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    rclpy.spin(node)


if __name__ == '__main__':
    main()