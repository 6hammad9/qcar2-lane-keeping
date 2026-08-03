
#!/usr/bin/env python3
"""
Path publisher for Regulated Pure Pursuit controller.
This node ONLY publishes the path - the controller handles the control.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import json
import math
import os


class PurePursuitPathPublisher(Node):

    def __init__(self):
        super().__init__('pure_pursuit_path_publisher')

        # Load trajectory
        self._trajectory_file = '/home/nvidia/ros2_ws/recorded_path_clean_latest.json'
        
        # Check if file exists
        if not os.path.exists(self._trajectory_file):
            self.get_logger().error(f"Trajectory file not found: {self._trajectory_file}")
            self._trajectory = []
        else:
            self._trajectory = []
            self._load_trajectory()

        # Publisher - publish to /plan for Nav2 controller
        self._path_pub = self.create_publisher(Path, '/plan', 10)
        
        # Timer to publish path periodically
        self.create_timer(0.5, self._publish_path)

        self.get_logger().info(f"Path Publisher initialized - {len(self._trajectory)} waypoints")
        self.get_logger().info("Publishing path to /plan for Regulated Pure Pursuit controller")

    def _load_trajectory(self):
        try:
            with open(self._trajectory_file, 'r') as f:
                poses = json.load(f)

            for p in poses:
                yaw = 2 * math.atan2(p['qz'], p['qw'])
                self._trajectory.append((p['x'], p['y'], yaw))
            
            self.get_logger().info(f"Successfully loaded {len(self._trajectory)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory: {e}")
            self._trajectory = []

    def _publish_path(self):
        if not self._trajectory:
            self.get_logger().warn("No trajectory loaded - cannot publish path")
            return

        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y, yaw in self._trajectory:
            p = PoseStamped()
            p.header.frame_id = "map"
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.z = math.sin(yaw / 2)
            p.pose.orientation.w = math.cos(yaw / 2)
            msg.poses.append(p)

        self._path_pub.publish(msg)
        self.get_logger().debug(f"Published path with {len(msg.poses)} poses")


def main():
    rclpy.init()
    node = PurePursuitPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
