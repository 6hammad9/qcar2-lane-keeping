#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import csv
import math
import os

class MarkerPathSaver(Node):
    def __init__(self):
        super().__init__('marker_path_saver')

        self.output_file = os.path.expanduser('~/recorded_path.csv')
        self.saved = False

        self.subscription = self.create_subscription(
            MarkerArray,
            '/trajectory_node_list',
            self.marker_cb,
            10
        )
        self.get_logger().info('Waiting for MarkerArray on /trajectory_node_list ...')

    def marker_cb(self, msg):
        if self.saved:
            return

        points = []

        for marker in msg.markers:
            if marker.points:
                # LINE_STRIP / LINE_LIST type
                for p in marker.points:
                    points.append((p.x, p.y))
            else:
                # SPHERE / ARROW type — position in pose
                p = marker.pose.position
                points.append((p.x, p.y))

        if len(points) < 2:
            self.get_logger().warn('Not enough points yet, waiting...')
            return

        # Remove duplicate consecutive points (min 2cm apart)
        cleaned = [points[0]]
        for p in points[1:]:
            dx = p[0] - cleaned[-1][0]
            dy = p[1] - cleaned[-1][1]
            if math.hypot(dx, dy) > 0.02:
                cleaned.append(p)

        # Compute yaw from consecutive points
        waypoints = []
        for i in range(len(cleaned)):
            if i < len(cleaned) - 1:
                dx = cleaned[i+1][0] - cleaned[i][0]
                dy = cleaned[i+1][1] - cleaned[i][1]
                yaw = math.atan2(dy, dx)
            else:
                yaw = waypoints[-1][2]
            waypoints.append((cleaned[i][0], cleaned[i][1], yaw))

        # Save to CSV
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'yaw'])
            writer.writerows(waypoints)

        self.get_logger().info(f'✅ Saved {len(waypoints)} waypoints to {self.output_file}')
        self.saved = True

def main():
    rclpy.init()
    node = MarkerPathSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()