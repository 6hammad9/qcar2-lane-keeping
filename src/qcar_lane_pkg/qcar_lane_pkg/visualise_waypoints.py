
import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')

        self.pub = self.create_publisher(MarkerArray, '/waypoints_marker', 10)

        data = np.loadtxt('/home/nvidia/ros2_ws/src/qcar_lane_pkg/qcar_lane_pkg/waypoints.csv', delimiter=',', skiprows=1)
        self.waypoints = data[:, :2]
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')

        # Publish repeatedly so RViz can catch it
        self.create_timer(1.0, self.publish_markers)

    def publish_markers(self):
        marker_array = MarkerArray()

        # ── Line strip connecting all points ──────────────────────────────────
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp    = self.get_clock().now().to_msg()
        line.ns              = 'path'
        line.id              = 0
        line.type            = Marker.LINE_STRIP
        line.action          = Marker.ADD
        line.scale.x         = 0.02   # line width in metres
        line.color.r         = 0.0
        line.color.g         = 1.0
        line.color.b         = 0.0
        line.color.a         = 1.0

        for wp in self.waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            line.points.append(p)

        marker_array.markers.append(line)

        # ── Start point (green sphere) ────────────────────────────────────────
        start = Marker()
        start.header.frame_id = 'map'
        start.header.stamp    = self.get_clock().now().to_msg()
        start.ns              = 'start_end'
        start.id              = 1
        start.type            = Marker.SPHERE
        start.action          = Marker.ADD
        start.pose.position.x = float(self.waypoints[0, 0])
        start.pose.position.y = float(self.waypoints[0, 1])
        start.pose.position.z = 0.0
        start.pose.orientation.w = 1.0
        start.scale.x = start.scale.y = start.scale.z = 0.15
        start.color.r = 0.0
        start.color.g = 1.0
        start.color.b = 0.0
        start.color.a = 1.0
        marker_array.markers.append(start)

        # ── End point (red sphere) ────────────────────────────────────────────
        end = Marker()
        end.header.frame_id = 'map'
        end.header.stamp    = self.get_clock().now().to_msg()
        end.ns              = 'start_end'
        end.id              = 2
        end.type            = Marker.SPHERE
        end.action          = Marker.ADD
        end.pose.position.x = float(self.waypoints[-1, 0])
        end.pose.position.y = float(self.waypoints[-1, 1])
        end.pose.position.z = 0.0
        end.pose.orientation.w = 1.0
        end.scale.x = end.scale.y = end.scale.z = 0.15
        end.color.r = 1.0
        end.color.g = 0.0
        end.color.b = 0.0
        end.color.a = 1.0
        marker_array.markers.append(end)

        self.pub.publish(marker_array)

def main():
    rclpy.init()
    node = WaypointVisualizer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
