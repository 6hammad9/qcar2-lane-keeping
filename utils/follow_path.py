
import rclpy
import json
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        with open('/home/nvidia/ros2_ws/recorded_path.json', 'r') as f:
            self.waypoints = json.load(f)

        self.idx = 0
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints, starting...')

    def get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            tr = t.transform.translation
            ro = t.transform.rotation
            yaw = 2 * math.atan2(ro.z, ro.w)
            return tr.x, tr.y, yaw
        except:
            return None, None, None

    def control_loop(self):
        if self.idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            self.get_logger().info('Path complete!')
            self.timer.cancel()
            return

        x, y, yaw = self.get_pose()
        if x is None:
            self.get_logger().warn('Waiting for TF...')
            return

        target = self.waypoints[self.idx]
        dx = target['x'] - x
        dy = target['y'] - y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.05:
            self.idx += 1
            self.get_logger().info(f'Waypoint {self.idx}/{len(self.waypoints)}')
            return

        angle_to_target = math.atan2(dy, dx)
        angle_error = angle_to_target - yaw

        # Normalize to [-pi, pi]
        while angle_error > math.pi:  angle_error -= 2*math.pi
        while angle_error < -math.pi: angle_error += 2*math.pi

        cmd = Twist()
        if abs(angle_error) < 0.3:
            cmd.linear.x = min(0.3, dist)
        else:
            cmd.linear.x = 0.05  # creep forward while turning
        cmd.angular.z = 1.5 * angle_error

        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f'WP {self.idx}/{len(self.waypoints)} | '
            f'car: ({x:.3f}, {y:.3f}) | '
            f'target: ({target["x"]:.3f}, {target["y"]:.3f}) | '
            f'dist: {dist:.3f}m | '
            f'angle_err: {math.degrees(angle_error):.1f}deg'
        )

def main():
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())  # stop car on exit
        rclpy.shutdown()

if __name__ == '__main__':
    main()
