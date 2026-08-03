import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import numpy as np
import math
import tf2_ros
import signal
import sys

from qcar2_interfaces.msg import MotorCommands
from geometry_msgs.msg import PoseWithCovarianceStamped


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')


        self.lookahead_dist = 0.45
        self.base_speed = 0.12
        self.min_speed = 0.08
        self.max_speed = 0.16
        self.wheelbase = 0.256
        self.max_steer = 0.6

        # AMCL convergence
        self.amcl_converged = False
        self.convergence_threshold = 0.03
        self.required_stable_count = 20   # ~1 sec at typical AMCL rate
        self.stable_counter = 0

        # Steering smoothing
        self.prev_steer = 0.0

        # Waypoint progress
        self.current_idx = 0

        self.waypoints = np.load('/home/nvidia/waypoints.npy')

        if len(self.waypoints.shape) != 2 or self.waypoints.shape[1] < 2:
            self.get_logger().error("Invalid waypoint file format.")
            sys.exit(1)

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints'
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self
        )


        self.cmd_pub = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )


        self.create_timer(0.05, self.control_loop)   # 20 Hz


        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

        self.get_logger().info('Path follower started.')

    def amcl_callback(self, msg):

        if self.amcl_converged:
            return

        cov_x = msg.pose.covariance[0]
        cov_y = msg.pose.covariance[7]

        self.get_logger().info(
            f'AMCL covariance x={cov_x:.4f}, y={cov_y:.4f}',
            throttle_duration_sec=1.0
        )

        if (
            cov_x < self.convergence_threshold and
            cov_y < self.convergence_threshold
        ):
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        if self.stable_counter >= self.required_stable_count:
            self.amcl_converged = True
            self.get_logger().info(
                'AMCL converged. Starting path following.'
            )

    def get_current_pose(self):

        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time(),
                timeout=Duration(seconds=0.2)
            )

            x = t.transform.translation.x
            y = t.transform.translation.y

            q = t.transform.rotation

            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            return x, y, yaw

        except Exception as e:
            self.get_logger().warn(
                f'TF unavailable: {str(e)}',
                throttle_duration_sec=2.0
            )
            return None

    def find_lookahead_point(self, cx, cy):

        for i in range(self.current_idx, len(self.waypoints)):

            wx = self.waypoints[i, 0]
            wy = self.waypoints[i, 1]

            d = math.hypot(wx - cx, wy - cy)

            if d >= self.lookahead_dist:
                self.current_idx = i
                return self.waypoints[i], i

        return None, self.current_idx

    def pure_pursuit(self, cx, cy, yaw, target):

        tx, ty = target[0], target[1]

        # Convert target into vehicle frame
        dx = math.cos(yaw) * (tx - cx) + math.sin(yaw) * (ty - cy)
        dy = -math.sin(yaw) * (tx - cx) + math.cos(yaw) * (ty - cy)

        ld = math.hypot(dx, dy)

        if ld < 0.001:
            return 0.0

        curvature = 2.0 * dy / (ld * ld)

        steer = math.atan(curvature * self.wheelbase)

        steer = max(-self.max_steer, min(self.max_steer, steer))

        return steer

    def compute_speed(self, steer):

        s = abs(steer)

        speed = self.max_speed - (s / self.max_steer) * (
            self.max_speed - self.min_speed
        )

        speed = max(self.min_speed, min(self.max_speed, speed))

        return speed


    def control_loop(self):

        # Wait for AMCL
        if not self.amcl_converged:
            self.send_command(0.0, 0.0)
            return

        pose = self.get_current_pose()

        if pose is None:
            self.send_command(0.0, 0.0)
            return

        cx, cy, yaw = pose

        target, idx = self.find_lookahead_point(cx, cy)

        # Path complete
        if target is None:
            self.get_logger().info('Path complete.')
            self.send_command(0.0, 0.0)
            return

        steer_raw = self.pure_pursuit(cx, cy, yaw, target)

        # Smooth steering
        steer = 0.7 * self.prev_steer + 0.3 * steer_raw
        self.prev_steer = steer

        speed = self.compute_speed(steer)

        self.get_logger().info(
            f'Pose({cx:.2f},{cy:.2f}) '
            f'WP:{idx}/{len(self.waypoints)} '
            f'Steer:{steer:.3f} '
            f'Speed:{speed:.2f}',
            throttle_duration_sec=0.5
        )

        self.send_command(steer, speed)

    def send_command(self, steer, throttle):

        msg = MotorCommands()
        msg.motor_names = [
            'steering_angle',
            'motor_throttle'
        ]
        msg.values = [
            float(steer),
            float(throttle)
        ]

        self.cmd_pub.publish(msg)


    def shutdown_handler(self, signum, frame):

        self.get_logger().info('Stopping vehicle...')

        for _ in range(10):
            self.send_command(0.0, 0.0)

        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)



def main(args=None):

    rclpy.init(args=args)

    node = PathFollower()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.send_command(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()