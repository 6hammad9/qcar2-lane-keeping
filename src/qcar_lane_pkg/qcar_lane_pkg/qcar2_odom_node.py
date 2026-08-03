#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from qcar2_interfaces.msg import MotorCommands

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

# ── QCar2 physical constants ──────────────────────────────────────────────────
WHEEL_RADIUS           = 0.0325   # metres
WHEELBASE              = 0.256    # metres (front-to-rear axle)
ENCODER_COUNTS_PER_REV = 30809.0  # measured: one full wheel revolution


class QCar2OdomNode(Node):

    def __init__(self):
        super().__init__('qcar2_odom_node')

        # ── State ─────────────────────────────────────────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self.last_ticks    = None
        self.last_time     = None
        self.steering      = 0.0  # radians, updated from motor commands

        # ── Publishers ────────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(JointState,      '/qcar2_joint',           self.joint_cb, 10)
        self.create_subscription(MotorCommands,   '/qcar2_motor_speed_cmd', self.cmd_cb,   10)

        self.get_logger().info(
            f'QCar2 odom node started ({ENCODER_COUNTS_PER_REV:.0f} counts/rev).')

    # ── Steering angle from motor commands ────────────────────────────────────
    def cmd_cb(self, msg: MotorCommands):
        for name, value in zip(msg.motor_names, msg.values):
            if name == 'steering_angle':
                self.steering = value

    # ── Encoder ticks → odometry ──────────────────────────────────────────────
    def joint_cb(self, msg: JointState):
        if not msg.position:
            return

        current_ticks = msg.position[0]
        current_time  = self.get_clock().now().to_msg()

        # First reading — just store and return
        if self.last_ticks is None:
            self.last_ticks = current_ticks
            self.last_time  = current_time
            return

        # Time delta in seconds
        dt = (current_time.sec - self.last_time.sec) + \
             (current_time.nanosec - self.last_time.nanosec) * 1e-9

        if dt <= 0.0:
            return  # guard against duplicate / out-of-order messages

        # Encoder counts down when moving forward → negate
        delta_ticks = (current_ticks - self.last_ticks)
        delta_dist  = (delta_ticks / ENCODER_COUNTS_PER_REV) * (2.0 * math.pi * WHEEL_RADIUS)

        self.last_ticks = current_ticks
        self.last_time  = current_time

        # ── Ackermann kinematics ──────────────────────────────────────────────
        if abs(self.steering) > 1e-4:
            R           = WHEELBASE / math.tan(self.steering)
            delta_theta = delta_dist / R
        else:
            delta_theta = 0.0  # straight line

        # Midpoint integration
        mid_theta   = self.theta + delta_theta * 0.5
        self.x     += delta_dist * math.cos(mid_theta)
        self.y     += delta_dist * math.sin(mid_theta)
        self.theta += delta_theta

        # Normalise theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # ── Quaternion ────────────────────────────────────────────────────────
        q = quaternion_from_euler(0.0, 0.0, self.theta)  # returns [x, y, z, w]

        # ── Publish TF: odom → base_link ──────────────────────────────────────
        tf_msg                         = TransformStamped()
        tf_msg.header.stamp            = current_time
        tf_msg.header.frame_id         = 'odom'
        tf_msg.child_frame_id          = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x    = q[0]
        tf_msg.transform.rotation.y    = q[1]
        tf_msg.transform.rotation.z    = q[2]
        tf_msg.transform.rotation.w    = q[3]
        self.tf_broadcaster.sendTransform(tf_msg)

        # ── Publish /odom ──────────────────────────────────────────────────────
        odom                          = Odometry()
        odom.header.stamp             = current_time
        odom.header.frame_id          = 'odom'
        odom.child_frame_id           = 'base_link'

        odom.pose.pose.position.x     = self.x
        odom.pose.pose.position.y     = self.y
        odom.pose.pose.position.z     = 0.0
        odom.pose.pose.orientation.x  = q[0]
        odom.pose.pose.orientation.y  = q[1]
        odom.pose.pose.orientation.z  = q[2]
        odom.pose.pose.orientation.w  = q[3]

        # Pose covariance (row-major 6x6, indices: 0=x, 7=y, 35=yaw)
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw

        v = delta_dist  / dt
        w = delta_theta / dt
        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w
        odom.twist.covariance[0]   = 0.01
        odom.twist.covariance[35]  = 0.05

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = QCar2OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()