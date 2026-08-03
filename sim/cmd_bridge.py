#!/usr/bin/env python3
"""Sim stand-in for the QCar's nav2_qcar2_converter.

path_mpc publishes /cmd_vel_nav with angular.z = STEERING ANGLE (rad),
because publish_steering_angle=True and the real car's
`nav2_qcar2_converter` expects that. Gazebo's vehicle plugin expects
angular.z = YAW RATE. Convert with the bicycle model:

    omega = v / L * tan(delta)

so the simulated car steers the same way the physical one does.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdBridge(Node):
    def __init__(self):
        super().__init__("qcar_cmd_bridge")
        self.declare_parameter("wheelbase", 0.256)
        self.declare_parameter("in_topic", "/cmd_vel_nav")
        self.declare_parameter("out_topic", "/model/qcar2/cmd_vel")
        self.declare_parameter("command_timeout_sec", 0.5)
        self.L = float(self.get_parameter("wheelbase").value)
        self.command_timeout = float(
            self.get_parameter("command_timeout_sec").value
        )
        in_topic = str(self.get_parameter("in_topic").value)
        out_topic = str(self.get_parameter("out_topic").value)

        self.pub = self.create_publisher(Twist, out_topic, 10)
        self.create_subscription(Twist, in_topic, self.cb, 10)
        self.n = 0
        self.last_command_time = None
        self.timeout_active = False
        self.create_timer(0.05, self.watchdog)
        self.get_logger().info(
            f"cmd bridge: {in_topic} (steer angle) -> {out_topic} "
            f"(yaw rate), L={self.L}"
        )

    def cb(self, msg):
        out = Twist()
        out.linear.x = msg.linear.x
        out.angular.z = msg.linear.x / self.L * math.tan(msg.angular.z)
        self.pub.publish(out)
        self.last_command_time = self.get_clock().now()
        self.timeout_active = False
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info(
                f"v={msg.linear.x:.2f} delta={msg.angular.z:.3f} rad "
                f"-> omega={out.angular.z:.3f} rad/s",
                throttle_duration_sec=2.0,
            )

    def publish_stop(self):
        self.pub.publish(Twist())

    def watchdog(self):
        if self.last_command_time is None or self.timeout_active:
            return
        age = (
            self.get_clock().now() - self.last_command_time
        ).nanoseconds * 1e-9
        if age < 0.0:
            # World reset: the previous command belongs to the old timeline.
            self.publish_stop()
            self.timeout_active = True
        elif age >= self.command_timeout:
            self.publish_stop()
            self.timeout_active = True
            self.get_logger().warn(
                f"command timeout after {age:.2f} s; publishing zero velocity"
            )


def main():
    rclpy.init()
    node = CmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
