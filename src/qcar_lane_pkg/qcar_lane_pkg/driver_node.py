import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from qcar2_interfaces.msg import MotorCommands, BooleanLeds


class QCarDriverNode(Node):
    def __init__(self):
        super().__init__('qcar_driver_node')

        # Publish motor commands to qcar2_hardware (no direct HIL access)
        self.motor_pub = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )

        # Publish LED commands to qcar2_hardware
        self.led_pub = self.create_publisher(
            BooleanLeds,
            '/qcar2_led_cmd',
            10
        )

        # Subscribe to Twist commands from LaneKeepingNode
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        self.get_logger().info('QCarDriverNode started — using qcar2_hardware for HIL')

    def cmd_callback(self, msg):
        throttle = float(msg.linear.x)
        steering = float(msg.angular.z)

        # Forward motor commands to qcar2_hardware
        motor_cmd = MotorCommands()
        motor_cmd.motor_names = ['steering_angle', 'motor_throttle']
        motor_cmd.values      = [steering, throttle]
        self.motor_pub.publish(motor_cmd)

        # Set driving LEDs
        led_cmd = BooleanLeds()
        led_cmd.led_names = ['left_rear_signal', 'right_rear_signal']
        led_cmd.values    = [True, True]
        self.led_pub.publish(led_cmd)

    def destroy_node(self):
        # Send a stop command via topic — no direct hardware calls
        self.get_logger().info('QCarDriverNode shutting down — sending stop command')

        stop_cmd = MotorCommands()
        stop_cmd.motor_names = ['steering_angle', 'motor_throttle']
        stop_cmd.values      = [0.0, 0.0]
        self.motor_pub.publish(stop_cmd)

        # Turn off LEDs
        led_cmd = BooleanLeds()
        led_cmd.led_names = ['left_rear_signal', 'right_rear_signal']
        led_cmd.values    = [False, False]
        self.led_pub.publish(led_cmd)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QCarDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
