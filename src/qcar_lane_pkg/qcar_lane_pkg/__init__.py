import rclpy
from rclpy.node import Node

class LaneKeepingNode(Node):
    def __init__(self):
        super().__init__('lane_keeping_node')
        self.get_logger().info("Lane Keeping Node has been started.")
        self.declare_parameter('manual_speed', 0.0)

def main(args=None):
    rclpy.init(args=args)
    lane_keeping_node = LaneKeepingNode()
    rclpy.spin(lane_keeping_node)
    lane_keeping_node.destroy_node()
    rclpy.shutdown()