#!/usr/bin/env python3
"""Relay TF from namespaced topic to global."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class TFRelay(Node):
    def __init__(self):
        super().__init__('tf_relay')
        
        # QoS for TF
        tf_qos = QoSProfile(depth=100)
        tf_qos.durability = DurabilityPolicy.VOLATILE
        tf_qos.reliability = ReliabilityPolicy.RELIABLE
        
        # QoS for TF static
        tf_static_qos = QoSProfile(depth=100)
        tf_static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        tf_static_qos.reliability = ReliabilityPolicy.RELIABLE
        
        # Subscribers
        self.create_subscription(TFMessage, '/rosbot3/tf', self._tf_cb, tf_qos)
        self.create_subscription(TFMessage, '/rosbot3/tf_static', self._tf_static_cb, tf_static_qos)
        
        # Publishers
        self._tf_pub = self.create_publisher(TFMessage, '/tf', tf_qos)
        self._tf_static_pub = self.create_publisher(TFMessage, '/tf_static', tf_static_qos)
        
        self.get_logger().info('TF Relay started: /rosbot3/tf -> /tf')

    def _tf_cb(self, msg):
        self._tf_pub.publish(msg)

    def _tf_static_cb(self, msg):
        self._tf_static_pub.publish(msg)


def main():
    rclpy.init()
    node = TFRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
