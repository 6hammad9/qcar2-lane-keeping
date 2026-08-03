import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hal.content.qcar import QCarRealSense
import time
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Publisher for camera images
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)

        # Initialize QCar RealSense camera
        self.qcarCam = QCarRealSense(mode='RGB', frameWidthRGB=640, frameHeightRGB=480,connect=True)

        # OpenCV bridge
        self.bridge = CvBridge()

        # Camera warmup
        self.get_logger().info("Warming up camera...")
        for _ in range(10):
            self.qcarCam.read_RGB()
            time.sleep(0.05)

        # Timer for periodic publishing (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.publish_frame)

    def publish_frame(self):
        self.qcarCam.read_RGB()
        frame = self.qcarCam.imageBufferRGB

        # Skip empty frames
        if frame is None or frame.sum() == 0:
            self.get_logger().warn("Empty frame, skipping publish")
            return

        # Convert OpenCV frame to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().debug("Published a valid frame")

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.qcarCam.terminate()
        except AttributeError:
            node.get_logger().warn("Camera not fully initialized")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()