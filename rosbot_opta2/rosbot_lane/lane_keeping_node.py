#!/usr/bin/env python3
"""Lane keeping node - uses core modules."""

import rclpy
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

from rosbot_lane.core.config import load_config
from rosbot_lane.core.segmentation import segment_white
from rosbot_lane.core.detection import fit_lane_lines, apply_corridor_mask, eval_line_at_y, eval_poly_at_y
from rosbot_lane.core.geometry import compute_lane_error
from rosbot_lane.core.control import LaneController
from rosbot_lane.core.logging import LaneLogger
from rosbot_lane.core.debug import draw_debug_frame


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE_ROOT / 'config' / 'lane_params.yaml'
LOG_PATH = WORKSPACE_ROOT / 'lane_log.csv'


class LaneKeepingNode(Node):

    def __init__(self):
        super().__init__('lane_keeping_node')
        self.get_logger().info('Lane keeping node starting...')

        self.cfg = load_config(CONFIG_PATH)
        self.bridge = CvBridge()
        self._frame_num = 0
        self._no_detection_count = 0

        # Corridor reference lines
        self._mask_cl_line = None
        self._mask_rl_line = None

        # Core modules
        self.controller = LaneController(self.cfg.kp, self.cfg.kd)
        self.logger = LaneLogger(str(LOG_PATH))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(Image, self.cfg.topic_image, self._image_cb, sensor_qos)
        self._cmd_pub = self.create_publisher(TwistStamped, self.cfg.topic_cmd_vel, 10)

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        self._frame_num += 1
        strip_y = int(self.cfg.img_height * self.cfg.strip_top_fraction)
        strip = frame[strip_y:, :]
        sh = strip.shape[0]

        # Segmentation
        binary = segment_white(
            strip, self.cfg.hsv_low, self.cfg.hsv_high,
            self.cfg.morph_close_iterations
        )

        # Corridor mask
        binary = apply_corridor_mask(
            binary, self._mask_cl_line, self._mask_rl_line,
            self.cfg.corridor_px
        )

        # Detection
        cl_line, rl_line = fit_lane_lines(binary, self.cfg.img_width)

        # Compute 3 points per line
        cl_top_x = eval_line_at_y(cl_line, 0) if cl_line else None
        cl_mid_x = eval_line_at_y(cl_line, sh / 2) if cl_line else None
        cl_bot_x = eval_line_at_y(cl_line, sh) if cl_line else None
        rl_top_x = eval_poly_at_y(rl_line, 0) if rl_line is not None else None
        rl_mid_x = eval_poly_at_y(rl_line, sh / 2) if rl_line is not None else None
        rl_bot_x = eval_poly_at_y(rl_line, sh) if rl_line is not None else None

        # Update corridor references
        if cl_top_x is not None:
            self._mask_cl_line = cl_line
        if rl_top_x is not None:
            self._mask_rl_line = rl_line

        # Compute error
        error, lane_centre_x = compute_lane_error(
            cl_bot_x, rl_bot_x, self.cfg.img_width, self.cfg.lane_width_px
        )

        # Control
        if cl_bot_x is None or rl_bot_x is None:
            self._no_detection_count += 1
            if self._no_detection_count >= self.cfg.no_detection_stop_frames:
                self._stop()
                self.get_logger().warn('Both lane lines not detected', throttle_duration_sec=1.0)
        else:
            self._no_detection_count = 0
            self._drive(error)

        # Logging
        self.logger.log(
            self._frame_num,
            cl_top_x, cl_mid_x, cl_bot_x,
            rl_top_x, rl_mid_x, rl_bot_x,
            lane_centre_x, error
        )

        # Debug
        if self.cfg.show_debug:
            draw_debug_frame(
                frame, strip, binary, cl_line, rl_line,
                cl_top_x, cl_mid_x, cl_bot_x,
                rl_top_x, rl_mid_x, rl_bot_x,
                lane_centre_x, error, strip_y, sh,
                self.cfg.img_width, self.cfg.split_fraction
            )

    def _drive(self, error: float):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()

        if abs(error) > self.cfg.max_error_to_drive:
            self.get_logger().warn(
                f'Error too large ({error:.2f}), stopping.',
                throttle_duration_sec=1.0)
            self.controller.reset()
            self._cmd_pub.publish(twist)
            return

        twist.twist.linear.x = self.cfg.drive_speed
        twist.twist.angular.z = self.controller.compute(error)
        self._cmd_pub.publish(twist)

    def _stop(self):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node._stop()
        node.logger.close()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
