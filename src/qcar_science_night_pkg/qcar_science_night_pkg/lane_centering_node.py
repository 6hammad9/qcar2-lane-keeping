#!/usr/bin/env python3

import os
from pathlib import Path

import cv2
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from qcar_science_night_pkg.lane_utils import lane_reference_offset


def dict_to_obj(d):
    if isinstance(d, dict):
        return type("Config", (), {k: dict_to_obj(v) for k, v in d.items()})()
    return d


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return dict_to_obj(yaml.safe_load(f))


def resolve_config_path(config_override=""):
    """Resolve config from an installed package or this source checkout."""
    if config_override:
        return os.path.abspath(os.path.expanduser(config_override))

    try:
        share = get_package_share_directory("qcar_science_night_pkg")
        return os.path.join(share, "config", "lane_params.yaml")
    except PackageNotFoundError:
        return str(Path(__file__).resolve().parents[1] / "config" / "lane_params.yaml")


class LaneCenteringNode(Node):

    def __init__(self):
        super().__init__("lane_centering_node")

        self.declare_parameter("config_file", "")
        self.declare_parameter("image_timeout_sec", 0.75)
        config_path = resolve_config_path(
            str(self.get_parameter("config_file").value)
        )
        self.cfg = load_config(config_path)
        self.image_timeout_sec = float(
            self.get_parameter("image_timeout_sec").value
        )
        if self.image_timeout_sec <= 0.0:
            raise ValueError("image_timeout_sec must be positive")

        self.bridge = CvBridge()
        self.filtered_offset = 0.0
        self.last_image_time = None
        self.last_valid = False
        self.frame_count = 0
        self.frame_skip = max(1, int(self.cfg.performance.frame_skip))

        self.prev_cl_line = None
        self.prev_rl_poly = None

        self.offset_pub = self.create_publisher(Float32, "/lane_center_offset", 10)
        self.valid_pub  = self.create_publisher(Bool,    "/lane_center_valid",  10)
        self.debug_pub  = self.create_publisher(Image,   "/lane_debug",         10)
        self.mask_pub   = self.create_publisher(Image,   "/lane_mask",          10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.image_sub = self.create_subscription(
            Image,
            self.cfg.image.topic,
            self.image_callback,
            sensor_qos
        )
        self.watchdog_timer = self.create_timer(0.1, self.image_watchdog)

        self.get_logger().info(
            f"Lane centering node ready: topic={self.cfg.image.topic}, "
            f"config={config_path}, process_every={self.frame_skip} frame(s)"
        )

    def publish_lane_state(self, offset, valid):
        self.offset_pub.publish(Float32(data=float(offset)))
        self.valid_pub.publish(Bool(data=bool(valid)))
        self.last_valid = bool(valid)

    def invalidate_lane(self):
        self.filtered_offset = 0.0
        self.publish_lane_state(0.0, False)

    def image_watchdog(self):
        if self.last_image_time is None:
            self.publish_lane_state(0.0, False)
            return

        age = (
            self.get_clock().now() - self.last_image_time
        ).nanoseconds * 1e-9
        if age > self.image_timeout_sec:
            if self.last_valid:
                self.get_logger().warn(
                    f"Camera stale for {age:.2f}s; disabling lane correction",
                    throttle_duration_sec=1.0,
                )
            self.invalidate_lane()

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.frame_skip:
            return

        frame = self.safe_convert_image(msg)
        if frame is None:
            self.invalidate_lane()
            return

        frame = cv2.resize(
            frame,
            (
                int(self.cfg.performance.resize_width),
                int(self.cfg.performance.resize_height)
            ),
            interpolation=cv2.INTER_AREA
        )

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            self.invalidate_lane()
            return

        # --- FIX 1: two-sided crop (top + bottom) ---
        strip_y_top = int(h * self.cfg.image.strip_top_fraction)
        strip_y_bot = int(h * self.cfg.image.strip_bottom_fraction)
        strip_y_bot = min(strip_y_bot, h)           # clamp to frame height

        strip = frame[strip_y_top:strip_y_bot, :]

        if strip is None or strip.size == 0:
            self.invalidate_lane()
            return

        # Only a successfully decoded, geometrically usable image refreshes
        # the watchdog. A stream of malformed images must not keep an old
        # lane correction alive.
        self.last_image_time = self.get_clock().now()

        binary = self.segment_white(strip)

        binary_masked = self.apply_corridor_mask(
            binary,
            self.prev_cl_line,
            self.prev_rl_poly,
            self.cfg.detection.corridor_px
        )

        cl_line, rl_poly = self.fit_lane_lines(binary_masked, w)

        sh = strip.shape[0]

        sample_y = max(0, sh - 1)
        cl_bot_x = (
            self.eval_line_at_y(cl_line, sample_y)
            if cl_line is not None else None
        )
        rl_bot_x = (
            self.eval_poly_at_y(rl_poly, sample_y)
            if rl_poly is not None else None
        )

        def in_image(value):
            return value is not None and np.isfinite(value) and 0.0 <= value < w

        if not in_image(cl_bot_x):
            cl_line = None
            cl_bot_x = None
            self.prev_cl_line = None
        if not in_image(rl_bot_x):
            rl_poly = None
            rl_bot_x = None
            self.prev_rl_poly = None

        if cl_bot_x is not None and rl_bot_x is not None:
            detected_width = rl_bot_x - cl_bot_x
            min_width = float(getattr(
                self.cfg.detection, "min_lane_width_px", 0.20 * w
            ))
            max_width = float(getattr(
                self.cfg.detection, "max_lane_width_px", 0.95 * w
            ))
            if not min_width <= detected_width <= max_width:
                cl_line = None
                rl_poly = None
                cl_bot_x = None
                rl_bot_x = None
                self.prev_cl_line = None
                self.prev_rl_poly = None

        if cl_line is not None:
            self.prev_cl_line = cl_line
        if rl_poly is not None:
            self.prev_rl_poly = rl_poly

        lane_center_x    = w / 2.0
        corrected_center_x = lane_center_x
        valid = False
        mode  = "NONE"

        if cl_bot_x is not None and rl_bot_x is not None:
            lane_center_x = (cl_bot_x + rl_bot_x) / 2.0
            valid = True
            mode  = "CL+RL"

        elif (
            rl_bot_x is not None
            and bool(getattr(self.cfg.detection, "allow_single_boundary", True))
        ):
            lane_center_x = rl_bot_x - self.cfg.detection.lane_width_px / 2.0
            valid = True
            mode  = "RL_ONLY"

        elif (
            cl_bot_x is not None
            and bool(getattr(self.cfg.detection, "allow_single_boundary", True))
        ):
            lane_center_x = cl_bot_x + self.cfg.detection.lane_width_px / 2.0
            valid = True
            mode  = "CL_ONLY"

        if valid:
            camera_offset_px = (
                self.cfg.vehicle.camera_offset_m
                / self.cfg.vehicle.lane_width_m
            ) * self.cfg.detection.lane_width_px

            corrected_center_x = lane_center_x + camera_offset_px

            pixel_error      = corrected_center_x - (w / 2.0)
            normalized_error = pixel_error / (w / 2.0)

            raw_offset = lane_reference_offset(
                normalized_error,
                self.cfg.control.gain,
            )
            raw_offset = float(np.clip(
                raw_offset,
                -self.cfg.control.max_offset,
                self.cfg.control.max_offset
            ))

            if abs(raw_offset) < self.cfg.control.deadband:
                raw_offset = 0.0

            self.filtered_offset = (
                (1.0 - self.cfg.control.filter_alpha) * self.filtered_offset
                + self.cfg.control.filter_alpha * raw_offset
            )

        else:
            self.filtered_offset *= 0.90

        self.publish_lane_state(self.filtered_offset, valid)

        if self.cfg.debug.publish:
            debug_img = self.draw_debug(
                strip,
                binary_masked,
                cl_line,
                rl_poly,
                lane_center_x,
                corrected_center_x,
                self.filtered_offset,
                valid,
                mode
            )
            self.publish_debug(debug_img, binary_masked, msg.header)

        self.get_logger().info(
            f"valid={valid}, mode={mode}, cl={cl_bot_x}, rl={rl_bot_x}, "
            f"offset={self.filtered_offset:.3f}",
            throttle_duration_sec=1.0
        )

    # ------------------------------------------------------------------
    def safe_convert_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return None

        if frame is None:
            return None

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame.copy()
        else:
            self.get_logger().warn(f"Unsupported image shape: {frame.shape}")
            return None

        if frame.size == 0:
            return None

        return frame

    # ------------------------------------------------------------------
    def segment_white(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_low  = np.array(self.cfg.segmentation.hsv_low,  dtype=np.uint8)
        hsv_high = np.array(self.cfg.segmentation.hsv_high, dtype=np.uint8)

        binary = cv2.inRange(hsv, hsv_low, hsv_high)

        kernel = np.ones((5, 5), np.uint8)
        for _ in range(int(self.cfg.segmentation.morph_close_iterations)):
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    # ------------------------------------------------------------------
    def fit_lane_lines(self, binary, img_width):
        segments = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=int(self.cfg.hough.threshold),
            minLineLength=int(self.cfg.hough.min_line_length),
            maxLineGap=int(self.cfg.hough.max_line_gap)
        )

        cl_line = self.detect_cl(binary, segments, img_width)
        rl_poly = self.detect_rl(binary, segments, img_width)

        return cl_line, rl_poly

    # ------------------------------------------------------------------
    def detect_cl(self, binary, segments, img_width):
        if segments is None:
            return None

        h, _ = binary.shape
        cx   = img_width / 2.0
        candidates = []

        for seg in segments:
            x1, y1, x2, y2 = seg[0]
            dy = float(y2 - y1)
            dx = float(x2 - x1)

            if abs(dy) < 1e-6:
                continue
            if abs(dy) < self.cfg.hough.min_slope * max(abs(dx), 1e-6):
                continue

            slope     = dx / dy
            intercept = x1 - slope * y1
            bottom_x  = slope * h + intercept

            if bottom_x < cx:
                candidates.append((bottom_x, slope, intercept))

        if not candidates:
            return None

        # Take the cluster closest to the centre line (rightmost left candidate)
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_x = candidates[0][0]
        group  = [
            (s, i)
            for bx, s, i in candidates
            if abs(bx - best_x) < 40
        ]

        if not group:
            return None

        avg_slope     = float(np.mean([p[0] for p in group]))
        avg_intercept = float(np.mean([p[1] for p in group]))
        return avg_slope, avg_intercept

    # ------------------------------------------------------------------
    def detect_rl(self, binary, segments, img_width):
        if segments is None:
            return None

        h, _ = binary.shape
        cx   = img_width / 2.0
        candidates = []

        for seg in segments:
            x1, y1, x2, y2 = seg[0]
            dy = float(y2 - y1)
            dx = float(x2 - x1)

            if abs(dy) < 1e-6:
                continue
            if abs(dy) < self.cfg.hough.min_slope * max(abs(dx), 1e-6):
                continue

            slope     = dx / dy
            intercept = x1 - slope * y1
            bottom_x  = slope * h + intercept

            if bottom_x >= cx:
                candidates.append(bottom_x)

        if not candidates:
            return None

        # --- FIX 2: cluster candidates, use median of nearest cluster ---
        candidates.sort()
        best_x  = candidates[0]                          # leftmost right-side
        cluster = [x for x in candidates if abs(x - best_x) < 40]
        start_x = int(np.median(cluster))

        return self.sliding_window(binary, start_x)

    # ------------------------------------------------------------------
    def sliding_window(
        self,
        binary,
        start_x,
        n_windows=20,
        margin=120,              # FIX 3: wider margin for dashed lines
        min_pixels_recenter=20,  # FIX 4: lower threshold for dashes
        min_pixels_to_fit=25     # FIX 5: lower threshold for dashes
    ):
        h, w = binary.shape
        nzy, nzx = binary.nonzero()

        current_x    = start_x
        window_h     = max(1, h // n_windows)
        pixel_indices = []

        for win in range(n_windows):
            y_lo = h - (win + 1) * window_h
            y_hi = h - win * window_h
            x_lo = max(0, current_x - margin)
            x_hi = min(w, current_x + margin)

            inds = np.where(
                (nzy >= y_lo) & (nzy < y_hi)
                & (nzx >= x_lo) & (nzx < x_hi)
            )[0]

            pixel_indices.append(inds)

            if len(inds) > min_pixels_recenter:
                current_x = int(np.mean(nzx[inds]))

        if not pixel_indices:
            return None

        all_inds = np.concatenate(pixel_indices)
        if len(all_inds) < min_pixels_to_fit:
            return None

        return np.polyfit(nzy[all_inds], nzx[all_inds], 2)

    # ------------------------------------------------------------------
    def apply_corridor_mask(self, binary, cl_line, rl_poly, corridor_px):
        if cl_line is None or rl_poly is None:
            return binary

        h, w   = binary.shape
        corridor = np.zeros_like(binary)

        for y in range(h):
            cl_x = int(self.eval_line_at_y(cl_line, y))
            rl_x = int(self.eval_poly_at_y(rl_poly, y))

            left_start = max(0, cl_x - corridor_px)
            right_end  = min(w, rl_x + corridor_px)

            if right_end > left_start:
                corridor[y, left_start:right_end] = 255

        return cv2.bitwise_and(binary, corridor)

    # ------------------------------------------------------------------
    @staticmethod
    def eval_line_at_y(line, y):
        return line[0] * y + line[1]

    @staticmethod
    def eval_poly_at_y(poly, y):
        return poly[0] * y ** 2 + poly[1] * y + poly[2]

    # ------------------------------------------------------------------
    def draw_debug(
        self,
        strip,
        binary,
        cl_line,
        rl_poly,
        lane_center_x,
        corrected_center_x,
        offset,
        valid,
        mode
    ):
        vis  = strip.copy()
        h, w = vis.shape[:2]

        for y in range(0, h, 5):
            if cl_line is not None:
                x = int(self.eval_line_at_y(cl_line, y))
                if 0 <= x < w:
                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)

            if rl_poly is not None:
                x = int(self.eval_poly_at_y(rl_poly, y))
                if 0 <= x < w:
                    cv2.circle(vis, (x, y), 2, (0, 165, 255), -1)   # orange = right

        # image centre (red)
        cv2.line(vis, (int(w / 2), 0), (int(w / 2), h), (0, 0, 255), 2)

        if valid:
            # lane centre (green)
            cv2.line(vis, (int(lane_center_x), 0), (int(lane_center_x), h), (0, 255, 0), 2)
            # corrected centre (cyan)
            cv2.line(vis, (int(corrected_center_x), 0), (int(corrected_center_x), h), (0, 255, 255), 2)

        cv2.putText(vis, f"valid={valid}",      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if valid else (0, 0, 255), 2)
        cv2.putText(vis, f"mode={mode}",        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis, f"offset={offset:.3f}",(10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return vis

    # ------------------------------------------------------------------
    def publish_debug(self, debug_img, mask, header):
        try:
            debug_msg        = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            mask_msg         = self.bridge.cv2_to_imgmsg(mask,      encoding="mono8")
            debug_msg.header = header
            mask_msg.header  = header
            self.debug_pub.publish(debug_msg)
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().warn(f"Debug publish failed: {e}")


# ----------------------------------------------------------------------
def main():
    rclpy.init()
    node = LaneCenteringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
