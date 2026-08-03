#!/usr/bin/env python3
"""
straight_lane_node.py — v2
===========================
Same node as before but _find_line_centroids is replaced with a proper
Hough-based line fitter. Each side gets one clean averaged line drawn
across the full strip. CL and RL dots sit on those fitted lines.

Changes from v1
---------------
  - _find_line_centroids()  →  replaced with Hough + line averaging
  - _draw_debug()           →  now draws the full extended lines on the strip
  - Everything else is identical — no other changes

What you should see now
-----------------------
  Two solid coloured lines drawn across the strip:
    Cyan  line = fitted centre dashed line  (CL)
    Yellow line = fitted right solid line   (RL)
  Both lines should sit stably ON the actual lane markings.
  Dots mark where each line intersects the bottom of the strip.
  Green vertical = lane centre (midpoint between the two dots).
  Red vertical   = image centre (where robot currently is).
"""

from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped


# ═════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

IMG_W = 768
IMG_H = 432

# Bottom strip — only this region is processed
# TUNE THIS: raise if strip still catches background, lower if road is cut off
STRIP_TOP_FRACTION = 0.72

# White segmentation
# TUNE THIS [2] value: lower (e.g. 150) if lines not detected
HSV_WHITE_LOW  = np.array([  0,   0, 170], dtype=np.uint8)
HSV_WHITE_HIGH = np.array([180,  45, 255], dtype=np.uint8)
MORPH_CLOSE_ITER = 2

# ── Hough parameters ─────────────────────────────────────────────────────────
# TUNE THIS: lower if dashes are being missed (e.g. 20), raise to reject noise
HOUGH_MIN_LINE_LENGTH = 30

# TUNE THIS: raise if dashes are split into too many tiny segments
HOUGH_MAX_LINE_GAP    = 15

# TUNE THIS: lower if lines are missed, raise if noise detected as lines
HOUGH_THRESHOLD       = 20

# Minimum Hough segments on one side to trust that detection
MIN_SEGMENTS_TO_TRUST = 1


# Width of the corridor kept visible around each line (pixels on each side).
# Everything to the left of (CL - CORRIDOR_PX) and to the right of
# (RL + CORRIDOR_PX) is blacked out in the binary mask before Hough runs.
# TUNE THIS: wider = more tolerant of line movement, narrower = less noise
CORRIDOR_PX = 40

# Filters out near-horizontal segments (road texture, shadows)
# TUNE THIS: lower if real lines are being filtered out
MIN_SLOPE = 0.3

# Left/right split as fraction of strip width
# Left of split  = centre dashes (CL)
# Right of split = right solid line (RL)
# TUNE THIS: adjust if CL is landing on the wrong line
SPLIT_FRACTION = 0.40

# ── Controller ────────────────────────────────────────────────────────────────
KP = 0.30
KD = 0.02

# Estimated lane width in pixels at strip level
# TUNE THIS: measure pixel distance between CL dot and RL dot when both visible
LANE_WIDTH_PX = 220

DRIVE_SPEED        = 0.20
MAX_ERROR_TO_DRIVE = 0.40

# Number of consecutive frames with no detection before stopping
NO_DETECTION_STOP_FRAMES = 10

# ── Topics ────────────────────────────────────────────────────────────────────
TOPIC_IMAGE   = '/oak/rgb/image_raw'
TOPIC_CMD_VEL = '/cmd_vel'

SHOW_DEBUG = True

# ═════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class StraightLaneNode(Node):

    def __init__(self):
        super().__init__('straight_lane_node')
        self.get_logger().info('Straight lane node v2 starting...')

        self.bridge      = CvBridge()
        self._prev_error = 0.0
        self._frame_num  = 0

        # Initializing the count of undetected lane frames
        self._no_detection_count = 0

        # First accepted line equations — used to build the dynamic ROI mask.
        # Set once on the first valid detection, then updated each accepted frame.
        self._mask_cl_line = None
        self._mask_rl_line = None

        # Open CSV log file — written every frame for post-run analysis
        self._log_file = open(WORKSPACE_ROOT / 'lane_log.csv', 'w')
        # CSV header: all 4 line points + geometry + controller output
        self._log_file.write(
            'frame,'
            'cl_top_x,cl_mid_x,cl_bot_x,'   # CL line: x at top, mid, bottom of strip
            'rl_top_x,rl_mid_x,rl_bot_x,'   # RL line: x at top, mid, bottom of strip
            'lane_centre_x,error\n'
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(Image, TOPIC_IMAGE, self._image_cb, sensor_qos)
        self._cmd_pub = self.create_publisher(TwistStamped, TOPIC_CMD_VEL, 10)

        self.get_logger().info(f'Subscribed to {TOPIC_IMAGE}')
        self.get_logger().info(f'Publishing  to {TOPIC_CMD_VEL}')

    # ─────────────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        # frame   = cv2.resize(frame, (IMG_W, IMG_H))
        strip_y = int(IMG_H * STRIP_TOP_FRACTION)
        strip   = frame[strip_y:, :]

        binary  = self._segment_white(strip)
        binary  = self._apply_corridor_mask(binary)

        cl_line, rl_line = self._fit_lane_lines(binary)

        self._frame_num += 1

        sh = binary.shape[0]  # height of the strip in pixels

        # Bottom point: where each fitted line crosses y = strip_h (closest to robot)
        cl_bot_x = self._eval_line_at_bottom(cl_line, sh) if cl_line else None
        rl_bot_x = self._eval_line_at_bottom(rl_line, sh) if rl_line else None

        # Top point: where each fitted line crosses y = 0 (farthest from robot)
        cl_top_x = (cl_line[0] * 0 + cl_line[1]) if cl_line else None
        rl_top_x = (rl_line[0] * 0 + rl_line[1]) if rl_line else None

        # Mid point: where each fitted line crosses y = strip_h / 2 (middle of strip)
        cl_mid_x = (cl_line[0] * (sh / 2) + cl_line[1]) if cl_line else None
        rl_mid_x = (rl_line[0] * (sh / 2) + rl_line[1]) if rl_line else None

        # Update corridor reference lines whenever points are accepted
        if cl_top_x is not None:
            self._mask_cl_line = cl_line
        if rl_top_x is not None:
            self._mask_rl_line = rl_line

        error, lane_centre_x = self._compute_error(cl_bot_x, rl_bot_x)

        # Calling the controls of the vehicle
        if cl_bot_x is None and rl_bot_x is None:
            self._no_detection_count += 1
            if self._no_detection_count >= NO_DETECTION_STOP_FRAMES:
                stop = TwistStamped()
                stop.header.stamp = self.get_clock().now().to_msg()
                self._cmd_pub.publish(stop)
        else:
            self._no_detection_count = 0        # reset counter when lines are visible
            self._control(error)


        # Write one CSV row per frame
        def v(x): return f'{x:.2f}' if x is not None else ''
        self._log_file.write(
            f'{self._frame_num},'
            f'{v(cl_top_x)},{v(cl_mid_x)},{v(cl_bot_x)},'
            f'{v(rl_top_x)},{v(rl_mid_x)},{v(rl_bot_x)},'
            f'{v(lane_centre_x)},{error:.4f}\n'
        )
        self._log_file.flush()

        if SHOW_DEBUG:
            self._draw_debug(frame, strip, binary,
                             cl_line, rl_line,
                             cl_top_x, cl_mid_x, cl_bot_x,
                             rl_top_x, rl_mid_x, rl_bot_x,
                             lane_centre_x, error, strip_y)

    # ─────────────────────────────────────────────────────────────────────────
    #  SEGMENTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _segment_white(self, strip_bgr: np.ndarray) -> np.ndarray:
        hsv    = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv, HSV_WHITE_LOW, HSV_WHITE_HIGH)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                  iterations=MORPH_CLOSE_ITER)
        return mask


    def _apply_corridor_mask(self, binary: np.ndarray) -> np.ndarray:
        """
        Blacks out everything outside a corridor around CL and RL.

        The mask keeps three regions visible:
        - CORRIDOR_PX pixels to the left and right of the CL line
        - Everything between CL and RL (the lane interior)
        - CORRIDOR_PX pixels to the left and right of the RL line

        Everything outside those regions is set to black before Hough runs,
        so noise far from the lines is never seen by the detector.

        Only activates once the first valid detection has been stored in
        self._mask_cl_line and self._mask_rl_line. Until then, the full
        binary mask is passed through unchanged.
        """
        if self._mask_cl_line is None or self._mask_rl_line is None:
            return binary   # first frame — no corridor yet, pass full mask

        h, w = binary.shape
        corridor = np.zeros_like(binary)   # start fully black

        # For each row y, compute the x position of CL and RL at that row
        for y in range(h):
            cl_x = int(self._mask_cl_line[0] * y + self._mask_cl_line[1])
            rl_x = int(self._mask_rl_line[0] * y + self._mask_rl_line[1])

            # Left corridor: CORRIDOR_PX to the left of CL
            left_start  = max(0, cl_x - CORRIDOR_PX)
            # Right corridor: CORRIDOR_PX to the right of RL
            right_end   = min(w, rl_x + CORRIDOR_PX)

            # Keep everything from left_start to right_end visible
            # This includes: left corridor + lane interior + right corridor
            corridor[y, left_start:right_end] = 255

        # AND with the original binary — only white pixels inside corridor survive
        return cv2.bitwise_and(binary, corridor)

    # ─────────────────────────────────────────────────────────────────────────
    #  HOUGH LINE FITTING
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_lane_lines(self, binary: np.ndarray):
        """
        Detect all Hough segments, convert each to a (slope, intercept) line,
        then pick the best CL and RL by proximity to image center.

        Strategy
        --------
        Rather than splitting the frame at a fixed fraction and averaging
        everything on each side, we:
          1. Find ALL valid line segments across the full strip.
          2. Evaluate each line's x position at the bottom of the strip.
          3. Split by image center (IMG_W / 2):
               - Lines whose bottom-x is LEFT  of center → CL candidates
               - Lines whose bottom-x is RIGHT of center → RL candidates
          4. From CL candidates pick the RIGHTMOST  (closest to center).
             From RL candidates pick the LEFTMOST   (closest to center).

        Why this beats the old split-fraction approach
        -----------------------------------------------
        Noise on the far right of the frame is a RL candidate, but it will
        never be the leftmost RL candidate — the actual solid line is closer
        to center. So noise is automatically discarded without any extra filter.
        Same logic applies to spurious detections on the far left.

        Line representation: x = slope * y + intercept
        We use x=f(y) because lane lines are near-vertical.

        Returns (cl_line, rl_line) — each is (slope, intercept) or None.
        """
        h, w = binary.shape
        cx   = IMG_W / 2.0   # image centre x — natural left/right divider

        segments = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        # Each entry: (bottom_x, slope, intercept)
        # bottom_x is where this line crosses y = strip_h — used for selection
        left_candidates  = []   # lines whose bottom_x is left  of center → CL
        right_candidates = []   # lines whose bottom_x is right of center → RL

        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = seg[0]

                dy = float(y2 - y1)
                dx = float(x2 - x1)

                # Reject near-horizontal segments (road texture, shadows)
                if abs(dx) < 1e-6 or abs(dy) < 1e-6:
                    continue
                if abs(dy / dx) < MIN_SLOPE:
                    continue

                # Fit line: x = slope * y + intercept
                slope     = dx / dy
                intercept = x1 - slope * y1

                # Evaluate x at the bottom of the strip (y = h)
                # This is the reference position used for left/right assignment
                bottom_x = slope * h + intercept

                if bottom_x < cx:
                    left_candidates.append((bottom_x, slope, intercept))
                else:
                    right_candidates.append((bottom_x, slope, intercept))

        # CL = rightmost candidate on the left  (largest bottom_x < center)
        # RL = leftmost  candidate on the right (smallest bottom_x > center)
        # If multiple segments are very close, average them — same robustness
        # as before but now only among the best-positioned candidates.

        cl_line = None
        if left_candidates:
            # Sort by bottom_x descending, take the rightmost (closest to center)
            left_candidates.sort(key=lambda c: c[0], reverse=True)
            # Group all candidates within 20px of the best one and average them
            best_x = left_candidates[0][0]
            group  = [(s, i) for (bx, s, i) in left_candidates
                      if abs(bx - best_x) < 20]
            cl_line = self._average_lines(group)

        rl_line = None
        if right_candidates:
            # Sort by bottom_x ascending, take the leftmost (closest to center)
            right_candidates.sort(key=lambda c: c[0])
            best_x = right_candidates[0][0]
            group  = [(s, i) for (bx, s, i) in right_candidates
                      if abs(bx - best_x) < 20]
            rl_line = self._average_lines(group)

        return cl_line, rl_line

    def _average_lines(self, params: list):
        """Average a list of (slope, intercept) pairs. Returns None if empty."""
        if len(params) < MIN_SEGMENTS_TO_TRUST:
            return None
        return (float(np.mean([p[0] for p in params])),
                float(np.mean([p[1] for p in params])))

    def _eval_line_at_bottom(self, line, strip_h: int) -> float:
        """x position where fitted line crosses the bottom of the strip."""
        slope, intercept = line
        return slope * strip_h + intercept

    def _get_line_endpoints(self, line, strip_h: int):
        """Top and bottom endpoints of a fitted line for cv2.line drawing."""
        slope, intercept = line
        x_top = int(slope * 0       + intercept)
        x_bot = int(slope * strip_h + intercept)
        return (x_top, 0), (x_bot, strip_h)

    # ─────────────────────────────────────────────────────────────────────────
    #  GEOMETRY
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_error(self, cl_x, rl_x):
        """
        Normalised lateral error [-1, +1].
        Positive = robot is left of lane centre → steer right.
        """
        if cl_x is not None and rl_x is not None:
            lane_centre_x = (cl_x + rl_x) / 2.0

        elif rl_x is not None:
            lane_centre_x = rl_x - LANE_WIDTH_PX / 2.0

        elif cl_x is not None:
            lane_centre_x = cl_x + LANE_WIDTH_PX / 2.0

        else:
            self.get_logger().warn(
                'No lane lines detected', throttle_duration_sec=1.0)
            return 0.0, IMG_W / 2.0

        error = (IMG_W / 2.0 - lane_centre_x) / (IMG_W / 2.0)
        return float(np.clip(error, -1.0, 1.0)), float(lane_centre_x)

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    def _control(self, error: float):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()

        if abs(error) > MAX_ERROR_TO_DRIVE:
            self.get_logger().warn(
                f'Error too large ({error:.2f}), stopping.',
                throttle_duration_sec=1.0)
            self._cmd_pub.publish(twist)
            self._prev_error = 0.0
            return

        d_error          = error - self._prev_error
        angular_z        = KP * error + KD * d_error
        self._prev_error = error

        twist.twist.linear.x  = DRIVE_SPEED
        twist.twist.angular.z = float(np.clip(angular_z, -1.5, 1.5))
        self._cmd_pub.publish(twist)


    # ─────────────────────────────────────────────────────────────────────────
    #  DEBUG
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_debug(self, frame, strip, binary,
                    cl_line, rl_line,
                    cl_top_x, cl_mid_x, cl_bot_x,
                    rl_top_x, rl_mid_x, rl_bot_x,
                    lane_centre_x, error, strip_y):
        """
        Lane Node window:
          Cyan  line + 3 dots = CL (centre dashes)
            - CL_T: dot at TOP    of strip  (y=0,      farthest from robot)
            - CL_M: dot at MIDDLE of strip  (y=sh/2)
            - CL_B: dot at BOTTOM of strip  (y=sh,     closest to robot)
          Yellow line + 3 dots = RL (right solid line)
            - RL_T, RL_M, RL_B same positions as CL
          Each dot shows its label + (x, y) pixel coordinates on screen.
          Green vertical = lane centre (midpoint of cl_bot_x and rl_bot_x)
          Red   vertical = image centre (where robot currently is)

        Top-left HUD panel shows all 6 point coordinates for easy reading.

        Strip | Mask window:
          Left  = colour strip with lines and dots drawn
          Right = binary mask with left/right split shown as orange line
        """
        strip_vis = strip.copy()
        sh        = strip_vis.shape[0]

        # Helper: draw a dot + "LABEL (x, y)" text next to it
        def draw_point(img, x, y, label, color):
            if x is None:
                return
            ix, iy = int(x), int(y)
            cv2.circle(img, (ix, iy), 8, color, -1)
            cv2.putText(img, f'{label} ({ix},{iy})',
                        (ix + 10, iy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ── CL line (cyan) ────────────────────────────────────────────────────
        if cl_line:
            pt_top, pt_bot = self._get_line_endpoints(cl_line, sh)
            cv2.line(strip_vis, pt_top, pt_bot, (255, 255, 0), 2)
            draw_point(strip_vis, cl_top_x, 5,      'CL_T', (255, 255, 0))
            draw_point(strip_vis, cl_mid_x, sh // 2,'CL_M', (255, 255, 0))
            draw_point(strip_vis, cl_bot_x, sh - 5, 'CL_B', (255, 255, 0))

        # ── RL line (yellow) ──────────────────────────────────────────────────
        if rl_line:
            pt_top, pt_bot = self._get_line_endpoints(rl_line, sh)
            cv2.line(strip_vis, pt_top, pt_bot, (0, 255, 255), 2)
            draw_point(strip_vis, rl_top_x, 5,      'RL_T', (0, 255, 255))
            draw_point(strip_vis, rl_mid_x, sh // 2,'RL_M', (0, 255, 255))
            draw_point(strip_vis, rl_bot_x, sh - 5, 'RL_B', (0, 255, 255))

        # ── Lane centre (green) and image centre (red) ────────────────────────
        if lane_centre_x is not None:
            cv2.line(strip_vis, (int(lane_centre_x), 0),
                     (int(lane_centre_x), sh), (0, 255, 0), 2)
        cv2.line(strip_vis, (IMG_W // 2, 0),
                 (IMG_W // 2, sh), (0, 0, 255), 2)

        # Paste strip back onto full frame for Lane Node window
        vis = frame.copy()
        vis[strip_y:, :] = strip_vis
        cv2.line(vis, (0, strip_y), (IMG_W, strip_y), (128, 128, 0), 1)

        # ── HUD: all 6 point coordinates in top-left corner ───────────────────
        def fmt(x, y): return f'({int(x)},{int(y)})' if x is not None else '(--,--)'
        hud_lines = [
            (f'CL_T {fmt(cl_top_x, 0)}',       (255, 255,   0)),
            (f'CL_M {fmt(cl_mid_x, sh//2)}',    (255, 255,   0)),
            (f'CL_B {fmt(cl_bot_x, sh-5)}',     (255, 255,   0)),
            (f'RL_T {fmt(rl_top_x, 0)}',        (  0, 255, 255)),
            (f'RL_M {fmt(rl_mid_x, sh//2)}',    (  0, 255, 255)),
            (f'RL_B {fmt(rl_bot_x, sh-5)}',     (  0, 255, 255)),
            (f'Error: {error:+.3f}',             (255, 255, 255)),
        ]
        for i, (text, col) in enumerate(hud_lines):
            cv2.putText(vis, text, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        # ── Status line (BOTH / ONE / NO LINES) ───────────────────────────────
        both   = cl_line is not None and rl_line is not None
        one    = (cl_line is not None) != (rl_line is not None)
        status = 'BOTH LINES' if both else ('ONE LINE' if one else 'NO LINES')
        s_col  = (0, 255, 0) if both else ((0, 165, 255) if one else (0, 0, 255))
        cv2.putText(vis, status, (10, 25 + len(hud_lines) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)

        # Binary mask with split line shown
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        split_px     = int(binary.shape[1] * SPLIT_FRACTION)
        cv2.line(binary_color, (split_px, 0),
                 (split_px, binary.shape[0]), (0, 100, 255), 1)

        strip_resized  = cv2.resize(strip_vis,    (IMG_W // 2, 160))
        binary_resized = cv2.resize(binary_color, (IMG_W // 2, 160))
        bottom_row     = np.hstack([strip_resized, binary_resized])

        cv2.imshow('Lane Node',    vis)
        cv2.imshow('Strip | Mask', bottom_row)
        cv2.waitKey(1)


# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = StraightLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        node._cmd_pub.publish(stop)
        node._log_file.close()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
