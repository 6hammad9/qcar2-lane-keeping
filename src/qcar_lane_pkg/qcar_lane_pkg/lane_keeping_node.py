# qcar2_lane_ros/lane_keeping_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from hal.content.qcar_functions import SpeedController
from pal.utilities.math import wrap_to_pi
from qcar2_interfaces.msg import MotorCommands
from qcar2_interfaces.msg import BooleanLeds


TRAILER_ATTACHED = False
L_CAR            =  0.256
L_TRAILER        = 0.20
K_TRAILER        = 0.8
JACKKNIFE_LIMIT  = np.deg2rad(35)
V_REF_TRAILER    = 0.6



PIXEL_LANE_WIDTH  = 390
LANE_WIDTH_MIN_PX = 300
LANE_WIDTH_MAX_PX = 450


LOST_FRAMES_THRESHOLD = 12
COAST_SPEED_FRACTION  = 0.4

# Minimum inlier pixels to trust a fit (left is lower — near BEV edge)
MIN_PIX_LEFT  = 50
MIN_PIX_RIGHT = 80

# Steering PD gains
KP_BASE = 0.55
KD      = 0.20

# Right lane stabilisation thresholds (pixels)
LEFT_STABLE_THRESH = 20
RIGHT_JUMP_THRESH  = 40


class TrailerModel:
    def __init__(self):
        self.psi = 0.0

    def update(self, v, delta, dt):
        dpsi = (
              (v / L_TRAILER) * np.sin(self.psi)
            - (v / L_CAR)     * np.cos(self.psi) * np.tan(delta)
        )
        self.psi = wrap_to_pi(self.psi + dpsi * dt)
        return self.psi

    def steering_correction(self):
        return -K_TRAILER * self.psi

    def speed_limit(self, v_ref):
        ratio = float(np.clip(abs(self.psi) / JACKKNIFE_LIMIT, 0.0, 1.0))
        return v_ref * (1.0 - 0.5 * ratio)



class LaneKeepingNode(Node):
    def __init__(self):
        super().__init__('lane_keeping_node')

        self.cmd_pub      = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)
        self.bev_pub      = self.create_publisher(Image, '/qcar2/bev_image', 10)
        self.led_pub      = self.create_publisher(BooleanLeds, 'qcar2_led_cmd', 10)
        self.sliding_pub  = self.create_publisher(Image, '/qcar2/bev_sliding_image', 10)
        self.raw_pub      = self.create_publisher(Image, '/qcar2/raw_image', 10)
        self.image_sub    = self.create_subscription(
            Image, '/camera/color_image', self.image_callback, 10)

        self.bridge       = CvBridge()
        self.latest_image = None
        self.declare_parameter('manual_speed', 0.0)

        # ── Controllers ────────────────────────────────────────────────────
        self.speed_control = SpeedController(Kp=0.01, Ki=0.005, Kff=1/60)
        self._prev_space   = False

        # Speed is always zero
        self.v_desire = 0.0

        if TRAILER_ATTACHED:
            self.trailer = TrailerModel()

        # ── Steering state ─────────────────────────────────────────────────
        self.steer      = 0.0
        self.max_dsteer = 0.25          # tightened from 0.5

        # PD error state
        self.prev_error = 0.0

        self.enable_lane_keep = True

        # ── Polynomial history (reduced from 7 → 3 to cut lag) ────────────
        self.left_a,  self.left_b,  self.left_c  = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        self.history = 3

        self.last_left_fit   = np.array([0.0, 0.0,  205.0])
        self.last_right_fit  = np.array([0.0, 0.0,  595.5])
        self.last_good_steer = 0.0
        self.lost_frames     = 0

        self.create_timer(0.1, self.control_loop)


    # ── Image callback ─────────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")


    # ── Perspective warp ───────────────────────────────────────────────────

    def perspective_warp(self, img, dst_size=(800, 800),
                         src=np.float32([
                             [0.3719, 0.6319], [0.6344, 0.6319],
                             [0.1953, 1.0000], [0.9688, 1.0000]]),
                         dst=np.float32([
                             [0.206, 0.000], [0.766, 0.000],
                             [0.206, 1.000], [0.766, 1.000]]),
                         reverse=False):
        if reverse:
            dst, src = src, dst
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, dst_size)


    # ── Thresholding ───────────────────────────────────────────────────────

    def threshold_lane(self, img):
        h, w = img.shape[:2]

        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        binary = cv2.adaptiveThreshold(
            bilateral, 1,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=-15
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_u8 = (binary * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
        filtered = np.zeros_like(binary)
        min_area = 50
        max_area = int(0.10 * h * w)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                filtered[labels == lbl] = 1
        return filtered


    # ── Mask out known-bad regions ─────────────────────────────────────────

    def overdraw_bad_lines(self, binary):
        xdim, ydim = binary.shape[1], binary.shape[0]
        l_width = int(0.01 * xdim)
        pm = np.array([
            [0.7675, 1.0], [1.0,       0.7],
            [0.075,  1.0], [0.0,       0.89],
            [0.4296875, 0.992], [0.5703125, 0.992]
        ])
        pa = (pm * [xdim, ydim]).astype(int)
        cv2.line(binary, tuple(pa[0]), tuple(pa[1]), 0, l_width * 3)
        cv2.line(binary, tuple(pa[2]), tuple(pa[3]), 0, l_width * 3)
        cv2.line(binary, tuple(pa[4]), tuple(pa[5]), 0, int(l_width * 1.5))
        return binary


    # ── Histogram ─────────────────────────────────────────────────────────

    def get_hist(self, binary):
        hist = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        pc = binary.shape[1] // 40
        hist[:pc]                   = 0
        hist[binary.shape[1] - pc:] = 0
        return hist

    def _best_base(self, hist, x_min, x_max, n_candidates=3):
        region = hist[x_min:x_max].copy()
        if region.max() == 0:
            return x_min + int(np.argmax(region))

        peaks = []
        for i in range(1, len(region) - 1):
            if region[i] > region[i-1] and region[i] > region[i+1]:
                peaks.append((region[i], x_min + i))
        if not peaks:
            return x_min + int(np.argmax(region))

        peaks.sort(key=lambda p: p[0], reverse=True)
        return peaks[0][1]


    # ── Per-lane augmented sliding window ──────────────────────────────────

    def _search_one_lane(self, nonzeroy, nonzerox, start_x, binary_h, binary_w,
                         nwindows, margin, minpix, colour, out_img, draw_windows):

        window_height = binary_h // nwindows
        x_current     = start_x
        slope         = 0.0
        prev_x        = None
        prev_y        = None

        all_inds = []

        for win in range(nwindows):
            win_y_low  = binary_h - (win + 1) * window_height
            win_y_high = binary_h - win * window_height
            win_y_mid  = (win_y_low + win_y_high) / 2.0

            x_anticipated = int(round(x_current + slope * (window_height / 2.0)))
            x_lo = max(0,        x_anticipated - margin)
            x_hi = min(binary_w, x_anticipated + margin)

            inds = ((nonzeroy >= win_y_low)  & (nonzeroy < win_y_high) &
                    (nonzerox >= x_lo)        & (nonzerox < x_hi)).nonzero()[0]
            all_inds.append(inds)

            if draw_windows:
                cv2.rectangle(out_img,
                              (x_lo, win_y_low), (x_hi, win_y_high),
                              colour, 2)

            if len(inds) >= minpix:
                new_x = float(np.mean(nonzerox[inds]))

                if prev_x is not None and prev_y is not None:
                    dy = prev_y - win_y_mid
                    dx = new_x  - prev_x
                    if abs(dy) > 1e-3:
                        new_slope = dx / dy
                        w = min(len(inds) / float(minpix * 2), 1.0)
                        slope = (1.0 - w) * slope + w * new_slope
                        slope = float(np.clip(slope, -2.0, 2.0))

                prev_x    = new_x
                prev_y    = win_y_mid
                x_current = int(round(new_x))

        all_inds = np.concatenate(all_inds)
        xs = nonzerox[all_inds]
        ys = nonzeroy[all_inds]
        return all_inds, xs, ys

    def sliding_window(self, binary, nwindows=36, margin=30, minpix=30,
                       draw_windows=True):
        out_img   = np.dstack((binary, binary, binary)) * 255
        histogram = self.get_hist(binary)
        h, w      = binary.shape[:2]
        midpoint  = w // 2

        nonzero  = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_base  = self._best_base(histogram, 0, midpoint, n_candidates=3)
        rightx_base = self._best_base(histogram, midpoint,  w, n_candidates=3)

        raw_width = rightx_base - leftx_base
        if raw_width < LANE_WIDTH_MIN_PX:
            leftx_base  = int(self.last_left_fit[2])
            rightx_base = int(self.last_right_fit[2])

        _, lx, ly = self._search_one_lane(
            nonzeroy, nonzerox, leftx_base,  h, w,
            nwindows, margin, minpix,
            (255, 180, 0), out_img, draw_windows)

        _, rx, ry = self._search_one_lane(
            nonzeroy, nonzerox, rightx_base, h, w,
            nwindows, margin, minpix,
            (0, 180, 255), out_img, draw_windows)

        n_left  = len(lx)
        n_right = len(rx)

        left_fit  = np.polyfit(ly, lx, 2) if n_left  >= MIN_PIX_LEFT  else None
        right_fit = np.polyfit(ry, rx, 2) if n_right >= MIN_PIX_RIGHT else None

        if n_left  > 0:
            out_img[ly.astype(int), lx.astype(int)] = [255, 0,   100]
        if n_right > 0:
            out_img[ry.astype(int), rx.astype(int)] = [0,   100, 255]

        return out_img, left_fit, right_fit, n_left, n_right


    # ── One-sided recovery ─────────────────────────────────────────────────

    def recover_missing_lane(self, left_fit, right_fit):
        ly = 800.0
        lx = self.last_left_fit[0]*ly**2  + self.last_left_fit[1]*ly  + self.last_left_fit[2]
        rx = self.last_right_fit[0]*ly**2 + self.last_right_fit[1]*ly + self.last_right_fit[2]
        measured = rx - lx
        width_px = measured if LANE_WIDTH_MIN_PX < measured < LANE_WIDTH_MAX_PX \
                   else PIXEL_LANE_WIDTH

        left_reconstructed  = False
        right_reconstructed = False

        if left_fit is None and right_fit is not None:
            left_fit            = right_fit.copy()
            left_fit[2]        -= width_px
            left_reconstructed  = True
        elif right_fit is None and left_fit is not None:
            right_fit            = left_fit.copy()
            right_fit[2]        += width_px
            right_reconstructed  = True
        elif left_fit is None and right_fit is None:
            left_fit             = self.last_left_fit.copy()
            right_fit            = self.last_right_fit.copy()
            left_reconstructed   = True
            right_reconstructed  = True

        return left_fit, right_fit, left_reconstructed, right_reconstructed


    # ── Right lane jump suppression ────────────────────────────────────────

    def stabilize_right_from_left(self, left_fit, right_fit,
                                   left_reconstructed, right_reconstructed):
        """If left is stable and right is jumping frame-to-frame,
        reconstruct right from the stable left fit."""

        # Only act when both lanes were genuinely detected this frame
        if left_reconstructed or right_reconstructed:
            return left_fit, right_fit, left_reconstructed, right_reconstructed

        right_jump = abs(right_fit[2] - self.last_right_fit[2])
        left_jump  = abs(left_fit[2]  - self.last_left_fit[2])

        if left_jump < LEFT_STABLE_THRESH and right_jump > RIGHT_JUMP_THRESH:
            right_fit            = left_fit.copy()
            right_fit[2]        += PIXEL_LANE_WIDTH
            right_reconstructed  = True
            self.get_logger().debug(
                f"Right lane jump={right_jump:.1f}px suppressed — reconstructed from left")

        return left_fit, right_fit, left_reconstructed, right_reconstructed


    # ── Polynomial smoothing ───────────────────────────────────────────────

    def smooth_fits(self, left_fit, right_fit):
        self.left_a.append(left_fit[0]);   self.left_b.append(left_fit[1]);   self.left_c.append(left_fit[2])
        self.right_a.append(right_fit[0]); self.right_b.append(right_fit[1]); self.right_c.append(right_fit[2])
        sm_left  = np.array([np.mean(self.left_a[-self.history:]),
                              np.mean(self.left_b[-self.history:]),
                              np.mean(self.left_c[-self.history:])])
        sm_right = np.array([np.mean(self.right_a[-self.history:]),
                              np.mean(self.right_b[-self.history:]),
                              np.mean(self.right_c[-self.history:])])
        return sm_left, sm_right


    # ── Lane error ─────────────────────────────────────────────────────────

    def lane_error(self, img, left_fit, right_fit,
                   left_reconstructed=False, right_reconstructed=False):

        h = img.shape[0]

        y_near = h * 0.95
        y_far  = h * 0.50          # was 0.30 — pulled back for more reliable BEV region

        def eval_poly(fit, y):
            return fit[0]*y**2 + fit[1]*y + fit[2]

        lx_near = eval_poly(left_fit,  y_near)
        rx_near = eval_poly(right_fit, y_near)
        lx_far  = eval_poly(left_fit,  y_far)
        rx_far  = eval_poly(right_fit, y_far)

        image_center = img.shape[1] / 2.0

        if not left_reconstructed and not right_reconstructed:
            center_near = (lx_near + rx_near) / 2.0
            center_far  = (lx_far  + rx_far)  / 2.0
            lane_width  = rx_near - lx_near

        elif left_reconstructed and not right_reconstructed:
            offset      = PIXEL_LANE_WIDTH / 2.0
            center_near = rx_near - offset
            center_far  = rx_far  - offset
            lane_width  = PIXEL_LANE_WIDTH

        elif right_reconstructed and not left_reconstructed:
            offset      = PIXEL_LANE_WIDTH / 2.0
            center_near = lx_near + offset
            center_far  = lx_far  + offset
            lane_width  = PIXEL_LANE_WIDTH

        else:
            center_near = (lx_near + rx_near) / 2.0
            center_far  = (lx_far  + rx_far)  / 2.0
            lane_width  = PIXEL_LANE_WIDTH

        error_near = center_near - image_center
        error_far  = center_far  - image_center

        # Weights corrected to sum to 1.0 (was 0.85+0.20=1.05)
        error_px = 0.75 * error_near + 0.25 * error_far

        xm_per_pix = 0.43 / lane_width

        return error_px * xm_per_pix, error_px


    # ── Drawing helpers ────────────────────────────────────────────────────

    def _draw_poly_curve(self, overlay, fit, colour, dashed=False, thickness=4):
        h, w = overlay.shape[:2]
        pts = []
        for y in range(0, h, 4):
            x = int(fit[0] * y**2 + fit[1] * y + fit[2])
            if 0 <= x < w:
                pts.append((x, y))
        if len(pts) < 2:
            return
        if dashed:
            for i in range(0, len(pts) - 1, 2):
                cv2.line(overlay, pts[i], pts[i + 1], colour, thickness)
        else:
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], colour, thickness)

    def draw_lanes(self, img, left_fit, right_fit,
                   is_lost=False, left_reconstructed=False, right_reconstructed=False):
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) \
                  if len(img.shape) == 2 else img.copy()
        h, w = overlay.shape[:2]

        y_pts  = np.linspace(0, h - 1, 50).astype(int)
        lx_pts = np.clip((left_fit[0]  * y_pts**2 + left_fit[1]  * y_pts + left_fit[2]).astype(int), 0, w - 1)
        rx_pts = np.clip((right_fit[0] * y_pts**2 + right_fit[1] * y_pts + right_fit[2]).astype(int), 0, w - 1)
        lane_poly  = np.array(list(zip(lx_pts, y_pts)) + list(zip(rx_pts[::-1], y_pts[::-1])), dtype=np.int32)
        fill_layer = overlay.copy()
        cv2.fillPoly(fill_layer, [lane_poly], (0, 0, 60) if is_lost else (0, 40, 0))
        cv2.addWeighted(fill_layer, 0.4, overlay, 0.6, 0, overlay)

        self._draw_poly_curve(overlay, left_fit,
                              (255, 255, 0) if left_reconstructed  else (255, 80,  0),
                              dashed=left_reconstructed,  thickness=4)
        self._draw_poly_curve(overlay, right_fit,
                              (0, 255, 255) if right_reconstructed else (0, 200, 80),
                              dashed=right_reconstructed, thickness=4)
        self._draw_poly_curve(overlay, (left_fit + right_fit) / 2.0,
                              (180, 0, 255), thickness=2)

        lx_b = int(np.clip(left_fit[0]  * h**2 + left_fit[1]  * h + left_fit[2], 0, w - 1))
        rx_b = int(np.clip(right_fit[0] * h**2 + right_fit[1] * h + right_fit[2], 0, w - 1))
        cv2.circle(overlay, (lx_b, h - 4), 14, (255,255,0) if left_reconstructed  else (255,80, 0), -1)
        cv2.circle(overlay, (rx_b, h - 4), 14, (0,255,255) if right_reconstructed else (0,200,80), -1)
        cv2.circle(overlay, ((lx_b+rx_b)//2, h - 4), 8, (180, 0, 255), -1)
        cv2.line(overlay, (w//2, h-4), (w//2, h-80), (255,255,255), 2)

        def _tag(text, col, yp):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(overlay, (8, yp-th-4), (8+tw+6, yp+4), (0,0,0), -1)
            cv2.putText(overlay, text, (11, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        _tag("LEFT:  " + ("RECONSTRUCTED" if left_reconstructed  else "detected"),
             (0,255,255) if left_reconstructed  else (100,255,100), 30)
        _tag("RIGHT: " + ("RECONSTRUCTED" if right_reconstructed else "detected"),
             (0,255,255) if right_reconstructed else (100,255,100), 62)
        if is_lost:
            _tag("!! LANE LOST — COASTING !!", (0,0,255), 94)

        return overlay


    # ── Safe stop ─────────────────────────────────────────────────────────

    def stop_car(self):
        try:
            cmd = MotorCommands()
            cmd.motor_names = ['steering_angle', 'motor_throttle']
            cmd.values      = [0.0, 0.0]
            self.cmd_pub.publish(cmd)
            led = BooleanLeds()
            led.led_names = ["left_front_signal", "left_rear_signal",
                             "right_front_signal", "right_rear_signal"]
            led.values = [False, False, False, False]
            self.led_pub.publish(led)
            self.get_logger().info("Car stopped safely.")
        except Exception as e:
            self.get_logger().error(f"Failed to stop car: {e}")


    # ── Main control loop ──────────────────────────────────────────────────

    def control_loop(self):
        try:
            if self.latest_image is None:
                return

            img = self.latest_image.copy()
            v   = float(self.get_parameter('manual_speed').value)

            # Vision pipeline
            gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bev_lane = self.perspective_warp(gray, dst_size=(800, 800))
            binary   = self.threshold_lane(bev_lane)
            binary   = self.overdraw_bad_lines(binary)

            # Sliding window
            out_img, left_fit, right_fit, n_left, n_right = \
                self.sliding_window(binary, nwindows=25, margin=25, minpix=25)

            # Recover fully missing lane(s)
            left_fit, right_fit, left_reconstructed, right_reconstructed = \
                self.recover_missing_lane(left_fit, right_fit)

            # Suppress jumping right lane when left is stable
            left_fit, right_fit, left_reconstructed, right_reconstructed = \
                self.stabilize_right_from_left(
                    left_fit, right_fit, left_reconstructed, right_reconstructed)

            # Lost counter — only when BOTH lanes are missing
            both_missing = left_reconstructed and right_reconstructed
            if both_missing:
                self.lost_frames += 1
            else:
                self.lost_frames = 0
                if not left_reconstructed and not right_reconstructed:
                    left_fit, right_fit = self.smooth_fits(left_fit, right_fit)
                if not left_reconstructed:
                    self.last_left_fit  = left_fit.copy()
                if not right_reconstructed:
                    self.last_right_fit = right_fit.copy()

            is_lost      = self.lost_frames >= LOST_FRAMES_THRESHOLD
            error, error_px = self.lane_error(
                binary, left_fit, right_fit, left_reconstructed, right_reconstructed)

            # ── Steering (PD) + throttle ───────────────────────────────────
            if self.enable_lane_keep:
                if is_lost:
                    delta_lane = self.last_good_steer
                    current_v  = self.v_desire * COAST_SPEED_FRACTION
                    self.get_logger().warn(
                        f"Lane lost {self.lost_frames} frames — coasting")
                else:
                    # PD controller — derivative damps oscillation
                    d_error         = (error - self.prev_error) / 0.1
                    self.prev_error = error
                    delta_lane      = float(np.clip(
                        -(KP_BASE * error + KD * d_error), -1.0, 1.0))
                    self.last_good_steer = delta_lane

                    if TRAILER_ATTACHED:
                        delta_lane = float(np.clip(
                            delta_lane + self.trailer.steering_correction(), -1.0, 1.0))

                    current_v = self.v_desire   # always 0.0

                d_steer     = np.clip(delta_lane - self.steer, -self.max_dsteer, self.max_dsteer)
                self.steer += d_steer

                if TRAILER_ATTACHED:
                    self.trailer.update(v, self.steer, 0.1)
                    current_v = min(current_v, self.trailer.speed_limit(self.v_desire))

                throttle = self.speed_control.update(v, current_v, 0.1)
            else:
                self.steer = 0.0
                throttle   = 0.0
                if TRAILER_ATTACHED:
                    self.trailer.psi = 0.0

            # Publish motor commands
            cmd = MotorCommands()
            cmd.motor_names = ['steering_angle', 'motor_throttle']
            cmd.values      = [float(self.steer), float(throttle)]
            self.cmd_pub.publish(cmd)

            # LED indicators
            led_msg = BooleanLeds()
            led_msg.led_names = ["left_front_signal", "left_rear_signal",
                                 "right_front_signal", "right_rear_signal"]
            if self.steer < -0.2:
                led_msg.values = [False, False, True, True]
            elif self.steer > 0.2:
                led_msg.values = [True,  True,  False, False]
            else:
                led_msg.values = [False, False, False, False]
            self.led_pub.publish(led_msg)

            self.get_logger().debug(
                f"lost={self.lost_frames}  steer={self.steer:.4f}  error={error:.4f}"
                + (f"  psi={np.rad2deg(self.trailer.psi):.1f}°" if TRAILER_ATTACHED else ""))

            # ── Visualisation ──────────────────────────────────────────────
            bev_vis = self.draw_lanes(bev_lane, left_fit, right_fit,
                                      is_lost, left_reconstructed, right_reconstructed)

            _steer_text = f"steer: {self.steer:+.4f}"
            (tw, th), _ = cv2.getTextSize(_steer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            _yp = 94
            cv2.rectangle(bev_vis, (8, _yp - th - 4), (8 + tw + 6, _yp + 4), (0, 0, 0), -1)
            cv2.putText(bev_vis, _steer_text, (11, _yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

            _error_text = f"err_px: {error_px:+.1f}  err_m: {error:+.4f}"
            (tw2, th2), _ = cv2.getTextSize(_error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            _yp2 = 126
            cv2.rectangle(bev_vis, (8, _yp2 - th2 - 4), (8 + tw2 + 6, _yp2 + 4), (0, 0, 0), -1)
            cv2.putText(bev_vis, _error_text, (11, _yp2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

            _h, _w    = bev_vis.shape[:2]
            image_center = binary.shape[1] / 2.0
            _cx = int(image_center + error_px)
            cv2.circle(bev_vis, (_cx, int(_h * 0.95)), 8, (255, 0, 255), -1)
            cv2.circle(bev_vis, (int(image_center), int(_h * 0.95)), 8, (255, 255, 255), -1)
            cv2.line(bev_vis, (int(image_center), int(_h * 0.95)),
                     (_cx, int(_h * 0.95)), (0, 220, 255), 2)

            raw_vis = self.latest_image.copy()
            ih, iw  = raw_vis.shape[:2]
            src_pts = (np.float32([
                [0.3719, 0.6319], [0.6344, 0.6319],
                [0.1953, 1.0000], [0.9688, 1.0000]
            ]) * np.float32([iw, ih])).astype(np.int32)
            cv2.polylines(raw_vis, [src_pts[[0, 1, 3, 2]]], isClosed=True,
                          color=(0, 255, 255), thickness=2)
            for (px, py), lbl, col in zip(src_pts,
                                          ["TL", "TR", "BL", "BR"],
                                          [(255,80,0),(0,200,80),(255,255,0),(0,255,255)]):
                cv2.circle(raw_vis, (px, py), 8, col, -1)
                cv2.putText(raw_vis, lbl, (px+10, py-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            cv2.waitKey(1)

            bev_msg = self.bridge.cv2_to_imgmsg(bev_vis, encoding="bgr8")
            self.bev_pub.publish(bev_msg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected, stopping car...")
        node.stop_car()
    finally:
        node.stop_car()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()