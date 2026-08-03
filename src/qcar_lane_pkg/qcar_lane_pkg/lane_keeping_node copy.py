# qcar2_lane_ros/lane_keeping_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from hal.content.qcar_functions import SpeedController
from pal.utilities.keyboard import PygameKeyboard
from pal.utilities.math import wrap_to_pi
from pal.products.qcar import QCAR_CONFIG


class LaneKeepingNode(Node):
    def __init__(self):
        super().__init__('lane_keeping_node')

        # -----------------------------
        # ROS publishers and subscribers
        # -----------------------------
        self.cmd_pub = self.create_publisher(Twist, '/qcar2_motor_speed_cmd', 10)
        self.bev_pub = self.create_publisher(Image, '/qcar2/bev_image', 10)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color_image',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.latest_image = None

        # -----------------------------
        # Lane keeping and speed control
        # -----------------------------
        self.speed_control = SpeedController(Kp=0.01, Ki=0.005, Kff=1/60)

        # Keyboard input
        self.kb = PygameKeyboard()
        self._prev_space = False

        # Control parameters
        self.v_desire = 10
        self.enable_lane_keep = False

        # -----------------------------
        # Temporal smoothing buffers (ported from working ROS1 node)
        # -----------------------------
        self.left_a,  self.left_b,  self.left_c  = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        self.history = 5          # frames to average over

        # Steering rate limiter
        self.steer = 0.0
        self.max_dsteer = 0.05    # max change in steering per tick

        # Timer for 10 Hz control loop
        self.create_timer(0.1, self.control_loop)

    # -----------------------------
    # Image callback
    # -----------------------------
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    # -----------------------------
    # Perspective warp
    # -----------------------------
    def perspective_warp(self, img, dst_size=(800, 800),
                         src=np.float32([[0.117, 0.989], [0.977, 0.989],
                                         [0.359, 0.671], [0.669, 0.671]]),
                         dst=np.float32([(0.285, 0.996), (0.828, 0.996),
                                         (0.285, 0), (0.828, 0)]),
                         reverse=False):
        if reverse:
            dst, src = src, dst
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    # -----------------------------
    # Thresholding for lane detection
    # Pipeline tuned for indoor printed-mat tracks:
    #   1. Bilateral filter   — smooths mat texture, preserves lane edges
    #   2. Adaptive threshold — handles uneven lighting
    #   3. ROI trapezoid mask — kills edges, corners, and top-half clutter
    #   4. Morph open         — removes speckles too small to be lane lines
    #   5. Connected component filter — removes blobs by area (too small = speckle,
    #                                   too large = mat edge / wall bleeding in)
    # Outputs 0/1 binary so histogram is naturally normalised.
    # -----------------------------
    def threshold_lane(self, img):
        h, w = img.shape[:2]

        # Step 1: Bilateral filter
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 2: Adaptive threshold
        binary = cv2.adaptiveThreshold(
            bilateral, 1,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=-15
        )

        

        # Step 4: Morphological open — kills speckles smaller than 3×3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Step 5: Connected component filtering
        # Lane line segments should be tall and narrow.
        # Min area removes speckles; max area removes large false blobs (wall, mat edge).
        binary_u8 = (binary * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
        filtered = np.zeros_like(binary)
        min_area = 50          # smaller than this = speckle noise
        max_area = int(0.10 * h * w)  # larger than this = wall / mat edge blob
        for lbl in range(1, num_labels):   # skip background label 0
            area = stats[lbl, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                filtered[labels == lbl] = 1
        return filtered

    # -----------------------------
    # Mask out known noisy regions in the BEV
    # (ported from overdrawbadlines in working node)
    # -----------------------------
    def overdraw_bad_lines(self, binary):
        xdim = binary.shape[1]
        ydim = binary.shape[0]
        l_width = int(0.01 * xdim)
        pm = np.array([
            [0.765625, 1.0],  [1.0,      0.713541],
            [0.25,     1.0],  [0.0,      0.6770833],
            [0.4296875,0.992],[0.5703125, 0.992]
        ])
        pa = (pm * [xdim, ydim]).astype(int)
        cv2.line(binary, tuple(pa[0]), tuple(pa[1]), 0, l_width * 2)
        cv2.line(binary, tuple(pa[2]), tuple(pa[3]), 0, l_width * 2)
        cv2.line(binary, tuple(pa[4]), tuple(pa[5]), 0, int(l_width * 1.5))
        return binary

    # -----------------------------
    # Histogram with edge suppression (ported from get_hist)
    # -----------------------------
    def get_hist(self, binary):
        hist = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        # Zero out outer 10% on each side to ignore edge noise
        pc = binary.shape[1] // 10
        hist[:pc] = 0
        hist[binary.shape[1] - pc:] = 0
        return hist

    # -----------------------------
    # Sliding window lane detection
    # Key improvements vs original:
    #   - more windows (36), tighter margin (30), higher minpix (200)
    #   - uses get_hist with edge suppression
    #   - temporal smoothing over self.history frames
    # -----------------------------
    def sliding_window(self, binary,
                       nwindows=36, margin=30, minpix=200,
                       draw_windows=True):

        left_fit_  = np.empty(3)
        right_fit_ = np.empty(3)

        out_img = np.dstack((binary, binary, binary)) * 255

        histogram = self.get_hist(binary)
        midpoint  = histogram.shape[0] // 2

        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = binary.shape[0] // nwindows

        nonzero  = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current  = leftx_base
        rightx_current = rightx_base

        left_lane_inds  = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low  = binary.shape[0] - (window + 1) * window_height
            win_y_high = binary.shape[0] - window * window_height

            win_xleft_low   = leftx_current  - margin
            win_xleft_high  = leftx_current  + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            if draw_windows:
                cv2.rectangle(out_img,
                              (win_xleft_low,  win_y_low),
                              (win_xleft_high, win_y_high),
                              (100, 255, 255), 2)
                cv2.rectangle(out_img,
                              (win_xright_low,  win_y_low),
                              (win_xright_high, win_y_high),
                              (100, 255, 255), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate indices
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx  = nonzerox[left_lane_inds]
        lefty  = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Guard against too few points
        if len(leftx) < 2:
            left_fit = np.array([0.0, 0.0, 0.0])
        else:
            left_fit = np.polyfit(lefty, leftx, 2)

        if len(rightx) < 2:
            right_fit = np.array([0.0, 0.0, float(binary.shape[1])])
        else:
            right_fit = np.polyfit(righty, rightx, 2)

        # ---- Temporal smoothing ----
        self.left_a.append(left_fit[0]);  self.left_b.append(left_fit[1]);  self.left_c.append(left_fit[2])
        self.right_a.append(right_fit[0]); self.right_b.append(right_fit[1]); self.right_c.append(right_fit[2])

        left_fit_[0]  = np.mean(self.left_a[-self.history:])
        left_fit_[1]  = np.mean(self.left_b[-self.history:])
        left_fit_[2]  = np.mean(self.left_c[-self.history:])
        right_fit_[0] = np.mean(self.right_a[-self.history:])
        right_fit_[1] = np.mean(self.right_b[-self.history:])
        right_fit_[2] = np.mean(self.right_c[-self.history:])

        # Colour detected pixels in visualisation
        out_img[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]]  = [255, 0,   100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0,   100, 255]

        return out_img, left_fit_, right_fit_

    # -----------------------------
    # Compute lane center error in pixels
    # Includes lane-width sanity check (ported logic)
    # -----------------------------
    def lane_error(self, img, left_fit, right_fit):
        y = img.shape[0]

        left_x  = left_fit[0]  * y**2 + left_fit[1]  * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]

        lane_width = right_x - left_x
        # Reject implausible detections for an 800 px BEV
        if lane_width < 100 or lane_width > 700:
            return 0.0

        lane_center    = (left_x + right_x) / 2
        vehicle_center = img.shape[1] / 2
        return lane_center - vehicle_center

    # -----------------------------
    # Lane overlay for BEV visualisation
    # -----------------------------
    def draw_lanes(self, img, left_fit, right_fit):
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h = overlay.shape[0]
        left_x  = int(left_fit[0]  * h**2 + left_fit[1]  * h + left_fit[2])
        right_x = int(right_fit[0] * h**2 + right_fit[1] * h + right_fit[2])
        cv2.circle(overlay, (left_x,  h - 1), 10, (255, 0, 0),  -1)
        cv2.circle(overlay, (right_x, h - 1), 10, (0, 255, 0),  -1)
        return overlay

    # -----------------------------
    # Control loop
    # -----------------------------
    def control_loop(self):
        try:
            if self.latest_image is None:
                return

            img = self.latest_image.copy()
            v   = 0

            # Convert to grayscale → BEV warp
            gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bev_lane  = self.perspective_warp(gray, dst_size=(800, 800))

            # Threshold → mask bad lines → sliding window
            binary    = self.threshold_lane(bev_lane)
            binary    = self.overdraw_bad_lines(binary)
            out_img, left_fit, right_fit = self.sliding_window(binary,15,30,30)

            error = self.lane_error(binary, left_fit, right_fit)

            # ---- Keyboard: toggle lane-keep on spacebar press (not hold) ----
            self.kb.read()
            if self.kb.k_space and not self._prev_space:
                self.enable_lane_keep = not self.enable_lane_keep
                self.get_logger().info(
                    f"Lane keep {'ON' if self.enable_lane_keep else 'OFF'}")
            self._prev_space = self.kb.k_space

            if self.kb.k_esc:
                self.get_logger().info("Emergency Stop triggered")
                rclpy.shutdown()
                return

            # ---- Steering with rate limiter ----
            if self.enable_lane_keep:
                target_steer = float(np.clip(-0.005 * error, -1.0, 1.0))
                delta = target_steer - self.steer
                delta = np.clip(delta, -self.max_dsteer, self.max_dsteer)
                self.steer += delta
                throttle = self.speed_control.update(v, self.v_desire, 0.1)
            else:
                self.steer = 0.0
                throttle   = 0.0

            # Publish motor commands
            cmd           = Twist()
            cmd.linear.x  = float(throttle)
            cmd.angular.z = float(self.steer)
            self.cmd_pub.publish(cmd)
            self.get_logger().debug(f"steer={self.steer:.4f}  error={error:.1f}")

            # ---- Visualisation ----
            bev_vis = self.draw_lanes(bev_lane, left_fit, right_fit)
            cv2.imshow("Lane BEV (fitted)",   bev_vis)
            cv2.imshow("Sliding Windows",      out_img)
            cv2.imshow("Camera Raw",           self.latest_image)
            cv2.imshow("Binary Lane",          binary * 255)   # scale 0/1 → 0/255
            cv2.waitKey(1)

            # Publish BEV image
            bev_msg = self.bridge.cv2_to_imgmsg(bev_vis, encoding="bgr8")
            self.bev_pub.publish(bev_msg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {str(e)}")


# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()