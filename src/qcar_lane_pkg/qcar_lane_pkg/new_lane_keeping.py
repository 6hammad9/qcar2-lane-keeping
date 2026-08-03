import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import time
from collections import deque
from typing import Optional, Tuple
from qcar2_interfaces.msg import MotorCommands
from qcar2_interfaces.msg import BooleanLeds
from pit.LaneNet.nets import LaneNet


class BEVLaneDetector:
    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        warp_w_top: int = 150,
        warp_w_bot: int = 640,
        warp_h: int = 180,
        warp_y_offset: int = 0,
        nwindows: int = 9,
        margin: int = 100,
        minpix: int = 50,
        steer_gain: float = 1.0,
        curve_gain: float = 200.0,
        camera_offset_m: float = 0.032,
        lane_width_ref_m: float = 0.43,
        single_lane_offset_px: int = 250,
        assumed_lane_width_px: int = 500,
        min_lane_width_px: int = 100,
        max_lane_width_px: int = 600,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.warp_w_top = warp_w_top
        self.warp_w_bot = warp_w_bot
        self.warp_h = warp_h
        self.warp_y_offset = warp_y_offset
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.steer_gain = steer_gain
        self.curve_gain = curve_gain
        self.camera_offset_m = camera_offset_m
        self.lane_width_ref_m = lane_width_ref_m
        self.single_lane_offset_px = single_lane_offset_px
        self.assumed_lane_width_px = assumed_lane_width_px
        self.min_lane_width_px = min_lane_width_px
        self.max_lane_width_px = max_lane_width_px

        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)

        self.M = None
        self.Minv = None
        self._update_transform_matrix()

    def _update_transform_matrix(self):
        cx = self.img_width // 2
        src = np.float32([
            [cx - self.warp_w_top // 2, self.img_height - self.warp_h - self.warp_y_offset],
            [cx + self.warp_w_top // 2, self.img_height - self.warp_h - self.warp_y_offset],
            [cx + self.warp_w_bot // 2, self.img_height - self.warp_y_offset],
            [cx - self.warp_w_bot // 2, self.img_height - self.warp_y_offset]
        ])
        dst_margin = self.img_width * 0.2
        dst = np.float32([
            [dst_margin, 0],
            [self.img_width - dst_margin, 0],
            [self.img_width - dst_margin, self.img_height],
            [dst_margin, self.img_height]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return white_mask

    def _find_lanes(self, binary_warped: np.ndarray) -> Tuple:
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = len(histogram) // 2
        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        window_height = binary_warped.shape[0] // self.nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = left_base
        rightx_current = right_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= leftx_current - self.margin) &
                (nonzerox < leftx_current + self.margin)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= rightx_current - self.margin) &
                (nonzerox < rightx_current + self.margin)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

        left_fit = None
        right_fit = None
        left_conf = 0.0
        right_conf = 0.0

        if len(left_lane_inds) > 50:
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            left_conf = min(len(left_lane_inds) / 1000.0, 1)
        if len(right_lane_inds) > 50:
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            right_conf = min(len(right_lane_inds) / 1000.0, 1)

        if left_fit is not None and right_fit is not None:
            y_eval = binary_warped.shape[0]
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_width = abs(right_x - left_x)

            if lane_width < self.min_lane_width_px or lane_width > self.max_lane_width_px:
                if len(left_lane_inds) < len(right_lane_inds):
                    left_fit = None
                    left_conf = 0
                else:
                    right_fit = None
                    right_conf = 0

        return left_fit, right_fit, left_conf, right_conf

    def _smooth_fit(self, new_fit, history: deque):
        if new_fit is None:
            return np.mean(history, axis=0) if history else None
        history.append(new_fit)
        return np.mean(history, axis=0)

    def _calculate_offset_curvature(self, left_fit, right_fit) -> Tuple[float, float]:
        y_eval = self.img_height
        left_x = 0
        right_x = self.img_width
        lane_width = self.assumed_lane_width_px

        if left_fit is not None:
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        if right_fit is not None:
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

        if left_fit is not None and right_fit is not None:
            lane_center = (left_x + right_x) / 2
            lane_width = right_x - left_x
        elif left_fit is not None:
            lane_center = left_x + self.single_lane_offset_px
        elif right_fit is not None:
            lane_center = right_x - self.single_lane_offset_px
        else:
            return 0.0, 0.0

        cam_offset_px = (self.camera_offset_m / self.lane_width_ref_m) * lane_width
        car_pos = (self.img_width / 2) - cam_offset_px
        offset = (car_pos - lane_center) / (lane_width / 2)
        offset = float(np.clip(offset, -1.0, 1.0))

        curvature = 0.0
        if left_fit is not None and right_fit is not None:
            curvature = float((left_fit[0] + right_fit[0]) / 2)
        elif left_fit is not None:
            curvature = float(left_fit[0])
        elif right_fit is not None:
            curvature = float(right_fit[0])

        return offset, curvature

    def detect(self, bgr_image: np.ndarray) -> dict:
        try:
            binary = self._preprocess(bgr_image)
            warped = cv2.warpPerspective(binary, self.M, (self.img_width, self.img_height))

            left_fit, right_fit, left_conf, right_conf = self._find_lanes(warped)

            left_fit = self._smooth_fit(left_fit, self.left_fit_history)
            right_fit = self._smooth_fit(right_fit, self.right_fit_history)

            is_valid = left_fit is not None and right_fit is not None

            offset, curvature = self._calculate_offset_curvature(left_fit, right_fit)

            steering = self.steer_gain * offset + self.curve_gain * curvature
            steering = float(np.clip(steering, -0.5, 0.5))

            confidence = (left_conf + right_conf) / 2.0
            debug_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

            ploty = np.linspace(0, self.img_height - 1, self.img_height)

            if left_fit is not None:
                left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
                pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int32)
                cv2.polylines(debug_img, pts, isClosed=False, color=(255, 0, 0), thickness=3)

            if right_fit is not None:
                right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
                pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int32)
                cv2.polylines(debug_img, pts, isClosed=False, color=(0, 0, 255), thickness=3)

            return dict(
                is_valid=is_valid,
                steering=steering,
                confidence=confidence,
                offset=offset,
                curvature=curvature,
                left_detected=left_fit is not None,
                right_detected=right_fit is not None,
                debug_binary=binary,
                debug_warped=warped,
                debug_overlay=debug_img,
            )
        except Exception as e:
            print(e)
            return dict(
                is_valid=False, steering=0.0, confidence=0.0, offset=0.0,
                curvature=0.0, left_detected=False, right_detected=False,
            )


class LaneNetDetector:
    def __init__(self,
                 img_width: int = 640,
                 img_height: int = 480,
                 steer_gain: float = 1.0,
                 row_upper_bound: int = 180,
                 ):
        self.img_width = img_width
        self.img_height = img_height
        self.steer_gain = steer_gain
        self.row_upper_bound = row_upper_bound
        self.lanenet = None

    def initialize(self) -> bool:
        self.lanenet = LaneNet(
            imageHeight=self.img_height,
            imageWidth=self.img_width,
            rowUpperBound=self.row_upper_bound
        )
        return True

    def detect(self, bgr_image: np.ndarray) -> dict:
        try:
            # Preprocess and predict
            processed = self.lanenet.pre_process(bgr_image)
            binary_pred, instance_pred = self.lanenet.predict(processed)

            if binary_pred is None:
                return self._invalid()

            # Compute lane pixels for confidence
            lane_pixels = np.sum(binary_pred > 0.5)
            total_pixels = binary_pred.size
            confidence = float(min(lane_pixels / (total_pixels * 0.1), 1.0))

            if confidence < 0.1:
                return self._invalid()

            # Simple steering from lane mask
            steering = self._steering_from_mask(binary_pred)
            if steering == 0.0:
                return self._invalid()

            # Render the overlaid image (no DBSCAN)
            annotated_img = self.lanenet.render(showFPS=True)

            rendered_img = self.lanenet.post_process_render(showFPS=True)
            return dict(
                is_valid=True,
                steering=steering,
                confidence=confidence,
                offset=steering / max(self.steer_gain, 1e-6),
                curvature=0.0,
                left_detected=True,
                right_detected=True,
                debug_lanenet=annotated_img,
                debug_clusters=None,  # DBSCAN removed
                rendered_img = rendered_img
            )
        except Exception as e:
            print(f"LaneNet detect error: {e}")
            return self._invalid()


    def _steering_from_mask(self, mask: np.ndarray)->float:
        try:
            height = mask.shape[0]
            bottom = mask[int(height* 0.7):,:]
            ys,xs = np.nonzero(bottom >0.5)
            if len(xs)<10:
                return 0.0
            lane_center =float(np.mean(xs))
            image_center = mask.shape[1] / 2
            offset =(lane_center -image_center) /(mask.shape[1]/ 2)
            steering = float(np.clip(-offset*self.steer_gain, -0.5,0.5))
            return steering
        except Exception as e:
            print(e)
            return 0.0

    @staticmethod
    def _invalid() -> dict:
        return dict(is_valid=False, steering=0.0, confidence=0.0,
                    offset=0.0, curvature=0.0,
                    left_detected=False, right_detected=False)


class LaneKeepingNode(Node):

    def __init__(self):
        super().__init__("lane_keeping_node")

        self.declare_parameter("detector", "lanenet")
        self.declare_parameter("throttle", 0)
        self.declare_parameter("steer_gain", 1.0)
        self.declare_parameter("curve_gain", 200.0)
        self.declare_parameter("img_width", 640)
        self.declare_parameter("img_height", 480)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("debug_images", True)
        self.declare_parameter("lanenet_row_upper_bound", 200)

        def p(n): return self.get_parameter(n).value

        self._throttle = float(p("throttle"))
        self.max_steer = 0.5
        self._debug = bool(p("debug_images"))

        detector_name = str(p("detector")).lower()
        img_w = int(p("img_width"))
        img_h = int(p("img_height"))
        steer_gain = float(p("steer_gain"))

        if detector_name == "lanenet":
            self._detector = LaneNetDetector(
                img_width=img_w,
                img_height=img_h,
                steer_gain=steer_gain,
                row_upper_bound=int(p("lanenet_row_upper_bound"))
            )
            try:
                self._detector.initialize()
                self.get_logger().info("Lanenet detector initialised successfully")
            except Exception as e:
                self.get_logger().fatal(f"Lanenet failed to initialise: {e}")
                raise SystemExit(1)
        else:
            self._detector = BEVLaneDetector(
                img_width=img_w,
                img_height=img_h,
                steer_gain=steer_gain,
                curve_gain=float(p("curve_gain"))
            )
            self.get_logger().info("BEV detector initialised")

        self._bridge = CvBridge()
        self._image_sub = self.create_subscription(
            Image, "/camera/color_image", self.image_callback, 10)
        self._cmd_pub = self.create_publisher(
            MotorCommands, "qcar2_motor_speed_cmd", 10)

        # --- Debug image publishers ---
        self._pub_camera   = self.create_publisher(Image, "~/debug/camera",   1)
        self._pub_binary   = self.create_publisher(Image, "~/debug/binary",   1)
        self._pub_warped   = self.create_publisher(Image, "~/debug/warped",   1)
        self._pub_overlay  = self.create_publisher(Image, "~/debug/overlay",  1)
        self._pub_lanenet  = self.create_publisher(Image, "~/debug/lanenet",  1)
        self._pub_clusters = self.create_publisher(Image, "~/debug/clusters", 1)

        self._latest_image: Optional[np.ndarray] = None
        self._last_steer = 0.0
        self._frame_count = 0
        self._t0 = time.time()

        rate_hz = float(p("publish_rate"))
        self._timer = self.create_timer(1.0 / rate_hz, self._control_loop)

        self.get_logger().info(
            f"LaneKeepingNode ready  throttle={self._throttle}  rate={rate_hz} Hz")

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if img is not None:
                img = cv2.resize(
                    img, (self._detector.img_width, self._detector.img_height))
            self._latest_image = img
        except CvBridgeError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2)

    # ------------------------------------------------------------------
    def _publish_image(self, publisher, img: np.ndarray, encoding: str = "bgr8"):
        """Convert a numpy array to a ROS Image and publish it."""
        try:
            ros_img = self._bridge.cv2_to_imgmsg(img, encoding=encoding)
            publisher.publish(ros_img)
        except CvBridgeError as e:
            self.get_logger().warn(f"Image publish error: {e}", throttle_duration_sec=2)

    # ------------------------------------------------------------------
    def _publish_cmds(self, throttle: float, steering: float):
        cmd = MotorCommands()
        cmd.motor_names = ['steering_angle', 'motor_throttle']
        cmd.values = [float(steering), float(throttle)]
        self._cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    def _control_loop(self):
        if self._latest_image is None:
            return

        result = self._detector.detect(self._latest_image)

        if result["is_valid"]:
            steering = float(np.clip(result["steering"], -self.max_steer, self.max_steer))
            self._last_steer = steering
            throttle = self._throttle
        else:
            steering = self._last_steer
            throttle = 0

        # self._publish_cmds(throttle, steering)

        self._frame_count += 1

        if self._debug and self._frame_count % 30 == 0:
            fps = self._frame_count / max(time.time() - self._t0, 1e-6)
            self.get_logger().info(f"fps={fps:.1f}")

        if self._debug:
            # Raw camera
            self._publish_image(self._pub_camera, self._latest_image)

            # BEV-specific debug images
            binary = result.get("debug_binary")
            warped = result.get("debug_warped")
            overlay = result.get("debug_overlay")

            if binary is not None:
                # binary is single-channel; publish as mono8
                self._publish_image(self._pub_binary, binary, encoding="mono8")

            if warped is not None:
                self._publish_image(self._pub_warped, warped, encoding="mono8")

            if overlay is not None:
                vis = overlay.copy()
                text = (f"steer: {result['steering']:.3f} "
                        f"conf: {result['confidence']:.2f}")
                cv2.putText(vis, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self._publish_image(self._pub_overlay, vis)

            # LaneNet-specific debug images
            
            
            lanenet_img = result.get("debug_lanenet")
            if lanenet_img is not None:
                vis = lanenet_img.copy()
                car_center_x = vis.shape[1] // 2
                cv2.line(vis, (car_center_x, 0), (car_center_x, vis.shape[0]), (0, 255, 255), 2)  # car center

                if "offset" in result:
                    lane_center_x = int(car_center_x - result["offset"] * (vis.shape[1] // 2))
                    cv2.line(vis, (lane_center_x, 0), (lane_center_x, vis.shape[0]), (255, 255, 0), 2)  # lane center

                text = (f"steer: {result['steering']:.3f} "
                        f"conf: {result['confidence']:.2f} "
                        f"offset: {result['offset']:.3f}")
                cv2.putText(vis, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self._publish_image(self._pub_lanenet, vis)

            

    # ------------------------------------------------------------------
    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected, stopping car...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()