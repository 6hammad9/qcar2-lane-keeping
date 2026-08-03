# ===============================================================
# HYBRID CONTROLLER FOR QCAR2
# Lane Keeping on Straights
# Pure Pursuit on Curves
# ===============================================================

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import cv2
import math
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from qcar2_interfaces.msg import MotorCommands

# ===============================================================
# MAIN NODE
# ===============================================================

class HybridController(Node):

    def __init__(self):
        super().__init__('hybrid_controller')

        # -------------------------------------------------------
        # CAMERA
        # -------------------------------------------------------
        self.bridge = CvBridge()
        self.latest_image = None

        self.create_subscription(
            Image,
            '/camera/color_image',
            self.image_callback,
            10
        )

        # -------------------------------------------------------
        # MOTOR PUB
        # -------------------------------------------------------
        self.cmd_pub = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )

        # -------------------------------------------------------
        # TF
        # -------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self
        )

        # -------------------------------------------------------
        # WAYPOINTS
        # -------------------------------------------------------
        self.waypoints = np.load('/home/nvidia/waypoints.npy')
        self.current_idx = 0

        # -------------------------------------------------------
        # PARAMETERS
        # -------------------------------------------------------
        self.lookahead = 0.45
        self.wheelbase = 0.256

        self.max_steer = 0.6

        self.speed_straight = 0.18
        self.speed_curve = 0.11

        self.prev_steer = 0.0

        # -------------------------------------------------------
        # LOOP
        # -------------------------------------------------------
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Hybrid controller started.")

    # ===========================================================
    # CAMERA CALLBACK
    # ===========================================================

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except:
            pass

    # ===========================================================
    # TF POSE
    # ===========================================================

    def get_pose(self):

        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time(),
                timeout=Duration(seconds=0.2)
            )

            x = t.transform.translation.x
            y = t.transform.translation.y

            q = t.transform.rotation

            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y*q.y + q.z*q.z)
            )

            return x, y, yaw

        except:
            return None

    # ===========================================================
    # SIMPLE LANE DETECTION
    # ===========================================================

    def lane_steering(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        roi = gray[int(h*0.55):, :]

        _, binary = cv2.threshold(
            roi, 180, 255, cv2.THRESH_BINARY
        )

        histogram = np.sum(binary, axis=0)

        center_lane = np.argmax(histogram)

        image_center = w // 2

        error_px = center_lane - image_center

        steer = -0.0035 * error_px

        steer = np.clip(steer, -self.max_steer, self.max_steer)

        return steer

    # ===========================================================
    # PURE PURSUIT
    # ===========================================================

    def pure_pursuit(self, x, y, yaw):

        if self.current_idx >= len(self.waypoints):
            return 0.0

        target = self.waypoints[self.current_idx]

        tx = target[0]
        ty = target[1]

        dx = math.cos(yaw)*(tx-x) + math.sin(yaw)*(ty-y)
        dy = -math.sin(yaw)*(tx-x) + math.cos(yaw)*(ty-y)

        ld = math.hypot(dx, dy)

        if ld < self.lookahead:
            if self.current_idx < len(self.waypoints)-1:
                self.current_idx += 1

        curvature = 2.0 * dy / (ld*ld + 1e-6)

        steer = math.atan(curvature * self.wheelbase)

        steer = np.clip(steer, -self.max_steer, self.max_steer)

        return steer

    # ===========================================================
    # CURVE DETECTOR FROM FUTURE WAYPOINTS
    # ===========================================================

    def is_curve(self):

        idx = self.current_idx

        if idx + 10 >= len(self.waypoints):
            return False

        p1 = self.waypoints[idx]
        p2 = self.waypoints[idx+5]
        p3 = self.waypoints[idx+10]

        h1 = math.atan2(
            p2[1]-p1[1],
            p2[0]-p1[0]
        )

        h2 = math.atan2(
            p3[1]-p2[1],
            p3[0]-p2[0]
        )

        d = abs(self.wrap_angle(h2-h1))

        return d > 0.12

    def wrap_angle(self, a):
        while a > math.pi:
            a -= 2*math.pi
        while a < -math.pi:
            a += 2*math.pi
        return a

    # ===========================================================
    # SEND COMMAND
    # ===========================================================

    def send_command(self, steer, speed):

        msg = MotorCommands()

        msg.motor_names = [
            'steering_angle',
            'motor_throttle'
        ]

        msg.values = [
            float(steer),
            float(speed)
        ]

        self.cmd_pub.publish(msg)

    # ===========================================================
    # CONTROL LOOP
    # ===========================================================

    def control_loop(self):

        pose = self.get_pose()

        if pose is None:
            return

        x, y, yaw = pose

        pp_steer = self.pure_pursuit(x, y, yaw)

        lane_ok = self.latest_image is not None

        if lane_ok:
            lane_steer = self.lane_steering(
                self.latest_image.copy()
            )
        else:
            lane_steer = 0.0

        curve = self.is_curve()

        # ------------------------------------------------------
        # MODE SWITCH
        # ------------------------------------------------------
        if curve:

            # CURVE MODE
            steer = 0.85 * pp_steer + 0.15 * lane_steer
            speed = self.speed_curve

            mode = "PURE_PURSUIT"

        else:

            # STRAIGHT MODE
            steer = 0.75 * lane_steer + 0.25 * pp_steer
            speed = self.speed_straight

            mode = "LANE_KEEP"

        # Smooth steering
        steer = 0.7*self.prev_steer + 0.3*steer
        self.prev_steer = steer

        steer = np.clip(
            steer,
            -self.max_steer,
            self.max_steer
        )

        self.send_command(steer, speed)

        self.get_logger().info(
            f'{mode} | WP:{self.current_idx}/{len(self.waypoints)} '
            f'Steer:{steer:.3f} Speed:{speed:.2f}',
            throttle_duration_sec=0.5
        )

# ===============================================================
# MAIN
# ===============================================================

def main(args=None):

    rclpy.init(args=args)

    node = HybridController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.send_command(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()