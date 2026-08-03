#!/usr/bin/env python3

import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String, Int32
from tf2_ros import Buffer, TransformListener, TransformException

from qcar_science_night_pkg.path_utils import PathUtils


class QCarTrailerPathMPC(Node):
    """
    Reverse-only trailer-aware MPC for QCar2.

    State:
        x, y, car_yaw, trailer_yaw

    Control:
        steering delta, velocity v (v <= 0 always)

    Notes:
        - L_trailer = 0.10 m (measured hitch-to-rear-axle)
        - Reverse-only test mode: loads reverse trajectory at startup
        - Pickup stop disabled
        - Reduced speed and tighter jackknife limit for short trailer safety
    """

    MODE_REVERSE = "REVERSE"

    def __init__(self):
        super().__init__("qcar_trailer_path_mpc_node")

        # ---------- Trajectory file (reverse only) ----------
        self.reverse_trajectory_file = "/home/nvidia/ros2_ws/bags_and_paths/recorded_path_reverse27.npy"

        # ---------- Vehicle / trailer parameters ----------
        self.N = 25
        self.dt = 0.08
        self.L_car = 0.256
        self.L_trailer = 0.10          # measured hitch-to-rear-axle distance

        self.max_reverse_steer = 0.45
        self.publish_steering_angle = True

        # ---------- Reverse speeds (conservative for short trailer) ----------
        self.v_min_reverse = -0.06
        self.v_max_reverse = 0.0
        self.reverse_speed = -0.04     # was -0.08; halved for L_trailer=0.10 safety

        # ---------- Safety ----------
        self.max_articulation = math.radians(25.0)  # tighter than default 35 deg

        # ---------- MPC weights ----------
        self.w_pos = 220.0
        self.w_yaw = 90.0
        self.w_trailer_align = 140.0
        self.w_delta_rate = 55.0
        self.w_speed_rate = 8.0
        self.w_speed_tracking = 65.0
        self.w_control = 4.0

        # ---------- Mode (reverse only) ----------
        self.mode = self.MODE_REVERSE
        self.reverse_mode = True
        self.mission_done = False

        self.closest_idx = 0
        self.last_solution = None

        self.prev_delta = 0.0
        self.prev_v = 0.0
        self.trailer_yaw_est = None

        self.motion_enabled = False
        self.drive_state = "DRIVE"
        self.depth_emergency = False

        # ---------- Pickup stop disabled for reverse test ----------
        self.pickup_done = True        # skip entirely
        self.pickup_active = False

        # ---------- Logging ----------
        self.tracking_log_path = "/home/nvidia/ros2_ws/trailer_mpc_reverse_test_log.csv"
        self.tracking_log = open(self.tracking_log_path, "w")
        self.tracking_log.write(
            "time,idx,x,y,yaw,trailer_yaw,articulation_deg,"
            "ref_x,ref_y,ref_yaw,track_error,yaw_error_deg,"
            "target_v,v,delta,drive_state,depth_emergency,motion_enabled\n"
        )
        self.tracking_log.flush()

        # ---------- ROS interfaces ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.idx_pub = self.create_publisher(Int32, "/current_path_idx", 10)
        self.yaw_stable_pub = self.create_publisher(Bool, "/overtake_yaw_stable", 10)

        self.create_subscription(Bool, "/motion_enable", self.motion_callback, 10)
        self.create_subscription(String, "/drive_state", self.drive_state_callback, 10)
        self.create_subscription(Bool, "/depth_emergency_stop", self.depth_emergency_callback, 10)

        # ---------- Load reverse trajectory ----------
        self.trajectory = PathUtils.load_trajectory(self.reverse_trajectory_file)
        if len(self.trajectory) < self.N + 2:
            raise RuntimeError(
                f"Reverse trajectory too short. Need {self.N + 2}, got {len(self.trajectory)}"
            )

        self._setup_optimizer()
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f"Reverse-only trailer MPC ready. "
            f"L_trailer={self.L_trailer}, "
            f"reverse_speed={self.reverse_speed}, "
            f"max_articulation={math.degrees(self.max_articulation):.1f} deg, "
            f"trajectory_length={len(self.trajectory)}"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def motion_callback(self, msg):
        self.motion_enabled = bool(msg.data)

    def drive_state_callback(self, msg):
        self.drive_state = str(msg.data)

    def depth_emergency_callback(self, msg):
        self.depth_emergency = bool(msg.data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.prev_v = 0.0
        self.prev_delta = 0.0

    def reset_mpc_memory(self):
        self.last_solution = np.zeros(self.nx + self.nu)
        self.prev_v = 0.0
        self.prev_delta = 0.0

    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            tr = t.transform.translation
            q = t.transform.rotation
            yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
            return np.array([tr.x, tr.y, yaw], dtype=float)
        except TransformException as e:
            self.get_logger().warn(f"TF unavailable: {e}")
            return None

    def update_trailer_yaw_estimate(self, car_yaw):
        """
        Integrate trailer yaw using the kinematic trailer model.
        Uses prev_v (the velocity commanded in the previous loop iteration).
        On first call, initialises trailer yaw to car yaw (straight assumption).
        """
        if self.trailer_yaw_est is None:
            self.trailer_yaw_est = car_yaw
            self.get_logger().info(
                f"Trailer yaw initialised to car yaw: {math.degrees(car_yaw):.2f} deg"
            )
            return

        articulation = self.wrap_angle(car_yaw - self.trailer_yaw_est)
        self.trailer_yaw_est += (
            self.dt * (self.prev_v / self.L_trailer) * math.sin(articulation)
        )
        self.trailer_yaw_est = self.wrap_angle(self.trailer_yaw_est)

    def publish_current_idx(self):
        idx_msg = Int32()
        idx_msg.data = int(self.closest_idx)
        self.idx_pub.publish(idx_msg)

    # ------------------------------------------------------------------
    # MPC setup
    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        # State: x, y, car_yaw, trailer_yaw
        X = ca.SX.sym("X", 4, self.N + 1)
        # Control: steering delta, velocity v
        U = ca.SX.sym("U", 2, self.N)
        # Parameters: current state (4) + prev delta/v (2) + target_v (1) + N refs * 3
        P = ca.SX.sym("P", 4 + 2 + 1 + self.N * 3)

        cost = 0
        g = []
        g.append(X[:, 0] - P[0:4])

        prev_delta = P[4]
        prev_v = P[5]
        target_v = P[6]

        for k in range(self.N):
            st = X[:, k]
            x = st[0]
            y = st[1]
            theta = st[2]
            phi = st[3]

            delta_k = U[0, k]
            v_k = U[1, k]

            idx = 7 + 3 * k
            x_ref = P[idx]
            y_ref = P[idx + 1]
            yaw_ref = P[idx + 2]

            # Position cost (log barrier to avoid large penalty spikes)
            pos_err = (x - x_ref) ** 2 + (y - y_ref) ** 2
            cost += self.w_pos * ca.log(1.0 + pos_err)

            # Car yaw cost
            yaw_err = ca.atan2(ca.sin(theta - yaw_ref), ca.cos(theta - yaw_ref))
            cost += self.w_yaw * yaw_err ** 2

            # Trailer alignment cost (penalise articulation)
            articulation = ca.atan2(ca.sin(theta - phi), ca.cos(theta - phi))
            cost += self.w_trailer_align * articulation ** 2

            # Rate costs
            if k == 0:
                ddelta = delta_k - prev_delta
                dv = v_k - prev_v
            else:
                ddelta = delta_k - U[0, k - 1]
                dv = v_k - U[1, k - 1]

            cost += self.w_delta_rate * ddelta ** 2
            cost += self.w_speed_rate * dv ** 2
            cost += self.w_speed_tracking * (v_k - target_v) ** 2
            cost += self.w_control * delta_k ** 2

            # Kinematic model (reverse-compatible: v_k <= 0)
            x_next = x + self.dt * v_k * ca.cos(theta)
            y_next = y + self.dt * v_k * ca.sin(theta)
            theta_next = theta + self.dt * (v_k / self.L_car) * ca.tan(delta_k)
            phi_next = phi + self.dt * (v_k / self.L_trailer) * ca.sin(theta - phi)

            st_next = ca.vertcat(x_next, y_next, theta_next, phi_next)
            g.append(X[:, k + 1] - st_next)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {"x": opt_vars, "f": cost, "g": ca.vertcat(*g), "p": P}

        self.solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 100,
                "ipopt.tol": 1e-3,
            },
        )

        self.nx = 4 * (self.N + 1)
        self.nu = 2 * self.N

        # State bounds (unconstrained)
        self.base_lbx = []
        self.base_ubx = []
        for _ in range(self.N + 1):
            self.base_lbx += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
            self.base_ubx += [ca.inf, ca.inf, ca.inf, ca.inf]

        self.lbg = [0.0] * (4 * (self.N + 1))
        self.ubg = [0.0] * (4 * (self.N + 1))
        self.last_solution = np.zeros(self.nx + self.nu)

    def get_control_bounds(self):
        lbx = list(self.base_lbx)
        ubx = list(self.base_ubx)
        for _ in range(self.N):
            lbx += [-self.max_reverse_steer, self.v_min_reverse]
            ubx += [self.max_reverse_steer, self.v_max_reverse]
        return lbx, ubx

    def shift(self, sol):
        X = sol[:self.nx].reshape(self.N + 1, 4)
        U = sol[self.nx:].reshape(self.N, 2)
        X = np.vstack([X[1:], X[-1]])
        U = np.vstack([U[1:], U[-1]])
        return np.concatenate([X.flatten(), U.flatten()])

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_tracking(self, pose, closest_point, tracking_error, yaw_error, target_v, v, delta):
        if not hasattr(self, "tracking_log"):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        articulation = self.wrap_angle(pose[2] - self.trailer_yaw_est)

        self.tracking_log.write(
            f"{now:.3f},"
            f"{self.closest_idx},"
            f"{pose[0]:.4f},"
            f"{pose[1]:.4f},"
            f"{pose[2]:.4f},"
            f"{self.trailer_yaw_est:.4f},"
            f"{math.degrees(articulation):.2f},"
            f"{closest_point[0]:.4f},"
            f"{closest_point[1]:.4f},"
            f"{closest_point[2]:.4f},"
            f"{tracking_error:.4f},"
            f"{math.degrees(yaw_error):.2f},"
            f"{target_v:.4f},"
            f"{v:.4f},"
            f"{delta:.4f},"
            f"{self.drive_state},"
            f"{self.depth_emergency},"
            f"{self.motion_enabled}\n"
        )
        self.tracking_log.flush()

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------
    def _control_loop(self):
        # --- Mission complete ---
        if self.mission_done:
            self.stop()
            return

        # --- Depth emergency ---
        if self.depth_emergency:
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().warn(
                "Immediate stop: depth emergency", throttle_duration_sec=0.5
            )
            return

        # --- Motion not enabled ---
        if not self.motion_enabled or self.drive_state in [
            "EMERGENCY_STOP",
            "WAIT_FOR_CLEAR",
            "STARTUP_WAIT",
            "LIDAR_TIMEOUT",
            "SENSOR_WAIT",
        ]:
            self.stop()
            return

        # --- Get pose ---
        pose = self._get_pose()
        if pose is None:
            self.stop()
            return

        # --- Trailer yaw estimate ---
        self.update_trailer_yaw_estimate(pose[2])
        articulation = self.wrap_angle(pose[2] - self.trailer_yaw_est)

        # --- Jackknife protection ---
        if abs(articulation) > self.max_articulation:
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().error(
                f"Jackknife protection stop. "
                f"articulation={math.degrees(articulation):.1f} deg "
                f"(limit={math.degrees(self.max_articulation):.1f} deg)",
                throttle_duration_sec=0.5,
            )
            return

        # --- Closest point on trajectory ---
        self.closest_idx = PathUtils.closest_point(pose, self.trajectory, self.closest_idx)
        self.publish_current_idx()

        closest_point = self.trajectory[self.closest_idx]
        tracking_error = math.hypot(pose[0] - closest_point[0], pose[1] - closest_point[1])
        yaw_error = self.wrap_angle(pose[2] - closest_point[2])

        yaw_stable_msg = Bool()
        yaw_stable_msg.data = bool(abs(yaw_error) < math.radians(8.0))
        self.yaw_stable_pub.publish(yaw_stable_msg)

        # --- Trajectory completion ---
        if self.closest_idx >= len(self.trajectory) - 5:
            self.get_logger().warn("Reverse trajectory complete. Mission done.")
            self.mission_done = True
            self.stop()
            return

        # --- Reference and target speed ---
        ref = PathUtils.build_reference(self.trajectory, self.closest_idx, self.N)
        target_v = self.reverse_speed  # constant slow reverse speed

        # --- Build parameter vector ---
        current_state = np.array(
            [pose[0], pose[1], pose[2], self.trailer_yaw_est], dtype=float
        )
        p = np.concatenate(
            [
                current_state,
                np.array([self.prev_delta, self.prev_v, target_v], dtype=float),
                ref,
            ]
        )

        # --- Solve MPC ---
        try:
            lbx, ubx = self.get_control_bounds()
            sol = self.solver(
                x0=self.last_solution,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p,
            )

            solx = sol["x"].full().flatten()

            if not np.all(np.isfinite(solx)):
                self.get_logger().error("MPC returned non-finite solution — stopping")
                self.stop()
                self.reset_mpc_memory()
                return

            self.last_solution = self.shift(solx)
            u = solx[self.nx:self.nx + self.nu].reshape(self.N, 2)

            delta = float(np.clip(u[0, 0], -self.max_reverse_steer, self.max_reverse_steer))
            v = float(np.clip(u[0, 1], self.v_min_reverse, self.v_max_reverse))
            v = min(v, 0.0)  # enforce reverse only, never forward

            self.prev_delta = delta
            self.prev_v = v

            self.log_tracking(
                pose=pose,
                closest_point=closest_point,
                tracking_error=tracking_error,
                yaw_error=yaw_error,
                target_v=target_v,
                v=v,
                delta=delta,
            )

            cmd = Twist()
            cmd.linear.x = v
            if self.publish_steering_angle:
                cmd.angular.z = delta
            else:
                cmd.angular.z = v / self.L_car * math.tan(delta)
            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"mode=REVERSE | "
                f"idx={self.closest_idx} | "
                f"track_err={tracking_error:.3f} m | "
                f"yaw_err={math.degrees(yaw_error):.1f} deg | "
                f"articulation={math.degrees(articulation):.1f} deg | "
                f"v={v:.3f} | "
                f"delta={delta:.3f}",
                throttle_duration_sec=0.5,
            )

        except Exception as e:
            self.get_logger().error(f"MPC solve failed: {e}")
            self.stop()
            self.reset_mpc_memory()


def main():
    rclpy.init()
    node = QCarTrailerPathMPC()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()
    if hasattr(node, "tracking_log"):
        node.tracking_log.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()