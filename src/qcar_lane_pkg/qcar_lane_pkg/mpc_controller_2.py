#!/usr/bin/env python3

import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String,Int32
from tf2_ros import Buffer, TransformListener, TransformException

from qcar_science_night_pkg.path_utils import PathUtils


class QCar2PathMPC(Node):

    def __init__(self):
        super().__init__("qcar2_path_mpc_node")

        self.forward_trajectory_file = "/home/nvidia/ros2_ws/recorded_path_mpc_new29_smooth_endyaw_fixed.npy"
        self.reverse_trajectory_file = "/home/nvidia/ros2_ws/recorded_path_reverse27.npy"

        # Set True only when you want reverse enabled.
        self.enable_reverse = False

        self.N = 25
        self.dt = 0.08
        self.L = 0.256

        self.v_min = -0.30
        self.v_max = 0.75
        self.v_curve_min = 0.40

        self.reverse_speed = -0.10
        self.reverse_done_distance = 0.12

        self.max_steer = 0.55
        self.publish_steering_angle = True

        self.reverse_mode = False
        self.mission_done = False

        self.pickup_idx = 610
        self.pickup_wait_sec = 5.0
        self.pickup_done = False
        self.pickup_active = False
        self.pickup_start_time = None
        self.pickup_tolerance_idx = 5

        self.overtake_curve_limit = 0.35
        self.overtake_mean_curve_limit = 0.15

        self.lane_centering_curve_limit = 0.45
        self.current_max_curvature = 999.0
        self.current_mean_curvature = 999.0
        self.max_lane_offset = 0.04

        self.w_pos = 180.0
        self.w_yaw = 70.0
        self.w_delta_rate = 45.0
        self.w_speed_rate = 1
        self.w_speed_tracking = 50.0
        self.w_control = 3.0

        self.closest_idx = 0
        self.last_solution = None

        self.prev_delta = 0.0
        self.prev_v = 0.0

        self.motion_enabled = False
        self.drive_state = "DRIVE"
        self.depth_emergency = False

        self.avoidance_offset = 0.0
        self.avoidance_offset_filtered = 0.0
        self.avoidance_alpha = 0.35

        self.lane_offset = 0.0
        self.lane_valid = False
        self.lane_offset_filtered = 0.0
        self.lane_alpha = 0.08
        
        self.startup_align_steps = 35
        self.startup_counter = 0
        self.startup_v_max = 0.22

        self.tracking_log_path = "/home/nvidia/ros2_ws/mpc_tracking_log.csv"
        self.tracking_log = open(self.tracking_log_path, "w")
        self.tracking_log.write(
            "time,mode,idx,x,y,yaw,ref_x,ref_y,ref_yaw,"
            "track_error,yaw_error_deg,target_v,v,delta,drive_state,"
            "lane_offset,lane_active,mean_curvature\n"
        )
        self.tracking_log.flush()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.allow_overtake_pub = self.create_publisher(Bool, "/allow_overtake", 10)
        self.idx_pub = self.create_publisher(Int32, "/current_path_idx", 10)

        self.create_subscription(Bool, "/motion_enable", self.motion_callback, 10)
        self.create_subscription(Float32, "/avoidance_offset", self.avoidance_callback, 10)
        self.create_subscription(String, "/drive_state", self.drive_state_callback, 10)
        self.create_subscription(Float32, "/lane_center_offset", self.lane_offset_callback, 10)
        self.create_subscription(Bool, "/lane_center_valid", self.lane_valid_callback, 10)
        self.create_subscription(Bool, "/depth_emergency_stop", self.depth_emergency_callback, 10)

        self.trajectory = PathUtils.load_trajectory(self.forward_trajectory_file)

        if len(self.trajectory) < self.N + 2:
            raise RuntimeError(
                f"Trajectory too short. Need {self.N + 2}, got {len(self.trajectory)}"
            )

        self._setup_optimizer()

        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            "QCar MPC ready: your values + pickup + lane centering + reverse flag"
        )

    def motion_callback(self, msg):
        self.motion_enabled = bool(msg.data)

    def avoidance_callback(self, msg):
        self.avoidance_offset = float(msg.data)

    def drive_state_callback(self, msg):
        self.drive_state = str(msg.data)

    def lane_offset_callback(self, msg):
        self.lane_offset = float(msg.data)

    def lane_valid_callback(self, msg):
        self.lane_valid = bool(msg.data)

    def depth_emergency_callback(self, msg):
        self.depth_emergency = bool(msg.data)

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        self.prev_v = 0.0
        self.prev_delta = 0.0

    def reset_mpc_memory(self):
        self.last_solution = np.zeros(self.nx + self.nu)

        self.avoidance_offset_filtered = 0.0
        self.lane_offset_filtered = 0.0
        self.prev_v = 0.0
        self.prev_delta = 0.0
        self.startup_counter = 0

    def switch_to_reverse(self):
        self.get_logger().warn("Forward complete. Switching to reverse trajectory.")

        self.trajectory = PathUtils.load_trajectory(self.reverse_trajectory_file)

        if len(self.trajectory) < self.N + 2:
            raise RuntimeError(
                f"Reverse trajectory too short. Need {self.N + 2}, got {len(self.trajectory)}"
            )

        self.closest_idx = 0
        self.reverse_mode = True
        self.drive_state = "DRIVE"

        self.reset_mpc_memory()

    def publish_overtake_disabled(self):
        msg = Bool()
        msg.data = False
        self.allow_overtake_pub.publish(msg)

    def handle_pickup_stop(self):
        if self.reverse_mode:
            return False

        if self.pickup_active:
            elapsed = (
                self.get_clock().now() - self.pickup_start_time
            ).nanoseconds * 1e-9

            self.stop()

            if elapsed >= self.pickup_wait_sec:
                self.pickup_active = False
                self.reset_mpc_memory()
                self.get_logger().warn("Pickup stop complete. Continuing path.")

            return True

        if (
            not self.pickup_done
            and abs(self.closest_idx - self.pickup_idx) <= self.pickup_tolerance_idx
        ):
            self.pickup_active = True
            self.pickup_done = True
            self.pickup_start_time = self.get_clock().now()

            self.stop()
            self.get_logger().warn(
                f"Pickup stop started at idx={self.closest_idx}"
            )

            return True

        return False

    def log_tracking_error(
        self,
        pose,
        closest_point,
        tracking_error,
        yaw_error,
        target_v,
        v,
        delta,
    ):
        if not hasattr(self, "tracking_log"):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        mode = "REVERSE" if self.reverse_mode else "FORWARD"
        lane_active = (
            self.current_mean_curvature < self.lane_centering_curve_limit
            and self.lane_valid
            and self.drive_state == "DRIVE"
        )

        self.tracking_log.write(
            f"{now:.3f},"
            f"{mode},"
            f"{self.closest_idx},"
            f"{pose[0]:.4f},"
            f"{pose[1]:.4f},"
            f"{pose[2]:.4f},"
            f"{closest_point[0]:.4f},"
            f"{closest_point[1]:.4f},"
            f"{closest_point[2]:.4f},"
            f"{tracking_error:.4f},"
            f"{math.degrees(yaw_error):.2f},"
            f"{target_v:.4f},"
            f"{v:.4f},"
            f"{delta:.4f},"
            f"{self.drive_state},"
            f"{self.lane_offset_filtered:.4f},"
            f"{lane_active},"
            f"{self.current_mean_curvature:.4f}\n"
        )

        self.tracking_log.flush()

    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
            )

            tr = t.transform.translation
            q = t.transform.rotation

            yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)

            return np.array([tr.x, tr.y, yaw], dtype=float)

        except TransformException as e:
            self.get_logger().warn(f"TF unavailable: {e}")
            return None

    def _setup_optimizer(self):
        X = ca.SX.sym("X", 3, self.N + 1)
        U = ca.SX.sym("U", 2, self.N)

        P = ca.SX.sym("P", 3 + 2 + 1 + self.N * 3)

        cost = 0
        g = []

        g.append(X[:, 0] - P[0:3])

        prev_delta = P[3]
        prev_v = P[4]
        target_v = P[5]

        for k in range(self.N):
            st = X[:, k]

            delta_k = U[0, k]
            v_k = U[1, k]

            idx = 6 + 3 * k

            x_ref = P[idx]
            y_ref = P[idx + 1]
            yaw_ref = P[idx + 2]

            pos_err = (
                (st[0] - x_ref) ** 2
                + (st[1] - y_ref) ** 2
            )

            cost += self.w_pos * ca.log(1.0 + pos_err)

            yaw_err = ca.atan2(
                ca.sin(st[2] - yaw_ref),
                ca.cos(st[2] - yaw_ref),
            )

            cost += self.w_yaw * yaw_err ** 2

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

            st_next = ca.vertcat(
                st[0] + self.dt * v_k * ca.cos(st[2]),
                st[1] + self.dt * v_k * ca.sin(st[2]),
                st[2] + self.dt * (v_k / self.L) * ca.tan(delta_k),
            )

            g.append(X[:, k + 1] - st_next)

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
        )

        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": ca.vertcat(*g),
            "p": P,
        }

        self.solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 80,
                "ipopt.tol": 1e-3,
            },
        )

        self.nx = 3 * (self.N + 1)
        self.nu = 2 * self.N

        self.lbx = []
        self.ubx = []

        for _ in range(self.N + 1):
            self.lbx += [-ca.inf, -ca.inf, -ca.inf]
            self.ubx += [ca.inf, ca.inf, ca.inf]

        for _ in range(self.N):
            self.lbx += [-self.max_steer, self.v_min]
            self.ubx += [self.max_steer, self.v_max]

        self.lbg = [0.0] * (3 * (self.N + 1))
        self.ubg = [0.0] * (3 * (self.N + 1))

        self.last_solution = np.zeros(self.nx + self.nu)

    def shift(self, sol):
        X = sol[:self.nx].reshape(self.N + 1, 3)
        U = sol[self.nx:].reshape(self.N, 2)

        X = np.vstack([X[1:], X[-1]])
        U = np.vstack([U[1:], U[-1]])

        return np.concatenate([X.flatten(), U.flatten()])

    def publish_overtake_permission(self):
        if self.reverse_mode:
            self.publish_overtake_disabled()
            return False, 0.0, 0.0

        last_idx = len(self.trajectory) - 1

        curvatures = np.array(
            [
                abs(
                    self.trajectory[
                        min(self.closest_idx + k, last_idx)
                    ][3]
                )
                for k in range(self.N)
            ],
            dtype=float,
        )

        max_curvature = float(np.max(curvatures))
        mean_curvature = float(np.mean(curvatures))

        self.current_max_curvature = max_curvature
        self.current_mean_curvature = mean_curvature

        allow_overtake = (
            max_curvature < self.overtake_curve_limit
            and mean_curvature < self.overtake_mean_curve_limit
        )

        msg = Bool()
        msg.data = bool(allow_overtake)
        self.allow_overtake_pub.publish(msg)

        return allow_overtake, max_curvature, mean_curvature

    def apply_state_machine_reference(self, ref, target_v):
        if self.reverse_mode:
            return ref, self.reverse_speed, False

        if (
            not self.motion_enabled
            or self.depth_emergency
            or self.drive_state in [
                "EMERGENCY_STOP",
                "WAIT_FOR_CLEAR",
                "WAIT_FOR_LEFT_CLEAR",
                "STARTUP_WAIT",
                "LIDAR_TIMEOUT",
                "SENSOR_WAIT",
            ]
            or self.avoidance_offset >= 999.0
        ):
            return None, 0.0, True

        if self.drive_state in ["OVERTAKE_LEFT", "RETURN_RIGHT"]:
            self.avoidance_offset_filtered = (
                (1.0 - self.avoidance_alpha)
                * self.avoidance_offset_filtered
                + self.avoidance_alpha
                * self.avoidance_offset
            )

            if abs(self.avoidance_offset_filtered) > 0.02:
                ref = PathUtils.apply_offset(
                    ref,
                    self.N,
                    self.avoidance_offset_filtered,
                )

            self.lane_offset_filtered = 0.0

            target_v = min(target_v, 0.30)
            target_v = max(target_v, 0.25)

            return ref, target_v, False

        if self.drive_state == "DRIVE":
            self.avoidance_offset_filtered = 0.0

            is_straight = (
                self.current_mean_curvature
                < self.lane_centering_curve_limit
            )

            if (
                is_straight
                and self.lane_valid
            ):
                self.lane_offset_filtered = (
                    (1.0 - self.lane_alpha)
                    * self.lane_offset_filtered
                    + self.lane_alpha
                    * self.lane_offset
                )

                self.lane_offset_filtered = float(
                    np.clip(
                        self.lane_offset_filtered,
                        -self.max_lane_offset,
                        self.max_lane_offset,
                    )
                )

                if abs(self.lane_offset_filtered) < 0.005:
                    self.lane_offset_filtered = 0.0

                if abs(self.lane_offset_filtered) > 0.005:
                    ref = PathUtils.apply_offset(
                        ref,
                        self.N,
                        self.lane_offset_filtered,
                    )

            else:
                self.lane_offset_filtered *= 0.90

            return ref, target_v, False

        return None, 0.0, True

    def _control_loop(self):
        if self.mission_done:
            self.stop()
            return

        if self.depth_emergency:
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().warn(
                "Immediate stop: depth emergency",
                throttle_duration_sec=0.5,
            )
            return

        pose = self._get_pose()

        if pose is None:
            self.stop()
            return

        self.closest_idx = PathUtils.closest_point(
            pose,
            self.trajectory,
            self.closest_idx,
        )

        if self.handle_pickup_stop():
            return

        closest_point = self.trajectory[self.closest_idx]

        tracking_error = math.hypot(
            pose[0] - closest_point[0],
            pose[1] - closest_point[1],
        )

        yaw_error = PathUtils.wrap_angle(
            pose[2] - closest_point[2]
        )

        if (
            not self.reverse_mode
            and self.closest_idx >= len(self.trajectory) - 5
        ):
            if self.enable_reverse:
                self.switch_to_reverse()
                return

            self.get_logger().warn("Forward trajectory complete. Reverse disabled. Stopping.")
            self.mission_done = True
            self.stop()
            return

        if (
            self.reverse_mode
            and self.closest_idx >= len(self.trajectory) - 5
        ):
            self.get_logger().warn("Reverse trajectory complete. Stopping.")
            self.mission_done = True
            self.stop()
            return

        allow_overtake, max_curvature, mean_curvature = (
            self.publish_overtake_permission()
        )

        ref = PathUtils.build_reference(
            self.trajectory,
            self.closest_idx,
            self.N,
        )

        if self.reverse_mode:
            target_v = self.reverse_speed
        else:
            target_v = PathUtils.target_speed(
                self.trajectory,
                self.closest_idx,
                self.N,
                self.v_max,
                self.v_curve_min,
            )
        if (
            not self.reverse_mode
            and self.startup_counter < self.startup_align_steps
        ):
            target_v = min(target_v, self.startup_v_max)
            self.startup_counter += 1

        ref, target_v, should_stop = self.apply_state_machine_reference(
            ref,
            target_v,
        )

        if should_stop:
            self.stop()
            if self.drive_state not in [
                "WAIT_FOR_CLEAR",
                "EMERGENCY_STOP",
                "LIDAR_TIMEOUT",
                "SENSOR_WAIT",
                "STARTUP_WAIT"
            ]:
                self.reset_mpc_memory()
            return

        p = np.concatenate(
            [
                pose,
                np.array([
                    self.prev_delta,
                    self.prev_v,
                    target_v,
                ]),
                ref,
            ]
        )

        try:
            sol = self.solver(
                x0=self.last_solution,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p,
            )

            solx = sol["x"].full().flatten()

            if not np.all(np.isfinite(solx)):
                self.get_logger().error("MPC returned non-finite solution")
                self.stop()
                self.reset_mpc_memory()
                return

            self.last_solution = self.shift(solx)

            u = solx[
                self.nx:self.nx + self.nu
            ].reshape(
                self.N,
                2,
            )

            delta = float(
                np.clip(
                    u[0, 0],
                    -self.max_steer,
                    self.max_steer,
                )
            )

            v = float(
                np.clip(
                    u[0, 1],
                    self.v_min,
                    self.v_max,
                )
            )

            self.prev_delta = delta
            self.prev_v = v

            self.log_tracking_error(
                pose=pose,
                closest_point=closest_point,
                tracking_error=tracking_error,
                yaw_error=yaw_error,
                target_v=target_v,
                v=v,
                delta=delta,
            )
            idx_msg = Int32()
            idx_msg.data = int(self.closest_idx)
            self.idx_pub.publish(idx_msg)

            cmd = Twist()
            cmd.linear.x = v

            if self.publish_steering_angle:
                cmd.angular.z = delta
            else:
                cmd.angular.z = v / self.L * math.tan(delta)

            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"mode={'REVERSE' if self.reverse_mode else 'FORWARD'}, "
                f"state={self.drive_state}, "
                f"idx={self.closest_idx}, "
                f"track_err={tracking_error:.3f}, "
                f"yaw_err={math.degrees(yaw_error):.1f}, "
                f"v={v:.2f}, "
                f"target_v={target_v:.2f}, "
                f"delta={delta:.2f}, "
                f"lane={self.lane_offset_filtered:.3f}, "
                f"lane_valid={self.lane_valid}, "
                f"lane_active={self.current_mean_curvature < self.lane_centering_curve_limit}, "
                f"allow_overtake={allow_overtake}, "
                f"max_curv={max_curvature:.3f}, "
                f"mean_curv={mean_curvature:.3f}, "
                f"reverse_enabled={self.enable_reverse}",
                throttle_duration_sec=1.0,
            )

        except Exception as e:
            self.get_logger().error(f"MPC failed: {e}")
            self.stop()
            self.reset_mpc_memory()


def main():
    rclpy.init()

    node = QCar2PathMPC()

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