#!/usr/bin/env python3

import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException


class QCar2MPC(Node):
    def __init__(self):
        super().__init__('qcar2_mpc_node')

        # ---------------- CONFIG ----------------
        self.trajectory_file = '/home/nvidia/ros2_ws/recorded_path_mpc_final.npy'

        self.N = 25
        self.dt = 0.08
        self.L = 0.256

        self.v_min = 0.2
        self.v_max = 0.7

        self.max_steer = 0.55

        # ---------------- COST WEIGHTS ----------------
        self.w_pos = 350.0
        self.w_yaw = 50.0
        self.w_delta = 25.0
        self.w_speed = 15.0
        self.w_control = 2.0

        # ---------------- STATE ----------------
        self.trajectory = []
        self.closest_idx = 0
        self.last_solution = None

        # ✅ FIX: initialize missing variables
        self.prev_delta = 0.0
        self.prev_v = self.v_min

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self._load_trajectory()

        if len(self.trajectory) < self.N + 1:
            raise RuntimeError("Trajectory too short")

        self._setup_optimizer()

        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("MPC READY (stable version)")

    # ---------------- UTIL ----------------
    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def stop(self):
        self.cmd_pub.publish(Twist())

    # ---------------- TRAJECTORY ----------------
    def _load_trajectory(self):
        data = np.load(self.trajectory_file)
        self.trajectory = data.tolist()
        self.get_logger().info(f"Loaded {len(self.trajectory)} points")

    # ---------------- POSE ----------------
    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )

            tr = t.transform.translation
            q = t.transform.rotation

            yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
            return np.array([tr.x, tr.y, yaw], dtype=float)

        except TransformException:
            return None

    # ---------------- MPC SETUP ----------------
    def _setup_optimizer(self):

        X = ca.SX.sym('X', 3, self.N + 1)
        U = ca.SX.sym('U', 2, self.N)

        P = ca.SX.sym('P', 3 + 2 + self.N * 3)

        cost = 0
        g = []

        # initial constraint
        g.append(X[:, 0] - P[0:3])

        prev_delta = P[3]
        prev_v = P[4]

        for k in range(self.N):

            st = X[:, k]
            delta_k = U[0, k]
            v_k = U[1, k]

            idx = 5 + 3 * k
            x_ref = P[idx]
            y_ref = P[idx + 1]
            yaw_ref = P[idx + 2]

            # ---------------- POSITION (highest priority) ----------------
            pos_err = (st[0] - x_ref)**2 + (st[1] - y_ref)**2
            cost += self.w_pos * ca.log(1 + pos_err)

            # ---------------- HEADING ----------------
            yaw_err = ca.atan2(ca.sin(st[2] - yaw_ref),
                               ca.cos(st[2] - yaw_ref))
            cost += self.w_yaw * yaw_err**2

            # ---------------- STEERING SMOOTHNESS ----------------
            if k == 0:
                ddelta = delta_k - prev_delta
                dv = v_k - prev_v
            else:
                ddelta = delta_k - U[0, k-1]
                dv = v_k - U[1, k-1]

            cost += self.w_delta * ddelta**2

            # ---------------- SPEED (stable form) ----------------
            cost += self.w_speed * (self.v_max - v_k)**2
            cost += 2.0 * dv**2

            # ---------------- CONTROL EFFORT ----------------
            cost += self.w_control * delta_k**2

            # ---------------- DYNAMICS ----------------
            st_next = ca.vertcat(
                st[0] + self.dt * v_k * ca.cos(st[2]),
                st[1] + self.dt * v_k * ca.sin(st[2]),
                st[2] + self.dt * (v_k / self.L) * ca.tan(delta_k)
            )

            g.append(X[:, k+1] - st_next)

        OPT = ca.vertcat(ca.reshape(X, -1, 1),
                         ca.reshape(U, -1, 1))

        nlp = {'x': OPT, 'f': cost, 'g': ca.vertcat(*g), 'p': P}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100
        })

        self.nx = 3 * (self.N + 1)
        self.nu = 2 * self.N

        # bounds
        self.lbx = []
        self.ubx = []

        for _ in range(self.N + 1):
            self.lbx += [-ca.inf, -ca.inf, -ca.inf]
            self.ubx += [ca.inf, ca.inf, ca.inf]

        for _ in range(self.N):
            self.lbx += [-self.max_steer, self.v_min]
            self.ubx += [ self.max_steer, self.v_max]

        self.lbg = [0.0] * (3 * (self.N + 1))
        self.ubg = [0.0] * (3 * (self.N + 1))

        self.last_solution = np.zeros(self.nx + self.nu)

    # ---------------- REF ----------------
    def _closest(self, pose):
        best = self.closest_idx
        window = 30

        for i in range(best, min(best + window, len(self.trajectory))):
            p = self.trajectory[i]
            if np.hypot(pose[0]-p[0], pose[1]-p[1]) < \
               np.hypot(pose[0]-self.trajectory[best][0],
                        pose[1]-self.trajectory[best][1]):
                best = i

        return best

    def _ref(self, idx):
        ref = []
        for i in range(self.N):
            ref.extend(self.trajectory[(idx + i) % len(self.trajectory)])
        return np.array(ref)

    # ---------------- SHIFT ----------------
    def shift(self, sol):
        X = sol[:self.nx].reshape(self.N + 1, 3)
        U = sol[self.nx:].reshape(self.N, 2)

        X = np.vstack([X[1:], X[-1]])
        U = np.vstack([U[1:], U[-1]])

        return np.concatenate([X.flatten(), U.flatten()])

    # ---------------- LOOP ----------------
    def _control_loop(self):

        pose = self._get_pose()
        if pose is None:
            self.stop()
            return

        self.closest_idx = self._closest(pose)
        ref = self._ref(self.closest_idx)

        p = np.concatenate([
            pose,
            np.array([self.prev_delta, self.prev_v]),
            ref
        ])

        try:
            sol = self.solver(
                x0=self.last_solution,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )

            solx = sol['x'].full().flatten()
            self.last_solution = self.shift(solx)

            u = solx[self.nx:self.nx + self.nu]
            u = u.reshape(2, self.N)

            delta = float(np.clip(u[0, 0], -self.max_steer, self.max_steer))
            v = float(np.clip(u[1, 0], self.v_min, self.v_max))

            self.prev_delta = delta
            self.prev_v = v

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = v / self.L * math.tan(delta)
            self.cmd_pub.publish(cmd)

            self.get_logger().info(f"v={v:.2f}, delta={delta:.2f}")

        except Exception as e:
            self.get_logger().error(str(e))
            self.stop()


def main():
    rclpy.init()
    node = QCar2MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()