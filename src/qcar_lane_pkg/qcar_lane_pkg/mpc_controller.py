#!/usr/bin/env python3

import json
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
        self.trajectory_file = '/home/nvidia/ros2_ws/recorded_path_mpc_smooth.json'

        self.N = 35
        self.dt = 0.08
        self.L = 0.256
        self.v_ref = 0.4
        self.max_steer = 0.5

        # cost weights
        self.w_pos = 350.0
        self.w_yaw = 50.0
        self.w_delta = 2.0
        self.w_delta_rate = 20.0

        self.map_frame = 'map'
        self.base_frame = 'base_link'

        # ---------------- STATE ----------------
        self.trajectory = []
        self.closest_idx = 0
        self.last_delta = 0.0
        self.last_solution = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        # load + setup
        self._load_trajectory()

        if len(self.trajectory) < self.N + 1:
            raise RuntimeError("Trajectory too short")

        self._setup_optimizer()

        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("MPC READY (oscillation-fixed version)")

    # ---------------- UTILS ----------------
    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    # ---------------- TRAJECTORY ----------------
    def _load_trajectory(self):
        with open(self.trajectory_file, 'r') as f:
            poses = json.load(f)

        self.trajectory = []
        for p in poses:
            yaw = self.quat_to_yaw(p.get('qx', 0.0),
                                   p.get('qy', 0.0),
                                   p['qz'],
                                   p['qw'])
            self.trajectory.append([p['x'], p['y'], yaw])

        self.get_logger().info(f"Loaded {len(self.trajectory)} points")

    # ---------------- POSE ----------------
    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
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
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        states = ca.vertcat(x, y, yaw)

        delta = ca.SX.sym('delta')
        controls = ca.vertcat(delta)

        P = ca.SX.sym('P', 3 + 1 + self.N * 3)

        X = ca.SX.sym('X', 3, self.N + 1)
        U = ca.SX.sym('U', 1, self.N)

        rhs = ca.vertcat(
            self.v_ref * ca.cos(yaw),
            self.v_ref * ca.sin(yaw),
            self.v_ref / self.L * ca.tan(delta)
        )

        f = ca.Function('f', [states, controls], [rhs])

        cost = 0
        g = []

        g.append(X[:, 0] - P[0:3])
        prev_delta = P[3]

        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]

            idx = 4 + 3 * k
            x_ref = P[idx]
            y_ref = P[idx + 1]
            yaw_ref = P[idx + 2]

            yaw_err = ca.atan2(ca.sin(st[2] - yaw_ref),
                               ca.cos(st[2] - yaw_ref))

            cost += self.w_pos * ((st[0] - x_ref)**2 + (st[1] - y_ref)**2)
            cost += self.w_yaw * (yaw_err**2)
            cost += self.w_delta * (con[0]**2)

            if k == 0:
                rate = con[0] - prev_delta
            else:
                rate = con[0] - U[:, k-1][0]

            cost += self.w_delta_rate * (rate**2)

            # NEW: curvature penalty (kills straight-line oscillation)
            cost += 5.0 * ca.power(ca.tan(con[0]), 2)

            st_next = st + self.dt * f(st, con)
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
        self.nu = self.N

        self.lbx = [-ca.inf] * self.nx + [-self.max_steer] * self.nu
        self.ubx = [ca.inf] * self.nx + [self.max_steer] * self.nu

        self.lbg = [0.0] * 3 * (self.N + 1)
        self.ubg = [0.0] * 3 * (self.N + 1)

        self.last_solution = np.zeros(self.nx + self.nu)

    # ---------------- WARM START SHIFT ----------------
    def shift(self, sol):
        X = sol[:self.nx].reshape(self.N + 1, 3)
        U = sol[self.nx:].reshape(self.N, 1)

        X = np.vstack([X[1:], X[-1]])
        U = np.vstack([U[1:], U[-1]])

        return np.concatenate([X.flatten(), U.flatten()])

    # ---------------- REF ----------------
    def _closest(self, pose):
        best = self.closest_idx
        window = 40

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
            p = self.trajectory[(idx + i) % len(self.trajectory)]
            ref.extend(p)
        return np.array(ref)

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
            np.array([self.last_delta]),
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

            u = solx[self.nx:self.nx+self.nu]
            delta = float(np.clip(u[0], -self.max_steer, self.max_steer))

            cmd = Twist()
            cmd.linear.x = self.v_ref
            cmd.angular.z = self.v_ref / self.L * math.tan(delta)

            self.cmd_pub.publish(cmd)
            self.last_delta = delta

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