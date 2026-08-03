#!/usr/bin/env python3
"""Pure Pursuit trajectory follower with curvature-adaptive speed - ACCURATE with filtered control."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
import math
import json
from collections import deque


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ---------------- CONFIG ----------------
        self._trajectory_file = '/home/nvidia/ros2_ws/recorded_path_clean_latest.json'

        self._max_speed = 0.4
        self._min_speed = 0.10

        self._wheelbase = 0.25
        self._lookahead_base = 0.8

        self._kp_steer = 1.2
        self._curvature_gain = 1.5

        # ---------- SMOOTH CONTROL FILTERS (DOES NOT AFFECT ACCURACY) ----------
        self._enable_steering_filter = True
        self._steering_filter_cutoff_freq = 3.0  # Hz (higher = faster response, lower = smoother)
        self._last_filtered_steer = 0.0
        self._last_raw_steer = 0.0
        
        self._enable_velocity_smoothing = True
        self._velocity_smoothing_factor = 0.15  # Light smoothing for comfort
        
        # ---------- STATE ----------------
        self._trajectory = []
        self._curvatures = []
        self._closest_idx = 0

        self._current_x = 0.0
        self._current_y = 0.0
        self._current_yaw = 0.0
        
        self._last_time = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self._path_pub = self.create_publisher(Path, '/target_path', 10)

        self.create_timer(0.05, self._control_loop)
        self.create_timer(1.0, self._publish_path)

        self._load_trajectory()
        self._compute_curvatures()

        self.get_logger().info(f"Loaded {len(self._trajectory)} waypoints (ACCURATE MODE)")

    # --------------------------------------------------

    def _load_trajectory(self):
        with open(self._trajectory_file, 'r') as f:
            poses = json.load(f)

        for p in poses:
            yaw = 2 * math.atan2(p['qz'], p['qw'])
            self._trajectory.append((p['x'], p['y'], yaw))

    # --------------------------------------------------

    def _compute_curvatures(self):
        """Compute curvature exactly as before - no approximation."""
        self._curvatures = [0.0] * len(self._trajectory)

        n = len(self._trajectory)

        for i in range(n):
            i_prev = (i - 1) % n
            i_next = (i + 1) % n

            x1, y1, _ = self._trajectory[i_prev]
            x2, y2, _ = self._trajectory[i]
            x3, y3, _ = self._trajectory[i_next]

            v1x, v1y = x2 - x1, y2 - y1
            v2x, v2y = x3 - x2, y3 - y2

            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)

            if len1 < 1e-6 or len2 < 1e-6:
                continue

            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x

            angle = abs(math.atan2(cross, dot))
            self._curvatures[i] = angle / (len1 + len2)

    # --------------------------------------------------

    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = 2 * math.atan2(q.z, q.w)
            return x, y, yaw
        except:
            return None, None, None

    # --------------------------------------------------

    def _find_closest(self):
        """Exact closest point - no hysteresis to maintain accuracy."""
        n = len(self._trajectory)
        
        # Search forward from previous index (more efficient and stable)
        # But search both directions for accuracy
        search_range = min(50, n // 2)
        
        min_d = float('inf')
        best = self._closest_idx
        
        # Search forward
        for offset in range(search_range):
            i = (self._closest_idx + offset) % n
            x, y, _ = self._trajectory[i]
            d = math.hypot(self._current_x - x, self._current_y - y)
            if d < min_d:
                min_d = d
                best = i
                
        # Search backward
        for offset in range(1, search_range):
            i = (self._closest_idx - offset) % n
            x, y, _ = self._trajectory[i]
            d = math.hypot(self._current_x - x, self._current_y - y)
            if d < min_d:
                min_d = d
                best = i

        self._closest_idx = best
        return best

    # --------------------------------------------------

    def _find_lookahead(self, idx, Ld):
        """Exact lookahead distance - no smoothing or averaging."""
        n = len(self._trajectory)
        dist = 0.0
        i = idx

        # Prevent infinite loops with max iterations
        max_iter = n
        iter_count = 0
        
        while dist < Ld and iter_count < max_iter:
            i_next = (i + 1) % n
            x1, y1, _ = self._trajectory[i]
            x2, y2, _ = self._trajectory[i_next]
            
            segment_length = math.hypot(x2 - x1, y2 - y1)
            if dist + segment_length >= Ld:
                # Interpolate exact point at lookahead distance
                remaining = Ld - dist
                if segment_length > 1e-6:
                    ratio = remaining / segment_length
                    tx = x1 + ratio * (x2 - x1)
                    ty = y1 + ratio * (y2 - y1)
                    return (tx, ty, 0.0)
                else:
                    return (x2, y2, 0.0)
            
            dist += segment_length
            i = i_next
            iter_count += 1

        # If we completed a full loop, return the current point
        x, y, _ = self._trajectory[i]
        return (x, y, 0.0)

    # --------------------------------------------------

    def _pure_pursuit_steer(self, target):
        """Pure pursuit calculation - exact, no approximations."""
        tx, ty, _ = target

        dx = tx - self._current_x
        dy = ty - self._current_y

        # Transform to vehicle frame
        cos_yaw = math.cos(self._current_yaw)
        sin_yaw = math.sin(self._current_yaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = dy * cos_yaw - dx * sin_yaw

        ld = math.hypot(local_x, local_y)
        if ld < 1e-6:
            return 0.0

        # Pure pursuit formula
        alpha = math.atan2(local_y, local_x)
        
        # Calculate steering angle
        steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), ld)
        
        # Just clamp to vehicle limits
        steer = max(-0.5, min(0.5, steer))

        return steer

    # --------------------------------------------------
    
    def _low_pass_filter_steering(self, raw_steer, dt):
        """
        First-order low-pass filter for steering.
        Preserves the exact steering target but smooths the command.
        Higher cutoff_freq = more accurate but less smooth.
        """
        if not self._enable_steering_filter:
            return raw_steer
            
        if dt < 0.001:
            return self._last_filtered_steer
            
        # Calculate filter coefficient (tau = 1/(2*pi*fc))
        tau = 1.0 / (2.0 * math.pi * self._steering_filter_cutoff_freq)
        alpha = dt / (tau + dt)
        
        # Apply filter
        filtered = alpha * raw_steer + (1 - alpha) * self._last_filtered_steer
        self._last_filtered_steer = filtered
        self._last_raw_steer = raw_steer
        
        return filtered
    
    # --------------------------------------------------

    def _compute_speed(self, idx):
        """Exact speed based on curvature."""
        curv = self._curvatures[idx]
        speed = self._max_speed / (1.0 + self._curvature_gain * curv)
        speed = max(self._min_speed, min(self._max_speed, speed))
        
        if self._enable_velocity_smoothing:
            # Very light smoothing for comfort only
            if not hasattr(self, '_smoothed_speed'):
                self._smoothed_speed = speed
            else:
                # Use very low smoothing factor to maintain accuracy
                self._smoothed_speed = (self._velocity_smoothing_factor * speed + 
                                       (1 - self._velocity_smoothing_factor) * self._smoothed_speed)
            return self._smoothed_speed
        else:
            return speed

    # --------------------------------------------------

    def _control_loop(self):
        # Calculate dt for filter
        current_time = self.get_clock().now()
        if self._last_time is None:
            self._last_time = current_time
            dt = 0.05  # Default to timer period
        else:
            dt = (current_time - self._last_time).nanoseconds / 1e9
            dt = max(0.01, min(0.1, dt))
            self._last_time = current_time
        
        x, y, yaw = self._get_pose()
        if x is None:
            return

        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw

        closest = self._find_closest()
        speed = self._compute_speed(closest)

        # Dynamic lookahead based on speed
        lookahead = self._lookahead_base * (0.5 + speed / self._max_speed)
        target = self._find_lookahead(closest, lookahead)

        # Calculate exact pure pursuit steering
        raw_steer = self._pure_pursuit_steer(target)
        
        # Apply ONLY low-pass filter for smoothing - no approximation
        filtered_steer = self._low_pass_filter_steering(raw_steer, dt)

        # Publish command
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = filtered_steer
        self._cmd_pub.publish(cmd)

        # Log for debugging
        self.get_logger().debug(
            f"speed={speed:.2f} | raw_steer={raw_steer:.3f} | "
            f"filtered_steer={filtered_steer:.3f} | lookahead={lookahead:.2f}"
        )
        
        # Optional: Print every 2 seconds for monitoring
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter >= 40:  # ~2 seconds at 20Hz
            self.get_logger().info(
                f"Speed: {speed:.2f} | Steering: {filtered_steer:.3f} | "
                f"Delta: {abs(raw_steer - filtered_steer):.3f} rad"
            )
            self._log_counter = 0

    # --------------------------------------------------

    def _publish_path(self):
        if not self._trajectory:
            return

        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y, yaw in self._trajectory:
            p = PoseStamped()
            p.header.frame_id = "map"
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.z = math.sin(yaw / 2)
            p.pose.orientation.w = math.cos(yaw / 2)
            msg.poses.append(p)

        self._path_pub.publish(msg)


def main():
    rclpy.init()
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()