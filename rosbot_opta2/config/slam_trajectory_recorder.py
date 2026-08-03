#!/usr/bin/env python3
"""Record trajectory using SLAM pose (map frame)."""

from pathlib import Path
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import math
import csv

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY_PATH = WORKSPACE_ROOT / 'config' / 'slam_trajectory.csv'


class SlamTrajectoryRecorderNode(Node):

    def __init__(self):
        super().__init__('slam_trajectory_recorder_node')
        
        # Config
        self._trajectory_file = str(TRAJECTORY_PATH)
        self._min_record_distance = 0.02  # Record every 2cm
        
        # TF2 listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        # Current pose
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_theta = 0.0
        
        # Last recorded pose
        self._last_recorded_x = None
        self._last_recorded_y = None
        
        # CSV file
        self._traj_file = open(self._trajectory_file, 'w', newline='')
        self._traj_writer = csv.writer(self._traj_file)
        self._traj_writer.writerow(['x', 'y', 'theta'])
        
        # Timer to poll TF
        self.create_timer(0.05, self._tf_cb)  # 20Hz
        
        self._point_count = 0
        self.get_logger().info(f'Recording SLAM trajectory to {self._trajectory_file}')
        self.get_logger().info('Drive the robot manually. Press Ctrl+C to stop.')

    def _tf_cb(self):
        """Get pose from TF and record."""
        try:
            transform = self._tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return
        
        # Extract position
        self._current_x = transform.transform.translation.x
        self._current_y = transform.transform.translation.y
        
        # Quaternion to yaw
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_theta = math.atan2(siny_cosp, cosy_cosp)
        
        self._record_position()

    def _record_position(self):
        """Record position if moved enough."""
        if self._last_recorded_x is None:
            # First point
            self._save_point()
            return
        
        dx = self._current_x - self._last_recorded_x
        dy = self._current_y - self._last_recorded_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist >= self._min_record_distance:
            self._save_point()

    def _save_point(self):
        """Save current point to CSV."""
        self._traj_writer.writerow([
            f'{self._current_x:.4f}',
            f'{self._current_y:.4f}',
            f'{self._current_theta:.4f}'
        ])
        self._traj_file.flush()
        
        self._last_recorded_x = self._current_x
        self._last_recorded_y = self._current_y
        self._point_count += 1
        
        self.get_logger().info(
            f'Point {self._point_count}: x={self._current_x:.3f}, y={self._current_y:.3f}, θ={math.degrees(self._current_theta):.1f}°'
        )

    def close(self):
        self._traj_file.close()
        self.get_logger().info(f'Saved {self._point_count} points to {self._trajectory_file}')


def main(args=None):
    rclpy.init(args=args)
    node = SlamTrajectoryRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
