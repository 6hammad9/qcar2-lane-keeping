#!/usr/bin/env python3
"""
Trajectory follower with segment-based approach.

Handles multiple direction flips:
- Detects all flip points at load time
- Drives to each flip point as a goal
- Switches direction and continues to next flip point
"""

from pathlib import Path
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
import math
from enum import Enum
import subprocess

# Add package path for imports
import sys
import threading
import select
import time
import os
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_path not in sys.path:
    sys.path.insert(0, package_path)

from rosbot_lane.core.pure_pursuit import PurePursuitController, PurePursuitConfig
from rosbot_lane.core.trajectory import Trajectory


class State(Enum):
    WAITING_FOR_LOCALIZATION = 0
    ROTATE_TO_START = 1
    DRIVE_TO_START = 2
    ALIGN_AT_START = 3
    FOLLOWING_SEGMENT = 4
    WAYPOINT_PAUSE = 5
    ALIGN_AT_END = 6
    OVERTAKING = 7
    COMPLETE = 8


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = WORKSPACE_ROOT / 'config' / 'smoothed_trajectory.csv'
SOUND_DIR = WORKSPACE_ROOT / 'rosbot_lane' / 'core'


class TrajectoryFollowerNode(Node):

    def __init__(self):
        super().__init__('trajectory_follower_node')
        
        # ===== PARAMETERS =====
        trajectory_file = str(TRAJECTORY_PATH)
        
        # Pure Pursuit config
        pp_config = PurePursuitConfig(
            lookahead_distance=0.15,  # Shorter lookahead for tighter following
            min_lookahead=0.10,
            max_lookahead=0.15,
            max_speed=0.40,
            min_speed=0.10,
            curve_speed=0.12,
            reverse_speed=0.10,
            max_angular=1.0,
            kp_angular=1.5,
            rotate_first_threshold=0.1,
            goal_tolerance=0.01,
            angle_tolerance=0.05,
        )
        
        # Segment transition tolerance
        self._segment_goal_tolerance = 0.04

        # Segment-following lookahead distances (arc length along path)
        self._lookahead_forward = 0.30   # m
        self._lookahead_reverse = 0.15   # m — tighter for reverse for better tracking

        # Lane corridor for obstacle detection (forward only)
        self._lane_width = 0.40            # m — robot footprint + small margin
        self._lane_max_lookahead = 2.5     # m — only flag obstacles within this forward distance

        # Overtaking (left side, German convention)
        self._overtake_trigger_time = 1.5        # s — must be in slow mode this long before triggering
        self._overtake_lateral_offset = 0.33      # m — how far left of the recorded path to swerve
        self._overtake_check_forward = 2.0       # m — verify left lane clear at least this far ahead
        self._overtake_phase_duration = 1.5      # s — OUT and IN ramp durations
        self._overtake_pass_distance = 1.0       # m — distance to travel in PASS before merging back
        self._overtake_pass_timeout = 8.0        # s — safety cap on PASS phase

        # Overtake: detour-list approach
        self._overtake_rejoin_distance = 1.5    # m — how far along recorded path the detour rejoins
        #self._overtake_lateral_offset = 0.5     # m — peak lateral offset of the bump
        self._overtake_max_curve = math.radians(23.0)  # max heading change allowed to trigger overtake
        self._overtake_num_points = 20          # detour resolution
        self._overtake_finish_tolerance = 0.22  # m — "reached rejoin point E"

        # Active detour (empty when not overtaking)
        self._overtake_path = []                # list of (x, y) tuples

        # Overtake state
        self._slowdown_timer = 0.0               # accumulated time in slow mode
        self._overtake_phase = None              # 'OUT' | 'PASS' | 'IN'
        self._overtake_start_x = None
        self._overtake_start_y = None
        self._overtake_phase_start_time = None
        self._overtake_pass_start_pos = None     # (x, y) at start of PASS
        self._latest_scan = None                 # stored for feasibility checks
        
        # LIDAR config (LIDAR is backwards - front at ±π)
        self._front_angle_range = math.radians(30)
        # Obstacle avoidance — stop-and-wait (forward only)
        self._obstacle_stop_distance = 0.40    # m — hard stop below this
        self._obstacle_slow_distance = 0.90    # m — start decelerating below this
        self._obstacle_clear_distance = 0.50   # m — resume only when clearer than this (hysteresis)
        self._obstacle_paused = False           # state: currently waiting?
        # Manual pause (keyboard)
        self._manual_paused = False
        self._term_old_settings = None
        # Waypoint pause — stop and wait at specific recorded coordinates
        self._pause_points = [
            # Map 4
            #(2.185, -7.155),
            #(13.3198,0.6645),
            #(14.699, 2.261),

            # Map3
            (-5.6675,-10.9492),
            #(3.900, -3.101),
            (5.241, -1.533),

            # Map 2
            #(-0.190, -15.378),
            #(7.515, -1.680),

            #(-3.116, 0.548)
            #(6.384, -2.526),
            #(-8.102, -7.342),
        ]
        self._pause_tolerance = 0.15      # m — "reached/crossed" this point
        self._pause_duration = 20.0       # s — how long to wait there
        self._pause_triggered = set()     # indices already used, so each point fires once
        self._pause_start_time = None
        self._pause_active_idx = None
        
        # =====================
        
        # Load trajectory
        self._trajectory = Trajectory(trajectory_file)
        self.get_logger().info(f'Loaded {self._trajectory.num_waypoints} waypoints in {self._trajectory.num_segments} segments')
        
        # Controller
        self._controller = PurePursuitController(pp_config)
        
        # State
        self._state = State.WAITING_FOR_LOCALIZATION

        # GO_TO_START tuning (three-phase: ROTATE → DRIVE → ALIGN)
        self._start_pos_tolerance = 0.10          # m — "close enough" to start
        self._rotate_exit_threshold = 0.08        # rad (~5°)  — tight, exits rotate
        self._rotate_reentry_threshold = 0.20     # rad (~23°) — loose, re-enters rotate (hysteresis!)
        self._drive_to_start_speed = 0.12         # m/s — constant forward speed
        self._drive_heading_kp = 0.10             # gentle correction while driving
        
        # Pose
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        # Audio files
        self._sound_obstacle = str(SOUND_DIR / 'obstacle.wav')
        self._pause_sounds = [
            str(SOUND_DIR / 'pickup.wav'),
            str(SOUND_DIR / 'delivery.wav'),
            str(SOUND_DIR / 'delivery.wav'),
        ]
        
        # LIDAR
        self._front_dist = float('inf')
        
        # TF
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        # Subscribers
        self.create_subscription(LaserScan, '/rosbot3/scan', self._scan_cb, 10)
        
        # Publisher
        self._cmd_pub = self.create_publisher(TwistStamped, '/rosbot3/cmd_vel', 10)
        
        # Control loop at 20Hz
        self.create_timer(0.05, self._control_loop)
        
        self.get_logger().info('Trajectory follower ready.')
        # Keyboard listener
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()

    def _scan_cb(self, msg: LaserScan):
        """Cache the scan and update front-lane distance."""
        self._latest_scan = msg
        self._front_dist = self._closest_in_region(
            msg,
            x_min=0.0, x_max=self._lane_max_lookahead,
            y_min=-self._lane_width / 2.0, y_max=self._lane_width / 2.0,
        )

    def _closest_in_region(self, msg: LaserScan, x_min: float, x_max: float,
                        y_min: float, y_max: float) -> float:
        """Find the closest forward distance to any LIDAR point inside a rectangle
        in the robot frame. Returns inf if region is empty."""
        closest = float('inf')
        for i, dist in enumerate(msg.ranges):
            if dist < msg.range_min or dist > msg.range_max:
                continue
            if math.isnan(dist) or math.isinf(dist):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            x = -dist * math.cos(angle)   # +x = forward
            y = -dist * math.sin(angle)   # +y = left
            if x_min < x < x_max and y_min < y < y_max:
                if x < closest:
                    closest = x
        return closest
    
    def _play_sound(self, path: str):
        """Fire-and-forget playback so it doesn't block the control loop."""
        threading.Thread(target=lambda: subprocess.run(['aplay', '-q', path]), daemon=True).start()

    def _check_obstacle(self) -> tuple:
        """
        Returns (speed_scale, should_stop).
        speed_scale: multiply normal speed by this (0..1) for smooth deceleration
        should_stop: True if currently halted waiting for obstacle to clear
        """
        dist = self._front_dist
        
        # Currently waiting — only resume when comfortably clear (hysteresis)
        if self._obstacle_paused:
            if dist > self._obstacle_clear_distance:
                self._obstacle_paused = False
                self.get_logger().info(f'Path clear (dist={dist:.2f}m) — resuming')
                return 1.0, False
            return 0.0, True
        
        # Currently moving — trigger pause if too close
        if dist < self._obstacle_stop_distance:
            self._obstacle_paused = True
            self._play_sound(self._sound_obstacle)
            self.get_logger().info(f'Obstacle at {dist:.2f}m — pausing until clear')
            return 0.0, True
        
        # In the slow-down zone — linear ramp
        if dist < self._obstacle_slow_distance:
            band = self._obstacle_slow_distance - self._obstacle_stop_distance
            scale = (dist - self._obstacle_stop_distance) / band
            return scale, False
    
        # Fully clear
        return 1.0, False
    
    def _is_overtake_feasible(self) -> bool:
        """Is the left lane (overtake corridor) clear of LIDAR points?"""
        if self._latest_scan is None:
            return False
        
        half_lane = self._lane_width / 2.0
        offset = self._overtake_lateral_offset
        
        # Check the overtake lane: shifted left by `offset`, full lane width.
        # Forward extent: from slightly behind the robot to overtake_check_forward.
        closest = self._closest_in_region(
            self._latest_scan,
            x_min=-0.2,
            x_max=self._overtake_check_forward,
            y_min=offset - half_lane,
            y_max=offset + half_lane,
        )
        return closest == float('inf')
    
    def _generate_detour(self, sx: float, sy: float, ex: float, ey: float) -> list:
        """Generate a sine-bump detour from (sx, sy) to (ex, ey), bowing left."""
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 0.01:
            return [(sx, sy), (ex, ey)]
        
        fx, fy = dx / length, dy / length    # unit along straight line
        px, py = -fy, fx                     # perpendicular, left
        
        waypoints = []
        n = self._overtake_num_points
        for i in range(n + 1):
            t = i / n
            cx = sx + t * dx
            cy = sy + t * dy
            off = self._overtake_lateral_offset * math.sin(math.pi * t)
            waypoints.append((cx + off * px, cy + off * py))
        return waypoints


    def _find_rejoin_idx(self, distance: float):
        """Find a waypoint index ~`distance` further along the current segment.
        Returns None if there isn't enough path left."""
        seg = self._trajectory.current_segment
        if not seg:
            return None
        
        closest = self._trajectory.find_closest_waypoint(self._x, self._y)
        cumulative = 0.0
        for i in range(closest, seg.end_idx):
            wp_a = self._trajectory.waypoints[i]
            wp_b = self._trajectory.waypoints[i + 1]
            cumulative += wp_a.distance_to(wp_b.x, wp_b.y)
            if cumulative >= distance:
                return i + 1
        return None   # ran out of segment before reaching target distance


    def _lookahead_on_detour(self, distance: float):
        """Walk the detour from the closest point, return target ~distance ahead."""
        min_d, closest_idx = float('inf'), 0
        for i, (wx, wy) in enumerate(self._overtake_path):
            d = math.hypot(wx - self._x, wy - self._y)
            if d < min_d:
                min_d, closest_idx = d, i
        
        cumulative = 0.0
        for i in range(closest_idx, len(self._overtake_path) - 1):
            wx1, wy1 = self._overtake_path[i]
            wx2, wy2 = self._overtake_path[i + 1]
            cumulative += math.hypot(wx2 - wx1, wy2 - wy1)
            if cumulative >= distance:
                return self._overtake_path[i + 1]
        return self._overtake_path[-1]

    def _update_pose(self) -> bool:
        """Get current pose from TF."""
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return False
        
        self._x = t.transform.translation.x
        self._y = t.transform.translation.y
        
        q = t.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._theta = math.atan2(siny, cosy)
        
        return True

    def _control_loop(self):
        """Main control loop."""
        if not self._update_pose():
            if self._state == State.WAITING_FOR_LOCALIZATION:
                return
            self.get_logger().warn('Lost localization!', throttle_duration_sec=2.0)
            return
        
        if self._state == State.WAITING_FOR_LOCALIZATION:
            self._state = State.ROTATE_TO_START   # ← was State.GO_TO_START
            self.get_logger().info(f'Localized at ({self._x:.2f}, {self._y:.2f}, θ={math.degrees(self._theta):.1f}°)')
            self.get_logger().info(f'Start: ({self._trajectory.start.x:.2f}, {self._trajectory.start.y:.2f})')
            return
        
        # Manual pause gate — halts motion without disturbing state
        if self._manual_paused and self._state in (State.FOLLOWING_SEGMENT, State.OVERTAKING):
            self._stop()
            return
        
        # State machine
        if self._state == State.ROTATE_TO_START:
            self._handle_rotate_to_start()
        elif self._state == State.DRIVE_TO_START:
            self._handle_drive_to_start()
        elif self._state == State.ALIGN_AT_START:
            self._handle_align_at_start()
        elif self._state == State.FOLLOWING_SEGMENT:
            self._handle_following_segment()
        elif self._state == State.WAYPOINT_PAUSE:
            self._handle_waypoint_pause()
        elif self._state == State.ALIGN_AT_END:
            self._handle_align_at_end()
        elif self._state == State.OVERTAKING:
            self._handle_overtaking()
        elif self._state == State.COMPLETE:
            self._handle_complete()

    def _handle_complete(self):
        """Lap finished — reset per-lap state and loop back to the start."""
        self._stop()
        self.get_logger().info('=== LAP COMPLETE — restarting ===')
        self._pause_triggered = set()
        self._pause_active_idx = None
        self._pause_start_time = None
        self._overtake_path = []
        self._slowdown_timer = 0.0
        self._state = State.WAITING_FOR_LOCALIZATION

    def _handle_rotate_to_start(self):
        """Phase 1: rotate in place until facing the start point."""
        start = self._trajectory.start
        dx = start.x - self._x
        dy = start.y - self._y
        dist = math.hypot(dx, dy)
        
        # Already at the start — skip the drive phase entirely
        if dist < self._start_pos_tolerance:
            self._stop()
            self._state = State.ALIGN_AT_START
            self.get_logger().info(f'Already at start (dist={dist:.2f}m), aligning to recorded heading...')
            return
        
        angle_to_start = math.atan2(dy, dx)
        heading_error = self._normalize_angle(angle_to_start - self._theta)
        
        # Exit when well-aligned (tight threshold)
        if abs(heading_error) < self._rotate_exit_threshold:
            self._stop()
            self._state = State.DRIVE_TO_START
            self.get_logger().info(
                f'Aimed at start (err={math.degrees(heading_error):.1f}°, dist={dist:.2f}m), driving...'
            )
            return
        
        angular = self._controller.cfg.kp_angular * heading_error
        angular = max(-self._controller.cfg.max_angular,
                    min(self._controller.cfg.max_angular, angular))
        self._publish_cmd(0.0, angular)
        
        self.get_logger().info(
            f'[ROTATE_TO_START] Dist: {dist:.2f}m | Err: {math.degrees(heading_error):.1f}°',
            throttle_duration_sec=0.5
        )

    def _handle_drive_to_start(self):
        """Phase 2: drive straight to start with mild heading correction."""
        start = self._trajectory.start
        dx = start.x - self._x
        dy = start.y - self._y
        dist = math.hypot(dx, dy)
        
        # Reached start position — go to final alignment
        if dist < self._start_pos_tolerance:
            self._stop()
            self._state = State.ALIGN_AT_START
            self.get_logger().info(f'Reached start (dist={dist:.2f}m), aligning to recorded heading...')
            return
        
        angle_to_start = math.atan2(dy, dx)
        heading_error = self._normalize_angle(angle_to_start - self._theta)
        
        # Drifted too far off — go back and re-aim (hysteresis)
        if abs(heading_error) > self._rotate_reentry_threshold:
            self._stop()
            self._state = State.ROTATE_TO_START
            self.get_logger().info(
                f'Drifted off heading ({math.degrees(heading_error):.1f}°), re-aiming...'
            )
            return
        
        linear = self._drive_to_start_speed
        angular = self._drive_heading_kp * heading_error
        angular = max(-self._controller.cfg.max_angular,
                    min(self._controller.cfg.max_angular, angular))
        self._publish_cmd(linear, angular)
        
        self.get_logger().info(
            f'[DRIVE_TO_START] Dist: {dist:.2f}m | Err: {math.degrees(heading_error):.1f}° | Spd: {linear:.2f}',
            throttle_duration_sec=0.5
        )

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _handle_align_at_start(self):
        """Align heading at start, with a position-creep safety net."""
        start = self._trajectory.start

        # If SLAM shifted between states and position is still off, creep in slowly
        # before rotating. Avoids rotating in place while displaced from the start.
        dist = math.hypot(start.x - self._x, start.y - self._y)
        if dist > self._start_pos_tolerance:
            angle_to_start = math.atan2(start.y - self._y, start.x - self._x)
            heading_error = self._normalize_angle(angle_to_start - self._theta)
            angular = self._controller.cfg.kp_angular * heading_error * 0.5
            angular = max(-self._controller.cfg.max_angular,
                          min(self._controller.cfg.max_angular, angular))
            self._publish_cmd(0.06, angular)   # very slow creep
            self.get_logger().info(
                f'[ALIGN_AT_START] Position refine: dist={dist:.3f}m',
                throttle_duration_sec=0.5
            )
            return

        # Position satisfied — align heading
        angular, aligned = self._controller.compute_align_control(self._theta, start.theta)
        if aligned:
            self._stop()
            self._trajectory.reset()
            self._state = State.FOLLOWING_SEGMENT
            seg = self._trajectory.current_segment
            self.get_logger().info(f'Aligned! Starting {seg}')
            return

        self._publish_cmd(0.0, angular)

    def _handle_following_segment(self):
        """Follow current segment toward its goal (the flip point)."""
        seg = self._trajectory.current_segment
        
        if not seg:
            self._stop()
            self._state = State.COMPLETE
            self.get_logger().info('No more segments!')
            return

        # Check pause points — stop and wait if we've reached/crossed a listed coordinate
        for idx, (px, py) in enumerate(self._pause_points):
            if idx in self._pause_triggered:
                continue
            if math.hypot(px - self._x, py - self._y) < self._pause_tolerance:
                self._pause_triggered.add(idx)
                self._pause_active_idx = idx
                self._stop()
                self._play_sound(self._pause_sounds[idx])
                self._state = State.WAYPOINT_PAUSE
                self.get_logger().info(
                    f'Reached pause point {idx} ({px:.2f}, {py:.2f}) — waiting {self._pause_duration:.0f}s'
                )
                return

        # Check if we reached the segment goal
        if self._trajectory.reached_segment_goal(self._x, self._y, self._segment_goal_tolerance):
            self._stop()
            self._trajectory.advance_segment()

            if self._trajectory.is_complete:
                self._state = State.COMPLETE
                self.get_logger().info('All segments complete!')
            else:
                self._state = State.FOLLOWING_SEGMENT   # was State.SEGMENT_TRANSITION — no wait now
                new_seg = self._trajectory.current_segment
                self.get_logger().info(f'Reached goal! Continuing to {new_seg} (no wait)')
            return
        
        # Advance past waypoints we've passed
        
        
        # Re-add the waypoint counter advance (cosmetic — keeps WP index live in logs)
        self._trajectory.advance_waypoint(self._x, self._y, tolerance=0.12)

        # Is this segment reverse?
        is_reverse = seg.is_reverse

        # Distance-based lookahead — picks a waypoint ~lookahead meters ahead on path
        la_dist = self._lookahead_reverse if is_reverse else self._lookahead_forward
        la_idx, la_wp = self._trajectory.find_lookahead_by_distance(
            self._x, self._y, la_dist
        )
        
        # Obstacle check (forward only — reverse skipped per design)
        if not is_reverse:
            obstacle_scale, should_stop = self._check_obstacle()
        else:
            obstacle_scale = 1.0
            should_stop = False

        # Track time in slow mode for overtake trigger.
        # Must happen BEFORE the early-return so the timer accumulates
        # during a full stop, not just in the decel band.
        dt = 0.05
        if not is_reverse and (obstacle_scale < 1.0 or should_stop):
            self._slowdown_timer += dt
        else:
            self._slowdown_timer = 0.0

        if should_stop:
            self._stop()
            return

        # If we've been slowed for long enough AND the left lane is clear, overtake
        if (self._slowdown_timer >= self._overtake_trigger_time
                and not is_reverse
                and self._is_overtake_feasible()):
            
            rejoin_idx = self._find_rejoin_idx(self._overtake_rejoin_distance)
            if rejoin_idx is None:
                # Not enough recorded path ahead — keep doing what we're doing (slow/stop)
                self.get_logger().info(
                    'Overtake desired but not enough path ahead — staying paused',
                    throttle_duration_sec=2.0
                )
                # don't fall through to overtake; just continue current behavior
            else:
                # --- NEW STRAIGHTNESS CHECK ---
                curr_idx = self._trajectory.find_closest_waypoint(self._x, self._y)
                theta_curr = self._trajectory.tangent_at(curr_idx)
                theta_rejoin = self._trajectory.tangent_at(rejoin_idx)
                
                # How much does the path curve over the overtake window?
                angle_diff = abs(self._normalize_angle(theta_rejoin - theta_curr))
                
                if angle_diff > self._overtake_max_curve:
                    self.get_logger().info(
                        f'Overtake blocked: Path is too curved ({math.degrees(angle_diff):.1f}° > 15°)',
                        throttle_duration_sec=2.0
                    )
                else:
                    # Path is straight enough, execute overtake
                    rejoin_wp = self._trajectory.waypoints[rejoin_idx]
                    self._overtake_path = self._generate_detour(
                        self._x, self._y, rejoin_wp.x, rejoin_wp.y
                    )
                    self.get_logger().info(
                        f'Overtake: detour generated, {len(self._overtake_path)} pts, '
                        f'rejoin at WP {rejoin_idx}'
                    )
                    self._state = State.OVERTAKING
                    self._slowdown_timer = 0.0
                    return
        
        # Compute control
        linear, angular, distance, heading_error, _ = self._controller.compute_control(
            self._x, self._y, self._theta,
            la_wp.x, la_wp.y,
            is_reverse=is_reverse,
            speed_factor=obstacle_scale
        )
        
        self._publish_cmd(linear, angular)
        
        # Logging
        mode = "REV" if is_reverse else "FWD"
        goal_dist = self._trajectory.distance_to_segment_goal(self._x, self._y)
        self.get_logger().info(
            f'[{mode}] Seg {self._trajectory.current_segment_idx+1}/{self._trajectory.num_segments} | '
            f'WP {self._trajectory.current_wp_idx} → LA:{la_idx} | '
            f'Goal: {goal_dist:.2f}m | Spd: {linear:.2f}',
            throttle_duration_sec=0.5
        )
    
    def _handle_overtaking(self):
        """Drive along the pre-computed detour. Exit when we reach its end."""
        if not self._overtake_path:
            self._state = State.FOLLOWING_SEGMENT
            return
        
        # Don't overtake during reverse segments
        seg = self._trajectory.current_segment
        if not seg or seg.is_reverse:
            self._overtake_path = []
            self._state = State.FOLLOWING_SEGMENT
            return
        
        # End-of-detour check
        end_x, end_y = self._overtake_path[-1]
        dist_to_end = math.hypot(end_x - self._x, end_y - self._y)
        if dist_to_end < self._overtake_finish_tolerance:
            self.get_logger().info(f'Overtake complete (rejoined path)')
            self._overtake_path = []
            self._state = State.FOLLOWING_SEGMENT
            self._slowdown_timer = 0.0
            return

        # Obstacle ahead mid-overtake — hold position, keep the detour intact,
        # resume the SAME path once clear (no re-trigger, no new detour)
        obstacle_scale, should_stop = self._check_obstacle()
        if should_stop:
            self._stop()
            self.get_logger().info(
                '[OVERTAKE] Obstacle ahead — holding position',
                throttle_duration_sec=1.0
            )
            return

        # Drive to lookahead point on the detour
        la_x, la_y = self._lookahead_on_detour(self._lookahead_forward)
        
        linear, angular, _, heading_error, _ = self._controller.compute_control(
            self._x, self._y, self._theta,
            la_x, la_y,
            is_reverse=False,
            speed_factor=obstacle_scale,   # was 1.0 — now ramps down approaching an obstacle too
        )
        self._publish_cmd(linear, angular)
        
        self.get_logger().info(
            f'[OVERTAKE] to_rejoin={dist_to_end:.2f}m '
            f'err={math.degrees(heading_error):.1f}° spd={linear:.2f}',
            throttle_duration_sec=0.5
        )

    def _handle_waypoint_pause(self):
        """Wait at a recorded pause-point coordinate, then resume following."""
        if self._pause_start_time is None:
            self._pause_start_time = time.time()
            self.get_logger().info(
                f'Pause point {self._pause_active_idx} — waiting {self._pause_duration:.0f}s'
            )
            return

        remaining = self._pause_duration - (time.time() - self._pause_start_time)
        if remaining > 0:
            self.get_logger().info(f'Resuming in {remaining:.0f}s …', throttle_duration_sec=1.0)
            return

        self._pause_start_time = None
        self._pause_active_idx = None
        self._state = State.FOLLOWING_SEGMENT
        self.get_logger().info('Resuming trajectory following')

    def _handle_align_at_end(self):
        """Align heading at end."""
        end = self._trajectory.end
        
        angular, aligned = self._controller.compute_align_control(self._theta, end.theta)
        
        if aligned:
            self._stop()
            self._state = State.COMPLETE
            self.get_logger().info('=== TRAJECTORY COMPLETE ===')
            return
        
        self._publish_cmd(0.0, angular)
    
    def _keyboard_listener(self):
        """Background thread: S = stop, R = resume."""
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            self._term_old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print('\n[KB] S = stop  |  R = resume\n', flush=True)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == 's':
                        self._manual_paused = True
                        self.get_logger().info('[MANUAL] STOPPED — press R to resume')
                    elif ch == 'r':
                        self._manual_paused = False
                        self.get_logger().info('[MANUAL] RESUMED')
                    elif ch in ('q', '\x03'):   # q or Ctrl+C
                        break
        except Exception:
            pass   # stdin not a tty (piped/launched remotely) — silently skip

    def _restore_terminal(self):
        """Restore terminal settings on shutdown."""
        if self._term_old_settings is not None:
            try:
                import termios
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._term_old_settings
                )
            except Exception:
                pass

    def _publish_cmd(self, linear: float, angular: float):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear.x = linear
        twist.twist.angular.z = angular
        self._cmd_pub.publish(twist)

    def _stop(self):
        self._publish_cmd(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop()
        node.get_logger().info('Stopped.')
    finally:
        node._restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
