"""Obstacle tracking for dynamic obstacle handling."""

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ObstacleState(Enum):
    NO_OBSTACLE = 0
    TRACKING = 1
    FOLLOWING = 2
    OVERTAKING = 3
    RETURNING_TO_PATH = 4


@dataclass
class ObstacleTrackerConfig:
    """Configuration for obstacle tracker."""
    # Detection thresholds
    detection_dist: float = 2.0        # Start tracking at this distance
    stop_dist: float = 0.4             # Emergency stop distance
    safe_following_dist: float = 0.6   # Maintain this distance when following
    
    # Speed matching
    min_speed: float = 0.02            # Minimum driving speed
    
    # Overtaking
    overtake_offset: float = 0.5       # Lateral offset for overtaking
    min_straight_for_overtake: float = 2.0  # Need this much straight road
    max_curvature_for_overtake: float = 0.3  # Max curvature to allow overtake
    min_side_clearance: float = 0.6    # Side must be this clear
    
    # Tracking
    history_size: int = 10             # Number of readings to track
    stationary_threshold: float = 0.03  # m/s - below this is stationary


class ObstacleTracker:
    """
    Tracks obstacles and determines appropriate response.
    """
    
    def __init__(self, config: ObstacleTrackerConfig = None):
        self.cfg = config or ObstacleTrackerConfig()
        
        self._history = deque(maxlen=self.cfg.history_size)
        self._state = ObstacleState.NO_OBSTACLE
        
        # Overtake waypoint
        self._overtake_x = 0.0
        self._overtake_y = 0.0
        self._overtake_side = 'left'
        
        # Current measurements
        self._front_dist = float('inf')
        self._left_dist = float('inf')
        self._right_dist = float('inf')
    
    @property
    def state(self) -> ObstacleState:
        return self._state
    
    @property
    def front_distance(self) -> float:
        return self._front_dist
    
    def update(
        self,
        timestamp: float,
        front_dist: float,
        left_dist: float,
        right_dist: float
    ):
        """Update obstacle measurements."""
        self._front_dist = front_dist
        self._left_dist = left_dist
        self._right_dist = right_dist
        
        self._history.append((timestamp, front_dist))
    
    def get_relative_velocity(self) -> float:
        """
        Get relative velocity of obstacle.
        
        Returns:
            Positive = obstacle moving away
            Negative = we're approaching obstacle
            Zero = same speed or stationary
        """
        if len(self._history) < 2:
            return 0.0
        
        t1, d1 = self._history[-2]
        t2, d2 = self._history[-1]
        
        dt = t2 - t1
        if dt < 0.001:
            return 0.0
        
        return (d2 - d1) / dt
    
    def is_stationary(self) -> bool:
        """Check if obstacle is stationary."""
        return abs(self.get_relative_velocity()) < self.cfg.stationary_threshold
    
    def is_approaching(self) -> bool:
        """Check if we're approaching the obstacle."""
        return self.get_relative_velocity() < -self.cfg.stationary_threshold
    
    def compute_adaptive_speed(
        self,
        current_speed: float,
        desired_speed: float
    ) -> float:
        """
        Compute speed to match obstacle or maintain safe distance.
        """
        front_dist = self._front_dist
        rel_vel = self.get_relative_velocity()
        
        # No obstacle nearby - full speed
        if front_dist > self.cfg.detection_dist:
            return desired_speed
        
        # Emergency stop
        if front_dist < self.cfg.stop_dist:
            return 0.0
        
        # Calculate approach rate (positive = getting closer)
        approach_rate = -rel_vel
        
        # If object moving away, we can maintain speed
        if approach_rate < 0:
            return desired_speed
        
        # Calculate target speed to match object
        if approach_rate > 0:
            # Object speed ≈ our_speed - approach_rate
            object_speed = max(0, current_speed - approach_rate)
            target_speed = object_speed
        else:
            target_speed = desired_speed
        
        # Distance-based adjustment
        if front_dist < self.cfg.safe_following_dist:
            # Too close - slow down proportionally
            closeness = front_dist / self.cfg.safe_following_dist
            target_speed *= closeness
        
        # Clamp
        target_speed = max(self.cfg.min_speed, min(target_speed, desired_speed))
        
        return target_speed
    
    def should_overtake(
        self,
        path_curvature: float,
        straight_distance: float,
        desired_speed: float
    ) -> Tuple[bool, str]:
        """
        Decide if overtaking is safe and beneficial.
        
        Returns:
            (should_overtake, reason)
        """
        front_dist = self._front_dist
        rel_vel = self.get_relative_velocity()
        
        # No obstacle - no need to overtake
        if front_dist > self.cfg.detection_dist:
            return False, "No obstacle"
        
        # 1. Is path straight enough?
        if path_curvature > self.cfg.max_curvature_for_overtake:
            return False, "Curve ahead"
        
        # 2. Enough straight road?
        if straight_distance < self.cfg.min_straight_for_overtake:
            return False, "Not enough straight road"
        
        # 3. Side clearance?
        side_clear = max(self._left_dist, self._right_dist)
        if side_clear < self.cfg.min_side_clearance:
            return False, "Side not clear"
        
        # 4. Is object slow enough to warrant overtake?
        object_speed = max(0, desired_speed + rel_vel)
        if object_speed >= desired_speed * 0.8:
            return False, "Object fast enough"
        
        # All checks passed
        if self.is_stationary():
            return True, "Stationary obstacle"
        else:
            return True, "Slow obstacle"
    
    def start_overtake(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        prefer_left: bool = True
    ) -> Tuple[float, float]:
        """
        Start overtaking maneuver.
        
        Returns:
            (waypoint_x, waypoint_y) - lateral waypoint to reach
        """
        # Choose side
        if prefer_left:
            self._overtake_side = 'left'
        else:
            if self._left_dist > self._right_dist:
                self._overtake_side = 'left'
            else:
                self._overtake_side = 'right'
        
        # Compute lateral waypoint
        if self._overtake_side == 'left':
            offset_angle = current_theta + math.pi / 2
        else:
            offset_angle = current_theta - math.pi / 2
        
        self._overtake_x = current_x + self.cfg.overtake_offset * math.cos(offset_angle)
        self._overtake_y = current_y + self.cfg.overtake_offset * math.sin(offset_angle)
        
        self._state = ObstacleState.OVERTAKING
        
        return self._overtake_x, self._overtake_y
    
    def get_overtake_waypoint(self) -> Tuple[float, float]:
        """Get current overtake waypoint."""
        return self._overtake_x, self._overtake_y
    
    def set_state(self, state: ObstacleState):
        """Set obstacle state."""
        self._state = state
    
    def reset(self):
        """Reset tracker."""
        self._history.clear()
        self._state = ObstacleState.NO_OBSTACLE
