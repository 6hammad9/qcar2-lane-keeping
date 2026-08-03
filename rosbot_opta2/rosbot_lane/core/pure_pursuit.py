"""Pure Pursuit controller with rotate-first hybrid for forward, direct for reverse."""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PurePursuitConfig:
    """Configuration for Pure Pursuit controller."""
    lookahead_distance: float = 0.5
    min_lookahead: float = 0.3
    max_lookahead: float = 1.0
    
    max_speed: float = 0.15
    min_speed: float = 0.03
    curve_speed: float = 0.08
    reverse_speed: float = 0.10  # Slower for reverse
    curve_blend_threshold: float = 0.5
    heading_blend: float = 0.5 
    heading_blend_curve: float = 0.5
    max_angular: float = 1.0
    kp_angular: float = 1.5
    
    # Rotate first threshold (forward only)
    rotate_first_threshold: float = 0.1
    
    goal_tolerance: float = 0.01
    angle_tolerance: float = 0.05


class PurePursuitController:
    """
    Pure Pursuit controller.
    - Forward: rotate-first if heading error > threshold
    - Reverse: always drive (no rotate-first)
    """
    
    def __init__(self, config: PurePursuitConfig = None):
        self.cfg = config or PurePursuitConfig()
    
    def compute_control(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        target_x: float,
        target_y: float,
        target_theta: float = None,
        local_curvature: float = 0.0,
        is_reverse: bool = False,
        speed_factor: float = 1.0
    ) -> Tuple[float, float, float, float, bool]:
        
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < self.cfg.goal_tolerance:
            return 0.0, 0.0, distance, 0.0, True
        
        # --- Single source of truth: "effective" heading for the geometry math ---
        # For reverse, pretend the robot is facing the opposite direction so all
        # downstream math is identical to the forward case.
        if is_reverse:
            effective_theta = self._normalize_angle(current_theta + math.pi)
            base_speed = self.cfg.reverse_speed
        else:
            effective_theta = current_theta
            base_speed = self.cfg.max_speed
        
        # Position error: from "effective" heading to direction-to-target
        angle_to_target = math.atan2(dy, dx)
        position_error = self._normalize_angle(angle_to_target - effective_theta)
        
        # Heading-track error: from "effective" heading to the recorded heading
        # at the lookahead point. Recorded theta is already in the natural travel
        # direction for both forward and reverse legs (because reverse legs were
        # recorded with theta pointing opposite travel — but we've already flipped
        # effective_theta by π, so the comparison is direct).
        if target_theta is not None:
            if is_reverse:
                # Flip recorded heading too so both are in "travel direction" frame
                desired_theta = self._normalize_angle(target_theta + math.pi)
            else:
                desired_theta = target_theta
            heading_track_error = self._normalize_angle(desired_theta - effective_theta)
        else:
            heading_track_error = position_error
        
        # Blend ratio: more heading-tracking in curves
        if local_curvature > self.cfg.curve_blend_threshold:
            blend = self.cfg.heading_blend_curve
        else:
            blend = self.cfg.heading_blend
        
        combined_error = (1.0 - blend) * position_error + blend * heading_track_error
        
        # Speed scaling on position error (geometric pull)
        speed_scale = max(0.5, math.cos(position_error))
        linear_magnitude = base_speed * speed_factor * speed_scale
        linear_magnitude = max(self.cfg.min_speed, min(linear_magnitude, base_speed))
        
        # Flip linear sign for reverse (the only place we care about direction)
        linear = -linear_magnitude if is_reverse else linear_magnitude
        
        angular = self.cfg.kp_angular * combined_error
        angular = max(-self.cfg.max_angular, min(self.cfg.max_angular, angular))
        
        return linear, angular, distance, position_error, False
    
    def compute_align_control(
        self,
        current_theta: float,
        target_theta: float
    ) -> Tuple[float, bool]:
        """Align heading to target."""
        heading_error = self._normalize_angle(target_theta - current_theta)
        
        if abs(heading_error) < self.cfg.angle_tolerance:
            return 0.0, True
        
        angular = self.cfg.kp_angular * heading_error
        angular = max(-self.cfg.max_angular, min(self.cfg.max_angular, angular))
        
        return angular, False
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle