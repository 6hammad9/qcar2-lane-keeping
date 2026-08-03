"""PD controller for lane keeping."""

import numpy as np
from typing import Tuple


class LaneController:
    """PD controller for steering."""
    
    def __init__(self, kp: float, kd: float, max_angular: float = 1.5):
        self.kp = kp
        self.kd = kd
        self.max_angular = max_angular
        self._prev_error = 0.0
    
    def compute(self, error: float) -> float:
        """
        Compute angular velocity from error.
        
        Returns angular_z (clipped).
        """
        d_error = error - self._prev_error
        angular_z = self.kp * error + self.kd * d_error
        self._prev_error = error
        return float(np.clip(angular_z, -self.max_angular, self.max_angular))
    
    def reset(self):
        """Reset previous error."""
        self._prev_error = 0.0
