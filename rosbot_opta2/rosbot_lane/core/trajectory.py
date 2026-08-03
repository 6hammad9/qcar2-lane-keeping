"""Trajectory loading with multi-segment direction detection."""

import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Waypoint:
    """Single waypoint on trajectory."""
    x: float
    y: float
    theta: float
    
    def distance_to(self, x: float, y: float) -> float:
        dx = self.x - x
        dy = self.y - y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class Segment:
    """A segment of the trajectory with consistent direction."""
    start_idx: int
    end_idx: int
    is_reverse: bool
    
    def __repr__(self):
        mode = "REV" if self.is_reverse else "FWD"
        return f"Segment({mode}: WP {self.start_idx} → {self.end_idx})"


class Trajectory:
    """
    Manages recorded trajectory with multi-segment direction detection.
    
    Detects all direction flips (forward ↔ reverse) and creates segments.
    Each segment has a consistent direction and a goal point (the flip point).
    """
    
    def __init__(self, filepath: str):
        self.waypoints: List[Waypoint] = []
        self.segments: List[Segment] = []
        self.current_segment_idx: int = 0
        self.current_wp_idx: int = 0
        
        self._load(filepath)
        self._detect_segments()
    
    def _load(self, filepath: str):
        """Load trajectory from CSV."""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wp = Waypoint(
                    x=float(row['x']),
                    y=float(row['y']),
                    theta=float(row['theta'])
                )
                self.waypoints.append(wp)
        print(f"[Trajectory] Loaded {len(self.waypoints)} waypoints")
    
    def _detect_segments(self):
        """
        Detect all direction changes and create segments.
        
        Uses dot product of consecutive direction vectors.
        Checks EVERY waypoint to catch close flip points.
        """
        if len(self.waypoints) < 3:
            # Too few points - single forward segment
            self.segments = [Segment(0, len(self.waypoints) - 1, False)]
            return
        
        # Step 1: Compute direction at each point (travel direction from positions)
        directions = []
        for i in range(len(self.waypoints) - 1):
            wp_curr = self.waypoints[i]
            wp_next = self.waypoints[i + 1]
            
            dx = wp_next.x - wp_curr.x
            dy = wp_next.y - wp_curr.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0.001:
                # Normalized direction vector
                directions.append((dx/dist, dy/dist))
            else:
                # Use previous direction if points are too close
                if directions:
                    directions.append(directions[-1])
                else:
                    directions.append((1.0, 0.0))  # Default
        
        # Step 2: Find flip points using dot product
        flip_indices = []
        
        for i in range(len(directions) - 1):
            d1 = directions[i]
            d2 = directions[i + 1]
            
            # Dot product
            dot = d1[0]*d2[0] + d1[1]*d2[1]
            
            # dot < -0.5 means angle > 120° (direction flip)
            if dot < -0.5:
                flip_indices.append(i + 1)
                print(f"[Trajectory] Flip detected at WP {i+1} (dot={dot:.2f})")
        
        # Step 3: Determine initial direction (forward or reverse)
        # Compare first segment's travel direction with recorded heading
        if len(directions) > 0:
            travel_dir = math.atan2(directions[0][1], directions[0][0])
            heading = self.waypoints[0].theta
            diff = abs(self._normalize_angle(travel_dir - heading))
            initial_is_reverse = diff > 2.0  # > ~115°
        else:
            initial_is_reverse = False
        
        # Step 4: Create segments
        self.segments = []
        segment_start = 0
        is_reverse = initial_is_reverse
        
        for flip_idx in flip_indices:
            # Create segment from segment_start to flip_idx
            self.segments.append(Segment(segment_start, flip_idx, is_reverse))
            
            # Next segment starts at flip point, direction flips
            segment_start = flip_idx
            is_reverse = not is_reverse
        
        # Add final segment
        self.segments.append(Segment(segment_start, len(self.waypoints) - 1, is_reverse))
        
        # Print summary
        print(f"[Trajectory] Detected {len(self.segments)} segments:")
        for seg in self.segments:
            print(f"  {seg}")
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    @property
    def num_segments(self) -> int:
        return len(self.segments)
    
    @property
    def start(self) -> Waypoint:
        return self.waypoints[0] if self.waypoints else None
    
    @property
    def end(self) -> Waypoint:
        return self.waypoints[-1] if self.waypoints else None
    
    @property
    def current_segment(self) -> Optional[Segment]:
        if 0 <= self.current_segment_idx < len(self.segments):
            return self.segments[self.current_segment_idx]
        return None
    
    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None
    
    @property
    def segment_goal(self) -> Optional[Waypoint]:
        """Get the goal waypoint for current segment (the flip point)."""
        seg = self.current_segment
        if seg:
            return self.waypoints[seg.end_idx]
        return None
    
    @property
    def segment_goal_idx(self) -> int:
        """Get the index of current segment's goal."""
        seg = self.current_segment
        return seg.end_idx if seg else -1
    
    @property
    def is_reverse(self) -> bool:
        """Is current segment reverse?"""
        seg = self.current_segment
        return seg.is_reverse if seg else False
    
    @property
    def is_complete(self) -> bool:
        return self.current_segment_idx >= len(self.segments)
    
    def reset(self):
        """Reset to start of trajectory."""
        self.current_segment_idx = 0
        self.current_wp_idx = 0
    
    def advance_segment(self):
        """Move to next segment."""
        if self.current_segment_idx < len(self.segments):
            self.current_segment_idx += 1
            if self.current_segment_idx < len(self.segments):
                self.current_wp_idx = self.segments[self.current_segment_idx].start_idx
    
    def find_closest_waypoint(self, x: float, y: float) -> int:
        """Find closest waypoint to robot within current segment."""
        seg = self.current_segment
        if not seg:
            return 0
        
        min_dist = float('inf')
        closest_idx = seg.start_idx
        
        for i in range(seg.start_idx, seg.end_idx + 1):
            dist = self.waypoints[i].distance_to(x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx

    def advance_waypoint(self, x: float, y: float, tolerance: float = 0.12) -> bool:
        """Advance past waypoints within tolerance (within current segment)."""
        seg = self.current_segment
        if not seg:
            return False
        
        advanced = False
        while self.current_wp_idx <= seg.end_idx:
            wp = self.waypoints[self.current_wp_idx]
            if wp.distance_to(x, y) < tolerance:
                self.current_wp_idx += 1
                advanced = True
            else:
                break
        
        return advanced
    
    def reached_segment_goal(self, x: float, y: float, tolerance: float = 0.15) -> bool:
        """Check if robot has reached current segment's goal."""
        goal = self.segment_goal
        if goal:
            return goal.distance_to(x, y) < tolerance
        return False
    
    def distance_to_segment_goal(self, x: float, y: float) -> float:
        """Distance from position to current segment's goal."""
        goal = self.segment_goal
        if goal:
            return goal.distance_to(x, y)
        return 0.0
    
    def find_lookahead_in_segment(self, x: float, y: float, num_points_ahead: int = 10) -> Tuple[int, Waypoint]:
        """Find lookahead waypoint N points ahead of closest."""
        seg = self.current_segment
        if not seg:
            return 0, self.waypoints[0]
    
        closest_idx = self.find_closest_waypoint(x, y)
        lookahead_idx = min(closest_idx + num_points_ahead, seg.end_idx)
    
        return lookahead_idx, self.waypoints[lookahead_idx]

    def find_lookahead_by_distance(self, x: float, y: float, lookahead_distance: float) -> Tuple[int, Waypoint]:
        """
        Find the waypoint that is approximately `lookahead_distance` meters
        ahead along the recorded path, starting from the waypoint closest to (x, y).
        """
        seg = self.current_segment
        if not seg:
            return 0, self.waypoints[0]
        
        closest_idx = self.find_closest_waypoint(x, y)
        
        cumulative = 0.0
        for i in range(closest_idx, seg.end_idx):
            wp_curr = self.waypoints[i]
            wp_next = self.waypoints[i + 1]
            cumulative += wp_curr.distance_to(wp_next.x, wp_next.y)
            if cumulative >= lookahead_distance:
                return i + 1, self.waypoints[i + 1]
        
        # Hit the segment end before reaching the lookahead distance
        return seg.end_idx, self.waypoints[seg.end_idx]

    def get_waypoint(self, idx: int) -> Optional[Waypoint]:
        """Get waypoint by index."""
        if 0 <= idx < len(self.waypoints):
            return self.waypoints[idx]
        return None
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def tangent_at(self, idx: int) -> float:
        """Path tangent angle (radians) at waypoint `idx`."""
        if idx < 0:
            idx = 0
        if idx >= len(self.waypoints) - 1:
            idx = len(self.waypoints) - 2
        if idx < 0:
            return 0.0
        dx = self.waypoints[idx + 1].x - self.waypoints[idx].x
        dy = self.waypoints[idx + 1].y - self.waypoints[idx].y
        return math.atan2(dy, dx)