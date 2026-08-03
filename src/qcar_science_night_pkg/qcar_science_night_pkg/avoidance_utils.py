import math
import numpy as np


class LidarSectorAnalyzer:

    def __init__(self, front_offset_deg=180.0):
        self.front_offset = math.radians(front_offset_deg)

    def sector_points(self, scan_msg, angle_min_deg, angle_max_deg, max_range=2.5):
        points = []

        amin = math.radians(angle_min_deg)
        amax = math.radians(angle_max_deg)

        for i, r in enumerate(scan_msg.ranges):
            if np.isnan(r) or np.isinf(r):
                continue

            if not (0.05 < r < max_range):
                continue

            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            rel = math.atan2(
                math.sin(angle - self.front_offset),
                math.cos(angle - self.front_offset)
            )

            if amin <= rel <= amax:
                points.append(r)

        return np.array(points)

    @staticmethod
    def is_blocked(points, distance, min_points):
        if len(points) == 0:
            return False

        return int(np.sum(points < distance)) >= min_points

    @staticmethod
    def min_distance(points):
        if len(points) == 0:
            return -1.0

        return float(np.min(points))