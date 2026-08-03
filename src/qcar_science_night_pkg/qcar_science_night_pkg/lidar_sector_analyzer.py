import math
import numpy as np

from qcar_science_night_pkg.overtake_types import ObstacleStatus


class LidarSectorAnalyzer:
    def __init__(
        self,
        front_offset_deg=180.0,
        max_range=1.2,
        min_range=0.03,
        lane_width=0.43,
        front_x_min=0.03,
        front_x_max=0.75,
        side_x_min=0.05,
        side_x_max=0.70,
        emergency_x_min=0.05,
        emergency_x_max=0.40,
        emergency_half_width=0.12,
        emergency_center_y=0.0,
        front_narrow_half_width=0.16,
        max_corridor_curvature=2.0,
        overtake_start_distance=0.75,
        emergency_distance=0.40,
        lane_clear_distance=0.75,
        min_front_points=1,
        min_side_points=1,
        min_emergency_points=2,
        return_x_min=0.16,
        return_x_max=0.70,
        return_transition_length=0.54,
        return_outer_margin=0.12,
        return_inner_margin=0.20,
    ):
        self.front_offset = math.radians(front_offset_deg)
        self.max_range = float(max_range)
        self.min_range = float(min_range)

        self.lane_width = float(lane_width)
        self.half_lane_width = self.lane_width / 2.0

        self.front_x_min = float(front_x_min)
        self.front_x_max = float(front_x_max)

        self.side_x_min = float(side_x_min)
        self.side_x_max = float(side_x_max)

        self.emergency_x_min = float(emergency_x_min)
        self.emergency_x_max = float(emergency_x_max)
        self.emergency_half_width = float(emergency_half_width)
        self.emergency_center_y = float(emergency_center_y)

        # Half-width of the curvature-following corridor. Wide enough to
        # contain the car body (chassis_width 0.192 m in the URDF, so 0.096 m
        # each side) with margin for tracking error, and comfortably narrower
        # than the 0.215 m lane half-width so the road edge stays outside it.
        self.front_narrow_half_width = float(front_narrow_half_width)
        if not 0.0 < self.front_narrow_half_width <= self.half_lane_width:
            raise ValueError(
                "front_narrow_half_width must be in (0, lane_width/2]"
            )

        # Guards the parabolic arc approximation below. Beyond this the small
        # angle assumption stops holding and the corridor would swing wildly
        # off the real path.
        self.max_corridor_curvature = abs(float(max_corridor_curvature))

        self.overtake_start_distance = float(overtake_start_distance)
        self.emergency_distance = float(emergency_distance)
        self.lane_clear_distance = float(lane_clear_distance)

        self.min_front_points = int(min_front_points)
        self.min_side_points = int(min_side_points)
        self.min_emergency_points = int(min_emergency_points)

        # A fixed right-hand rectangle also contains the outside road wall
        # whenever the QCar is in the passing lane.  In the current map that
        # wall produced 70--105 returns per scan after the ROSbot was already
        # behind the QCar.  For a return manoeuvre we instead inspect the
        # volume swept from the current lane to the target lane.
        self.return_x_min = float(return_x_min)
        self.return_x_max = float(return_x_max)
        self.return_transition_length = float(return_transition_length)
        self.return_outer_margin = float(return_outer_margin)
        self.return_inner_margin = float(return_inner_margin)
        if not (
            0.0 <= self.return_x_min < self.return_x_max
            and self.return_transition_length > 0.0
            and self.return_outer_margin > 0.0
            and self.return_inner_margin > 0.0
        ):
            raise ValueError("invalid swept-return-corridor geometry")

    def scan_to_xy(self, scan_msg):
        points = []

        for i, r in enumerate(scan_msg.ranges):
            if np.isnan(r) or np.isinf(r):
                continue

            if not (self.min_range < r < self.max_range):
                continue

            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            rel = math.atan2(
                math.sin(angle - self.front_offset),
                math.cos(angle - self.front_offset),
            )

            x = float(r * math.cos(rel))  # forward
            y = float(r * math.sin(rel))  # left positive

            if x <= 0.0:
                continue

            points.append((x, y, float(r)))

        return points

    def corridor_center_y(self, x, curvature):
        """Lateral offset of the route at ``x`` metres ahead.

        Second-order arc approximation ``y = kappa * x^2 / 2`` for a path
        through the origin tangent to +x. Over the <=0.9 m the front boxes
        reach this is well within a centimetre of the true arc, and it costs
        no trigonometry per point.
        """
        kappa = max(
            -self.max_corridor_curvature,
            min(self.max_corridor_curvature, float(curvature)),
        )
        return 0.5 * kappa * x * x

    def corridor_points(
        self,
        points,
        x_min,
        x_max,
        half_width,
        curvature,
        center_y=0.0,
    ):
        """Ranges inside a corridor that bends along the route.

        The straight rectangles in ``box_points`` are only honest on a
        straight. In a curve they simultaneously miss what the car is about
        to drive into and collect the road edge it will safely pass, which is
        why the wide front box had to be disabled in curves altogether. A
        corridor that follows the path does neither.
        """
        selected = []

        for x, y, r in points:
            if not (x_min <= x <= x_max):
                continue

            offset = center_y + self.corridor_center_y(x, curvature)
            if abs(y - offset) <= half_width:
                selected.append(r)

        return np.array(selected, dtype=float)

    def box_points(self, points, x_min, x_max, y_min, y_max):
        selected = []

        for x, y, r in points:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected.append(r)

        return np.array(selected, dtype=float)

    @staticmethod
    def min_distance(points):
        if len(points) == 0:
            return -1.0
        return float(np.min(points))

    @staticmethod
    def count_points(points):
        return int(len(points))

    def swept_return_points(self, points, lateral_shift_m):
        """Ranges inside the planned right-lane-change swept volume.

        ``lateral_shift_m`` is the remaining distance to the target lane and
        is positive for a target to the vehicle's right.  Both boundaries
        follow a smooth lane-change centreline.  This is important: using the
        final target-lane edge as the lower bound creates a large triangular
        wedge and mistakes the map's parallel road edge for a new obstacle
        while the car is still near the passing lane.

        Returns closer than ``return_x_min`` are excluded as vehicle/sensor
        self artefacts.  This exclusion is used only to certify a lane change;
        the independent front/emergency boxes retain close-object authority.
        """
        shift = max(0.0, float(lateral_shift_m))
        target_y = -shift
        selected = []

        for x, y, r in points:
            if not self.return_x_min <= x <= self.return_x_max:
                continue

            phase = float(np.clip(
                (x - self.return_x_min) / self.return_transition_length,
                0.0,
                1.0,
            ))
            # Cubic smoothstep has zero lateral slope at both lane centres.
            blend = phase * phase * (3.0 - 2.0 * phase)
            swept_center_y = target_y * blend

            # Inspect the vehicle footprint at each future longitudinal
            # station, not the union of every intermediate lane position.
            # The latter includes static road boundaries that the planned
            # return trajectory never occupies.
            outer_edge_y = swept_center_y - self.return_outer_margin
            inner_edge_y = swept_center_y + self.return_inner_margin
            if outer_edge_y <= y <= inner_edge_y:
                selected.append(r)

        return np.asarray(selected, dtype=float)

    def analyze(
        self,
        scan_msg,
        return_lateral_shift_m=None,
        path_curvature=0.0,
    ):
        points = self.scan_to_xy(scan_msg)

        half = self.half_lane_width
        lane = self.lane_width

        # Ego/front lane: y = -half to +half
        # front_x_min is reduced to 0.05 so low/close objects are not missed.
        front_points = self.box_points(
            points,
            self.front_x_min,
            self.front_x_max,
            -half,
            half,
        )

        # Left lane: y = +half to +(half + lane)
        left_points = self.box_points(
            points,
            self.side_x_min,
            self.side_x_max,
            half,
            half + lane,
        )

        if return_lateral_shift_m is None:
            # Right lane: y = -(half + lane) to -half.  This legacy box is
            # retained outside an active pass, where it is useful for the
            # initial left/right availability decision.
            right_points = self.box_points(
                points,
                self.side_x_min,
                self.side_x_max,
                -(half + lane),
                -half,
            )
        else:
            right_points = self.swept_return_points(
                points,
                return_lateral_shift_m,
            )

        # The volume the body actually sweeps over the full front lookahead.
        # This is what stays trustworthy in a curve.
        front_narrow_points = self.corridor_points(
            points,
            self.front_x_min,
            self.front_x_max,
            self.front_narrow_half_width,
            path_curvature,
        )

        # Sudden close obstacle in ego lane. Follows the route for the same
        # reason: on a bend, an object in the car's path is not straight
        # ahead of the bumper.
        emergency_points = self.corridor_points(
            points,
            self.emergency_x_min,
            self.emergency_x_max,
            self.emergency_half_width,
            path_curvature,
            center_y=self.emergency_center_y,
        )

        front_min = self.min_distance(front_points)
        left_min = self.min_distance(left_points)
        right_min = self.min_distance(right_points)
        emergency_min = self.min_distance(emergency_points)

        front_narrow_min = self.min_distance(front_narrow_points)

        front_count = self.count_points(front_points)
        left_count = self.count_points(left_points)
        right_count = self.count_points(right_points)
        emergency_count = self.count_points(emergency_points)
        front_narrow_count = self.count_points(front_narrow_points)

        # No wall filtering here.
        # Waypoint/curve logic in lidar_overtake_node should decide where to ignore walls.
        obstacle_ahead = (
            front_count >= self.min_front_points
            and front_min > 0.0
            and front_min <= self.overtake_start_distance
        )

        emergency = (
            emergency_count >= self.min_emergency_points
            and emergency_min > 0.0
            and emergency_min <= self.emergency_distance
        )

        left_clear = (
            left_count < self.min_side_points
            or (left_min > self.lane_clear_distance and left_min > 0)
        )
        right_clear = (
            right_count < self.min_side_points
            or (right_min > self.lane_clear_distance and right_min > 0)
        )

        return ObstacleStatus(
            obstacle_ahead=obstacle_ahead,
            emergency=emergency,
            left_clear=left_clear,
            right_clear=right_clear,
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
            front_count=front_count,
            left_count=left_count,
            right_count=right_count,
            front_narrow_min=front_narrow_min,
            front_narrow_count=front_narrow_count,
            emergency_min=emergency_min,
            emergency_count=emergency_count,
        )
