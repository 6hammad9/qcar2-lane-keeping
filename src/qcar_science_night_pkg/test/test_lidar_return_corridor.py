"""Focused geometry tests for the LiDAR swept return corridor."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.lidar_sector_analyzer import (  # noqa: E402
    LidarSectorAnalyzer,
)


def analyzer():
    return LidarSectorAnalyzer(
        front_offset_deg=0.0,
        min_range=0.01,
        max_range=2.0,
        lane_width=0.43,
        front_x_min=0.03,
        front_x_max=0.90,
        side_x_min=0.10,
        side_x_max=0.70,
        emergency_x_min=0.03,
        emergency_x_max=0.70,
        emergency_half_width=0.12,
        min_front_points=1,
        min_side_points=1,
        min_emergency_points=1,
        return_x_min=0.16,
        return_x_max=0.70,
        return_transition_length=0.54,
        return_outer_margin=0.12,
        return_inner_margin=0.20,
    )


def scan_from_xy(points, count=3600):
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / count
    ranges = np.full(count, np.inf, dtype=float)
    for x, y in points:
        angle = math.atan2(y, x)
        index = int(round((angle - angle_min) / angle_increment)) % count
        ranges[index] = min(ranges[index], math.hypot(x, y))
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
    )


def test_swept_corridor_rejects_outer_boundary_and_close_self_returns():
    detector = analyzer()
    # Both sets lie in the old rectangular right box.  The dense points are
    # the parallel outside boundary; the close points model vehicle/sensor
    # returns below the forward clearance used to certify a lane change.
    boundary = [(x, -0.62) for x in np.linspace(0.22, 0.68, 18)]
    self_returns = [(0.12, -0.28), (0.13, -0.34)]
    scan = scan_from_xy(boundary + self_returns)

    legacy = detector.analyze(scan)
    swept = detector.analyze(scan, return_lateral_shift_m=0.47)

    assert not legacy.right_clear
    assert legacy.right_count > 10
    assert swept.right_clear
    assert swept.right_count == 0


def test_return_footprint_does_not_form_target_lane_wedge_near_vehicle():
    detector = analyzer()
    # A parallel edge can be close to the final target lane but is not in the
    # footprint of the lane-change trajectory at this near-forward station.
    # The previous fixed target-edge lower bound included this point and could
    # restart the left offset after a return had already begun.
    near_parallel_edge = [(0.18, -0.52), (0.20, -0.52)]
    status = detector.analyze(
        scan_from_xy(near_parallel_edge),
        return_lateral_shift_m=0.47,
    )

    assert status.right_clear
    assert status.right_count == 0


def test_real_object_centered_in_target_lane_blocks_return():
    detector = analyzer()
    # A compact object at the far portion of the predicted return, where the
    # reference has reached the target lane centre.
    target_lane_object = [
        (0.54, -0.43),
        (0.58, -0.47),
        (0.62, -0.45),
    ]
    status = detector.analyze(
        scan_from_xy(target_lane_object),
        return_lateral_shift_m=0.47,
    )

    assert not status.right_clear
    assert status.right_count >= 3
    assert 0.0 < status.right_min < 1.0


def test_contracted_corridor_tracks_target_lane_during_return():
    detector = analyzer()
    remaining_shift = 0.15
    outside_boundary = [(x, -0.40) for x in np.linspace(0.22, 0.68, 12)]
    target_object = [(0.35, -remaining_shift), (0.50, -remaining_shift)]

    boundary_only = detector.analyze(
        scan_from_xy(outside_boundary),
        return_lateral_shift_m=remaining_shift,
    )
    with_object = detector.analyze(
        scan_from_xy(outside_boundary + target_object),
        return_lateral_shift_m=remaining_shift,
    )

    assert boundary_only.right_clear
    assert not with_object.right_clear
    assert with_object.right_count >= 2


def test_close_current_lane_object_keeps_emergency_authority():
    detector = analyzer()
    status = detector.analyze(
        scan_from_xy([(0.12, 0.0)]),
        return_lateral_shift_m=0.47,
    )

    assert status.right_clear
    assert status.emergency
    assert status.obstacle_ahead
