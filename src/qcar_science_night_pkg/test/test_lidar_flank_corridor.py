"""The lane being vacated, measured beside and behind the car.

Every other box in the analyzer starts ahead of the bumper, so all of them
report clear the moment a lead being passed draws level with the scanner --
which is exactly when merging back is a collision.  These tests pin the
corridor that covers that region and the geometry it must not mistake for it.
"""

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

# The pass amplitude the node commands, i.e. how far left of the lane being
# vacated the car sits while alongside the lead.
SHIFT = 0.55


def analyzer(**overrides):
    kwargs = dict(
        front_offset_deg=0.0,
        min_range=0.01,
        max_range=2.0,
        lane_width=0.43,
        front_x_min=0.10,
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
        flank_x_min=-0.35,
        flank_x_max=0.70,
        flank_half_width=0.14,
        flank_body_half_width=0.12,
        min_flank_points=2,
    )
    kwargs.update(overrides)
    return LidarSectorAnalyzer(**kwargs)


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


def rosbot_at(longitudinal):
    """A ROSbot-sized body centred on the lane being vacated."""
    return [
        (longitudinal + dx, -SHIFT + dy)
        for dx in (-0.08, 0.0, 0.08)
        for dy in (-0.10, 0.0, 0.10)
    ]


def test_lead_exactly_abeam_blocks_the_flank():
    """The case that produced contact: level with the scanner, zero range."""
    status = analyzer().analyze(
        scan_from_xy(rosbot_at(0.0)),
        flank_lateral_shift_m=SHIFT,
    )

    assert not status.flank_clear
    assert status.flank_count >= 2


def test_lead_level_with_the_tail_still_blocks_the_flank():
    """0.22 m of QCar trails the scanner; a lead there is still a collision."""
    status = analyzer().analyze(
        scan_from_xy(rosbot_at(-0.20)),
        flank_lateral_shift_m=SHIFT,
    )

    assert not status.flank_clear


def test_every_forward_box_calls_an_abeam_lead_clear():
    """Why the flank corridor has to exist at all.

    The same scan that blocks the flank is invisible to the front, side and
    swept-return boxes, because all three begin ahead of the bumper.
    """
    scan = scan_from_xy(rosbot_at(0.0))
    status = analyzer().analyze(
        scan,
        return_lateral_shift_m=SHIFT,
        flank_lateral_shift_m=SHIFT,
    )

    assert status.right_clear
    assert not status.obstacle_ahead
    assert not status.flank_clear


def test_lead_fully_behind_the_car_clears_the_flank():
    status = analyzer().analyze(
        scan_from_xy(rosbot_at(-0.62)),
        flank_lateral_shift_m=SHIFT,
    )

    assert status.flank_clear


def test_parallel_road_edge_beyond_the_lane_is_not_a_flank_obstacle():
    """The corridor is one lane wide, not everything to the right.

    Getting this wrong strands the car in the passing lane for the rest of
    the lap, which is the failure the old overtake_allowed gate produced.
    """
    lane_edge = [(x, -(SHIFT + 0.215)) for x in np.linspace(-0.3, 0.68, 25)]
    status = analyzer().analyze(
        scan_from_xy(lane_edge),
        flank_lateral_shift_m=SHIFT,
    )

    assert status.flank_clear


def test_corridor_is_not_sampled_once_it_would_overlap_the_car():
    """Near the end of a return there is no gap left to inspect.

    Reporting the chassis as a flank obstacle would deadlock the merge it is
    supposed to protect.
    """
    detector = analyzer()
    own_body = [(0.0, -0.09), (-0.10, -0.09), (0.10, -0.09)]

    assert detector.flank_corridor_points(
        detector.scan_to_xy(scan_from_xy(own_body)), 0.20, 0.0
    ) is None

    status = detector.analyze(
        scan_from_xy(own_body),
        flank_lateral_shift_m=0.20,
    )
    assert status.flank_clear
    assert status.flank_count == 0


def test_flank_is_unmeasured_and_clear_outside_a_pass():
    status = analyzer().analyze(scan_from_xy(rosbot_at(0.0)))

    assert status.flank_clear
    assert status.flank_count == 0
    assert status.flank_min == -1.0


def test_single_stray_return_does_not_block_the_flank():
    status = analyzer().analyze(
        scan_from_xy([(0.05, -SHIFT)]),
        flank_lateral_shift_m=SHIFT,
    )

    assert status.flank_clear


def test_corridor_follows_route_curvature_like_the_front_corridors():
    """In a bend the lane being vacated is not straight back either."""
    kappa = 1.2
    x = 0.60
    # Where that lane actually is at x metres ahead, per y = kappa*x^2/2.
    bent = [(x + dx, -SHIFT + 0.5 * kappa * x * x) for dx in (-0.02, 0.0, 0.02)]
    scan = scan_from_xy(bent)

    straight = analyzer().analyze(scan, flank_lateral_shift_m=SHIFT)
    bending = analyzer().analyze(
        scan,
        flank_lateral_shift_m=SHIFT,
        path_curvature=kappa,
    )

    assert straight.flank_clear
    assert not bending.flank_clear


def test_rearward_returns_survive_the_scan_conversion():
    """They were discarded outright, so no box could ever see behind."""
    detector = analyzer()
    points = detector.scan_to_xy(scan_from_xy([(-0.30, -SHIFT)]))

    assert any(x < 0.0 for x, _y, _r in points)
