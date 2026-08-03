"""Why a person stopped the car in a curve and a ROSbot did not.

The wide front rectangle was discarded outright whenever the route was too
curved to overtake, which on this route is most of its length.  That left the
0.12 m half-width emergency corridor as the only detector.  A person is wide
enough to always clip the centreline; a ROSbot or a bottle standing a little
off centre is not, so it was never seen.

The fix is a corridor that follows the route's own curvature, so a curve no
longer has to choose between seeing the road edge it bends away from and
seeing the obstacle it is driving into.  These tests pin both halves of that:
the edge stays out, the obstacle comes in.
"""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.lidar_sector_analyzer import (  # noqa: E402
    LidarSectorAnalyzer,
)
from qcar_science_night_pkg.overtake_safety import (  # noqa: E402
    limit_status_for_path_context,
)
from qcar_science_night_pkg.overtake_types import ObstacleStatus  # noqa: E402


def analyzer(front_narrow_half_width=0.16):
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
        front_narrow_half_width=front_narrow_half_width,
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


def body_at(x, y, width=0.20, samples=9):
    """A solid object of ``width`` centred on (x, y), broadside to the car."""
    return [
        (x, y - width / 2.0 + width * i / (samples - 1))
        for i in range(samples)
    ]


def arc(curvature, x_values, lateral_offset):
    """Points parallel to the route at a fixed lateral offset."""
    return [
        (x, 0.5 * curvature * x * x + lateral_offset)
        for x in x_values
    ]


# ---- Analyzer geometry ----

def test_object_on_the_bend_is_in_the_corridor_but_not_the_straight_box():
    # kappa = 1.82 is this route's measured peak curvature.  A 0.20 m ROSbot
    # 0.60 m along that bend sits 0.33 m off the straight-ahead axis: clear of
    # both the 0.16 m corridor and the 0.12 m emergency box when they are
    # aimed straight, which is precisely the object the car drove into.
    curvature = 1.82
    x = 0.60
    y = 0.5 * curvature * x * x

    detector = analyzer()
    scan = scan_from_xy(body_at(x, y))

    straight = detector.analyze(scan, path_curvature=0.0)
    curved = detector.analyze(scan, path_curvature=curvature)

    assert straight.front_narrow_count == 0
    assert straight.emergency_count == 0

    assert curved.front_narrow_count > 0
    # Nearest corner of the body, not its centre.
    nearest = math.hypot(x, y - 0.10)
    assert curved.front_narrow_min == pytest.approx(nearest, abs=1e-6)
    assert curved.emergency_count > 0


def test_inside_road_edge_stays_out_of_the_curved_corridor():
    # The reason the wide box had to be abandoned in curves: on a bend the
    # inside edge cuts across the straight-ahead rectangle.  It must not
    # enter the corridor that follows the route.
    curvature = 1.0
    detector = analyzer()

    x_values = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    edge = arc(curvature, x_values, lateral_offset=-0.215)
    scan = scan_from_xy(edge)

    curved = detector.analyze(scan, path_curvature=curvature)
    straight = detector.analyze(scan, path_curvature=0.0)

    assert curved.front_narrow_count == 0
    # The same returns do land in the naive straight box, which is what used
    # to force the blanket suppression.
    assert straight.front_count > 0


def test_corridor_half_width_must_fit_inside_the_lane():
    try:
        analyzer(front_narrow_half_width=0.30)
    except ValueError:
        return
    raise AssertionError("expected a corridor wider than the lane to fail")


# ---- Curve/straight limiting ----

def status(**overrides):
    base = dict(
        obstacle_ahead=False,
        emergency=False,
        left_clear=True,
        right_clear=True,
        front_min=2.0,
        left_min=2.0,
        right_min=2.0,
        front_count=0,
        left_count=0,
        right_count=0,
    )
    base.update(overrides)
    return ObstacleStatus(**base)


def limited(st, allow_overtake, min_narrow=2):
    return limit_status_for_path_context(
        st,
        allow_overtake=allow_overtake,
        front_stop_straight_m=1.50,
        emergency_stop_straight_m=1.20,
        front_stop_curve_m=0.70,
        emergency_stop_curve_m=0.60,
        min_front_narrow_points=min_narrow,
    )


def test_rosbot_in_a_curve_is_now_an_obstacle():
    # The headline regression.  Nothing in the wide box (the route bends away
    # from the wall it sees), but the corridor the car will actually sweep is
    # occupied at 0.55 m.
    out, context, _, _ = limited(
        status(
            obstacle_ahead=False,
            front_min=-1.0,
            front_narrow_min=0.55,
            front_narrow_count=14,
        ),
        allow_overtake=False,
    )
    assert context == "CURVE"
    assert out.obstacle_ahead
    # front_min must report what the decision was taken on, because the
    # hard-stop check downstream reads it.
    assert out.front_min == 0.55
    assert out.front_count == 14


def test_curve_obstacle_beyond_the_curve_limit_is_still_ignored():
    out, _, _, _ = limited(
        status(front_narrow_min=0.85, front_narrow_count=14),
        allow_overtake=False,
    )
    assert not out.obstacle_ahead


def test_single_stray_return_does_not_stop_the_car_in_a_curve():
    out, _, _, _ = limited(
        status(front_narrow_min=0.50, front_narrow_count=1),
        allow_overtake=False,
    )
    assert not out.obstacle_ahead


def test_wall_in_a_curve_with_an_empty_corridor_is_still_not_an_obstacle():
    # The §3.2 seam wall: 0.54 m of wall in the wide box, nothing in the
    # corridor.  This is the behaviour the previous blanket suppression got
    # right and the fix must not lose.
    out, _, _, _ = limited(
        status(
            obstacle_ahead=True,
            front_min=0.54,
            front_count=110,
            front_narrow_count=0,
        ),
        allow_overtake=False,
    )
    assert not out.obstacle_ahead
    assert out.front_min == 0.54


def test_emergency_is_gated_on_the_emergency_box_not_the_front_box():
    # front_min belongs to a wider, longer box.  A wall at 0.30 m in the wide
    # box must not authorise an emergency whose own corridor only reaches
    # 0.90 m -- beyond the 0.60 m curve limit.
    out, _, _, _ = limited(
        status(
            emergency=True,
            front_min=0.30,
            emergency_min=0.90,
            emergency_count=4,
        ),
        allow_overtake=False,
    )
    assert not out.emergency


def test_emergency_within_its_own_corridor_still_stops_the_car():
    out, _, _, _ = limited(
        status(
            emergency=True,
            front_min=2.0,
            emergency_min=0.35,
            emergency_count=4,
        ),
        allow_overtake=False,
    )
    assert out.emergency


def test_a_status_without_corridor_data_behaves_as_before():
    # An older publisher, or any hand-built status, reports no corridor.  The
    # curve then falls back to exactly the previous suppression.
    out, _, _, _ = limited(
        status(obstacle_ahead=True, front_min=0.54, front_count=110),
        allow_overtake=False,
    )
    assert not out.obstacle_ahead
