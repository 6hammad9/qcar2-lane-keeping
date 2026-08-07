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


# ---- The wide box on a straight ----

def test_wall_on_a_straight_with_a_clear_corridor_is_not_an_obstacle():
    # Measured on the course: the wide box held a wall at 1.30 m while the
    # corridor the car would sweep was clear to 1.64 m (beyond the 1.50 m
    # limit here), and the car stopped for scenery.  "Straight" is only the
    # MPC's lane-change permission; it does not mean the road has no edges,
    # so the wide box cannot be the sole authority there either.
    out, context, _, _ = limited(
        status(
            obstacle_ahead=True,
            front_min=1.30,
            front_count=72,
            front_narrow_min=1.64,
            front_narrow_count=36,
        ),
        allow_overtake=True,
    )
    assert context == "STRAIGHT"
    assert not out.obstacle_ahead


def test_corridor_hit_on_a_straight_is_an_obstacle():
    out, context, _, _ = limited(
        status(
            obstacle_ahead=True,
            front_min=1.30,
            front_count=40,
            front_narrow_min=1.10,
            front_narrow_count=12,
        ),
        allow_overtake=True,
    )
    assert context == "STRAIGHT"
    assert out.obstacle_ahead
    # The decision was taken on the corridor, so report its measurement.
    assert out.front_min == 1.10
    assert out.front_count == 12


def test_wide_off_centre_object_close_in_is_still_caught_on_a_straight():
    # The backstop the wide box is kept for: something the corridor's width
    # misses, inside the emergency range (1.20 m here).
    out, _, _, _ = limited(
        status(
            obstacle_ahead=True,
            front_min=0.95,
            front_count=30,
            front_narrow_count=0,
        ),
        allow_overtake=True,
    )
    assert out.obstacle_ahead
    assert out.front_min == 0.95


def test_wide_box_beyond_the_backstop_range_defers_to_the_corridor():
    out, _, _, _ = limited(
        status(
            obstacle_ahead=True,
            front_min=1.45,
            front_count=30,
            front_narrow_count=0,
        ),
        allow_overtake=True,
    )
    assert not out.obstacle_ahead


# ---- Adaptive passing width (left-only) ----

def test_feasible_offset_is_left_only_by_default():
    """Two lanes, drive right, overtake left.  A left-hand bend too tight to
    pass on the inside must be refused, not silently passed on the right.
    """
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [1.6] * 60          # radius 0.63 m, left-hand
    out = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 1.50,
    )
    assert out == 0.0


def test_feasible_offset_narrows_instead_of_refusing():
    # A bend that cannot take the full 0.68 m but can take a narrower pass.
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [0.75] * 60
    full = 0.75 / (1.0 - 0.75 * 0.68)
    assert full > 1.50          # the full-width lane really is undrivable
    out = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 1.50,
    )
    assert 0.35 <= out < 0.68
    assert abs(0.75 / (1.0 - 0.75 * out)) <= 1.50


def test_feasible_offset_prefers_the_widest_berth_on_a_straight():
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    out = feasible_overtake_offset(
        [0.0] * 60, 0, 40, True, 0.35, 0.68, 0.03, 1.50,
    )
    assert out == 0.68


def test_feasible_offset_rejects_a_cusp():
    # 1 - kappa*d == 0 exactly: the offset path is degenerate, not merely
    # tight, and must never be reported as drivable.
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    out = feasible_overtake_offset(
        [1.0 / 0.5] * 60, 0, 40, True, 0.5, 0.5, 0.03, 1e9,
    )
    assert out == 0.0


# ---- Wide-box backstop reach ----

def test_close_wall_outside_the_swept_corridor_is_not_an_obstacle():
    """Measured live: front=0.70 fc=76 with narrow=-1.00 nc=0.

    The wide box held a wall 0.70 m ahead while the corridor the car would
    actually sweep was empty, and the car parked in front of scenery.  With
    the backstop pulled inside the wall, the corridor decides alone.
    """
    wall = status(
        obstacle_ahead=True,
        front_min=0.70,
        front_count=76,
        front_narrow_min=-1.0,
        front_narrow_count=0,
    )

    out, _context, _a, _b = limit_status_for_path_context(
        wall,
        allow_overtake=True,
        front_stop_straight_m=1.45,
        emergency_stop_straight_m=0.60,
        front_stop_curve_m=0.75,
        emergency_stop_curve_m=0.55,
        min_front_narrow_points=2,
        front_backstop_m=0.65,
    )

    assert not out.obstacle_ahead


def test_the_same_wall_still_stops_the_car_once_it_is_inside_the_backstop():
    # The backstop has not been switched off, only pulled in: a wide object
    # the corridor misses is still caught, with braking room.
    near = status(
        obstacle_ahead=True,
        front_min=0.62,
        front_count=76,
        front_narrow_min=-1.0,
        front_narrow_count=0,
    )

    out, _context, _a, _b = limit_status_for_path_context(
        near,
        allow_overtake=True,
        front_stop_straight_m=1.45,
        emergency_stop_straight_m=0.60,
        front_stop_curve_m=0.75,
        emergency_stop_curve_m=0.55,
        min_front_narrow_points=2,
        front_backstop_m=0.65,
    )

    assert out.obstacle_ahead


def test_the_rosbot_beyond_the_backstop_is_still_detected_by_the_corridor():
    # Pulling the backstop in must not cost detection range: the corridor
    # reaches front_stop_straight_m regardless.
    lead = status(
        obstacle_ahead=True,
        front_min=1.30,
        front_count=40,
        front_narrow_min=1.30,
        front_narrow_count=12,
    )

    out, _context, _a, _b = limit_status_for_path_context(
        lead,
        allow_overtake=True,
        front_stop_straight_m=1.45,
        emergency_stop_straight_m=0.60,
        front_stop_curve_m=0.75,
        emergency_stop_curve_m=0.55,
        min_front_narrow_points=2,
        front_backstop_m=0.65,
    )

    assert out.obstacle_ahead
    assert out.front_min == 1.30


# ---- Percentile window statistic ----

def test_one_outlier_waypoint_no_longer_vetoes_an_open_stretch():
    """The route is straight but for a single spiking sample.

    Scoring the window by its maximum lets that one waypoint refuse the
    whole 2 m maneuver.  The curvature fit spans 0.75 m, so a real corner
    occupies ~15 consecutive samples and still fails p95; a lone spike is
    an artifact of the recording and must not.
    """
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [0.0] * 40
    kappa[17] = 3.0

    strict = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 2.00,
    )
    relaxed = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 2.00, percentile=95.0,
    )

    assert strict == 0.0
    assert relaxed == 0.68


def test_percentile_still_refuses_a_genuinely_tight_corner():
    # Sustained, not a spike: 30 of 40 samples at radius 0.4 m.
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [2.5] * 30 + [0.0] * 10
    out = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 2.00, percentile=95.0,
    )
    assert out == 0.0


def test_percentile_defaults_to_the_previous_worst_waypoint_rule():
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [0.0] * 40
    kappa[9] = 3.0
    assert feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 2.00,
    ) == feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.68, 0.03, 2.00, percentile=100.0,
    )


def test_percentile_never_admits_a_cusp():
    # A fold is geometric degeneracy, not a noisy sample: even at p95, and
    # even when only one waypoint in the window folds, it stays refused.
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    kappa = [0.0] * 40
    kappa[20] = 1.0 / 0.35
    out = feasible_overtake_offset(
        kappa, 0, 40, True, 0.35, 0.35, 0.03, 1e9, percentile=95.0,
    )
    assert out == 0.0


def test_percentile_outside_range_is_rejected():
    from qcar_science_night_pkg.path_utils import feasible_overtake_offset
    for bad in (49.0, 100.5):
        try:
            feasible_overtake_offset(
                [0.0] * 40, 0, 40, True, 0.35, 0.68, 0.03, 2.00,
                percentile=bad,
            )
        except ValueError:
            continue
        raise AssertionError(f"percentile={bad} should have been rejected")
