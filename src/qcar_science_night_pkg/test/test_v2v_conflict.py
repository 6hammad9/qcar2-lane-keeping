"""Conflict prediction: does an interaction actually impend, and what then."""

import math

import numpy as np

from qcar_science_night_pkg.v2v_conflict import (
    CROSSING,
    FOLLOWING,
    GIVE_WAY,
    HEAD_ON,
    NONE,
    OVERTAKEN,
    PROCEED,
    YIELD,
    FOLLOW,
    classify_encounter,
    closest_approach,
    resample_horizon,
    resolve_encounter,
    sample_constant_speed,
)

STEP = 0.2
RADIUS = 0.55


def straight(x0, y0, yaw, speed, horizon=6.0, step=STEP):
    return sample_constant_speed((x0, y0, yaw), speed, horizon, step)


def encounter(**overrides):
    values = {
        "ego_pose": (0.0, 0.0, 0.0),
        "ego_speed": 0.15,
        "other_pose": (1.0, 0.0, 0.0),
        "other_speed": 0.40,
        "signed_gap_m": 1.0,
        "ego_horizon": straight(0.0, 0.0, 0.0, 0.15),
        "other_horizon": straight(1.0, 0.0, 0.0, 0.40),
        "step_s": STEP,
        "conflict_radius": RADIUS,
        "on_path": True,
    }
    values.update(overrides)
    return classify_encounter(**values)


# ---------------------------------------------------------------- the gate

def test_no_predicted_approach_is_not_an_encounter():
    # Two vehicles on the same road, both moving, never getting close. The
    # whole point: proximity in the abstract owes nothing.
    result = encounter(
        other_pose=(8.0, 0.0, 0.0),
        signed_gap_m=8.0,
        other_horizon=straight(8.0, 0.0, 0.0, 0.40),
    )
    assert result.kind == NONE
    assert not result.is_conflict
    assert math.isinf(result.time_to_conflict)


def test_a_faster_leader_pulling_away_is_not_an_encounter():
    # Ahead of us and accelerating away: separation only grows.
    result = encounter(
        other_pose=(1.0, 0.0, 0.0),
        other_speed=0.40,
        ego_speed=0.05,
        signed_gap_m=1.0,
        ego_horizon=straight(0.0, 0.0, 0.0, 0.05),
        other_horizon=straight(1.0, 0.0, 0.0, 0.40),
    )
    assert result.kind == NONE


def test_parallel_lane_traffic_owes_nothing():
    # Physically close and same heading, but not on our path.
    result = encounter(
        other_pose=(0.0, 0.5, 0.0),
        other_speed=0.40,
        signed_gap_m=0.0,
        other_horizon=straight(0.0, 0.5, 0.0, 0.40),
        on_path=False,
    )
    assert result.kind == NONE


# -------------------------------------------------- being caught from behind

def test_faster_vehicle_closing_from_behind_is_detected():
    # The dominant real case: ROSbot at 0.40 catching QCar at 0.15.
    result = encounter(
        other_pose=(-1.0, 0.0, 0.0),
        other_speed=0.40,
        ego_speed=0.15,
        signed_gap_m=-1.0,
        other_horizon=straight(-1.0, 0.0, 0.0, 0.40),
    )
    assert result.kind == OVERTAKEN
    assert result.closing_speed == pytest_approx(0.25)
    assert result.time_to_conflict < 6.0


def test_being_overtaken_denies_our_own_lane_change():
    # Steering into a vehicle that is passing us is the worst available move.
    result = encounter(
        other_pose=(-1.0, 0.0, 0.0),
        other_speed=0.40,
        ego_speed=0.15,
        signed_gap_m=-1.0,
        other_horizon=straight(-1.0, 0.0, 0.0, 0.40),
    )
    response = resolve_encounter(
        result,
        ego_speed=0.15,
        other_speed=0.40,
        free_speed=0.15,
        yield_speed=0.10,
        give_way_speed=0.05,
        lead_pass_speed_max=0.10,
    )
    assert response.action == YIELD
    assert not response.allow_overtake
    # We yield by holding a steady line, not by stopping dead in its path.
    assert 0.0 < response.speed_cap <= 0.15
    # And we do not stop the vehicle that has the right of way here.
    assert not response.hold_other


# ------------------------------------------------------ closing on a lead

def test_closing_on_a_slower_lead_is_a_following_encounter():
    result = encounter(
        other_pose=(0.8, 0.0, 0.0),
        other_speed=0.02,
        ego_speed=0.15,
        signed_gap_m=0.8,
        other_horizon=straight(0.8, 0.0, 0.0, 0.02),
    )
    assert result.kind == FOLLOWING
    assert result.closing_speed == pytest_approx(0.13)


def test_only_a_lead_we_can_out_run_is_worth_passing():
    slow = encounter(
        other_pose=(0.8, 0.0, 0.0), other_speed=0.02, ego_speed=0.15,
        signed_gap_m=0.8, other_horizon=straight(0.8, 0.0, 0.0, 0.02),
    )
    # Close enough that we do run up on it inside the horizon -- otherwise
    # there is no encounter at all and nothing to decide.
    quick = encounter(
        other_pose=(0.62, 0.0, 0.0), other_speed=0.13, ego_speed=0.15,
        signed_gap_m=0.62, other_horizon=straight(0.62, 0.0, 0.0, 0.13),
    )
    assert quick.kind == FOLLOWING
    common = dict(
        ego_speed=0.15, free_speed=0.15, yield_speed=0.10,
        give_way_speed=0.05, lead_pass_speed_max=0.10,
    )
    assert resolve_encounter(slow, other_speed=0.02, **common).allow_overtake
    passing = resolve_encounter(quick, other_speed=0.13, **common)
    assert passing.action == FOLLOW
    assert not passing.allow_overtake


# ------------------------------------------------------------- crossing

def test_paths_that_cross_are_detected_despite_a_large_arc_gap():
    # The self-intersecting loop: arc positions far apart, physical positions
    # about to coincide. The along-path gap sees nothing here.
    result = encounter(
        ego_pose=(0.0, 0.0, 0.0),
        ego_speed=0.15,
        other_pose=(0.9, -0.9, math.pi / 2.0),
        other_speed=0.40,
        signed_gap_m=11.4,
        ego_horizon=straight(0.0, 0.0, 0.0, 0.15),
        other_horizon=straight(0.9, -0.9, math.pi / 2.0, 0.40),
    )
    assert result.kind == CROSSING
    assert result.min_separation <= RADIUS


def test_an_ambiguous_crossing_is_resolved_by_giving_way():
    result = encounter(
        ego_pose=(0.0, 0.0, 0.0),
        ego_speed=0.15,
        other_pose=(0.9, -0.9, math.pi / 2.0),
        other_speed=0.40,
        signed_gap_m=11.4,
        ego_horizon=straight(0.0, 0.0, 0.0, 0.15),
        other_horizon=straight(0.9, -0.9, math.pi / 2.0, 0.40),
    )
    response = resolve_encounter(
        result, ego_speed=0.15, other_speed=0.40, free_speed=0.15,
        yield_speed=0.10, give_way_speed=0.05, lead_pass_speed_max=0.10,
    )
    # Slower vehicle, no clear priority: slow rather than race for the point.
    assert response.action == GIVE_WAY
    assert response.speed_cap == 0.05
    assert not response.allow_overtake


# -------------------------------------------------------------- head-on

def test_opposed_headings_stop_us_and_hold_them():
    result = encounter(
        ego_pose=(0.0, 0.0, 0.0),
        other_pose=(1.5, 0.0, math.pi),
        other_speed=0.40,
        signed_gap_m=1.5,
        ego_horizon=straight(0.0, 0.0, 0.0, 0.15),
        other_horizon=straight(1.5, 0.0, math.pi, 0.40),
    )
    assert result.kind == HEAD_ON
    response = resolve_encounter(
        result, ego_speed=0.15, other_speed=0.40, free_speed=0.15,
        yield_speed=0.10, give_way_speed=0.05, lead_pass_speed_max=0.10,
    )
    assert response.speed_cap == 0.0
    assert response.hold_other
    assert not response.allow_overtake


# ------------------------------------------------------------- mechanics

def test_closest_approach_finds_the_crossing_stage_and_time():
    ego = straight(0.0, 0.0, 0.0, 1.0, horizon=2.0)
    other = straight(1.0, -1.0, math.pi / 2.0, 1.0, horizon=2.0)
    separation, when, stage = closest_approach(ego, other, STEP)
    assert separation < 0.15
    assert 0.9 < when < 1.1
    assert stage > 0


def test_broadcast_horizon_is_extended_by_holding_its_last_pose():
    # A vehicle whose prediction runs out is modelled as stopping there.
    # Under-running the horizon must not silently shorten the check.
    poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
    out = resample_horizon(poses, source_step_s=0.08, horizon_s=2.0, step_s=STEP)
    assert len(out) == 11
    assert out[-1][0] == pytest_approx(0.1)
    assert out[0][0] == pytest_approx(0.0)


def test_degenerate_inputs_report_no_approach_rather_than_crashing():
    assert closest_approach([], [], STEP)[0] == float("inf")
    assert closest_approach([[0.0, 0.0, 0.0]], [], STEP)[0] == float("inf")
    bad = np.array([[float("nan"), 0.0, 0.0]])
    assert closest_approach(bad, bad, STEP)[0] == float("inf")
    assert resample_horizon(np.zeros((0, 3)), 0.08, 2.0, STEP) is None


def test_unknown_along_path_order_never_guesses_who_is_ahead():
    # Without a usable gap we cannot tell follower from leader, and guessing
    # wrong inverts the response. It degrades to the conservative crossing
    # treatment instead.
    result = encounter(
        other_pose=(0.4, 0.0, 0.0),
        signed_gap_m=float("nan"),
        other_horizon=straight(0.4, 0.0, 0.0, 0.40),
    )
    assert result.kind == CROSSING


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - float(value)) <= tol

        def __repr__(self):
            return f"~{value}"

    return _Approx()
