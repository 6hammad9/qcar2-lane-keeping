"""Tests for the IDEAM-derived lane-probing and risk gating."""

import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.ideam import (  # noqa: E402
    LANE_CHANGING,
    LANE_KEEPING,
    LANE_PROBING,
    constraint_state,
    ellipse_barrier,
    ellipse_closest_point,
    probe_speed_limit,
    required_risk_gap,
    spatial_advantage_gained,
    stable_choice,
)


# ---- Risk gap, Eq. (20) ----

def gap(ego_lead, follower):
    return required_risk_gap(
        ego_lead_speed=ego_lead,
        follower_speed=follower,
        diagonal_length_m=0.20,
        epsilon_m=0.10,
        closing_gain=3.0,
    )


def test_gap_is_two_body_lengths_when_we_are_faster():
    assert gap(0.6, 0.2) == pytest.approx(2 * 0.20 + 0.10)


def test_gap_grows_with_a_closing_follower():
    # The margin is being eaten, so two body lengths is not enough.
    assert gap(0.2, 0.6) == pytest.approx(2 * 0.20 + 3.0 * 0.4 + 0.10)
    assert gap(0.2, 0.6) > gap(0.6, 0.2)


def test_gap_fails_closed_on_bad_input():
    assert required_risk_gap(
        ego_lead_speed=float("nan"), follower_speed=0.0,
        diagonal_length_m=0.2, epsilon_m=0.1, closing_gain=3.0,
    ) == math.inf
    assert required_risk_gap(
        ego_lead_speed=0.0, follower_speed=0.0,
        diagonal_length_m=-1.0, epsilon_m=0.1, closing_gain=3.0,
    ) == math.inf


# ---- LP -> LC exit condition, §V-A2 ----

def test_level_with_the_follower_is_not_advantage():
    # Bodies still overlap longitudinally as the lateral offset collapses.
    assert not spatial_advantage_gained(
        ego_station_m=5.0, target_follower_station_m=5.0,
        vehicle_length_m=0.39,
    )


def test_half_a_body_length_ahead_is_advantage():
    assert spatial_advantage_gained(
        ego_station_m=5.20, target_follower_station_m=5.0,
        vehicle_length_m=0.39,
    )


# ---- Constraint state selection, §V-A ----

def test_same_lane_is_lane_keeping():
    assert constraint_state(
        desired_lane_differs=False, advantage_gained=True,
        lane_change_permitted=True,
    ) == LANE_KEEPING


def test_wanting_the_lane_without_advantage_probes():
    assert constraint_state(
        desired_lane_differs=True, advantage_gained=False,
        lane_change_permitted=True,
    ) == LANE_PROBING


def test_advantage_and_permission_commits_to_the_change():
    assert constraint_state(
        desired_lane_differs=True, advantage_gained=True,
        lane_change_permitted=True,
    ) == LANE_CHANGING


def test_a_curve_forbids_changing_but_not_probing():
    # The regression that motivated this: the car sat in WAIT_FOR_CLEAR at
    # idx=286 with a clear passing lane, because a curve suppressed the whole
    # manoeuvre rather than only the lateral part of it.
    assert constraint_state(
        desired_lane_differs=True, advantage_gained=True,
        lane_change_permitted=False,
    ) == LANE_PROBING


# ---- Ellipse geometry, Eq. (25)-(27) ----

def test_closest_point_lies_on_the_ellipse():
    a, b = 0.30, 0.15
    s, lat = ellipse_closest_point(
        station_m=1.0, lateral_m=0.5,
        center_station_m=0.0, center_lateral_m=0.0,
        semi_major_m=a, semi_minor_m=b,
    )
    assert (s / a) ** 2 + (lat / b) ** 2 == pytest.approx(1.0, abs=1e-6)


def test_barrier_is_positive_outside_and_negative_inside():
    kw = dict(
        center_station_m=0.0, center_lateral_m=0.0,
        semi_major_m=0.30, semi_minor_m=0.15,
    )
    assert ellipse_barrier(station_m=1.0, lateral_m=0.0, **kw) > 0
    assert ellipse_barrier(station_m=0.05, lateral_m=0.0, **kw) < 0


def test_barrier_permits_a_lateral_pass_a_circle_would_forbid():
    # A ROSbot is far longer than it is wide. A circumscribing circle
    # (radius = semi-major) would call this point unsafe; the ellipse does
    # not, which is what makes passing at a realistic lane offset possible.
    a, b = 0.30, 0.12
    point = dict(station_m=0.0, lateral_m=0.20)
    value = ellipse_barrier(
        center_station_m=0.0, center_lateral_m=0.0,
        semi_major_m=a, semi_minor_m=b, **point,
    )
    assert value > 0
    assert math.hypot(point["station_m"], point["lateral_m"]) < a


# ---- Probing speed, from Eq. (21) ----

def test_probe_speed_decays_as_the_gap_closes():
    # Both 1.00 m and 0.45 m still saturate the 0.30 m/s probe cap
    # ((0.45-0.30)/0.5 = 0.30), so the decay only shows below that.
    kw = dict(stop_distance_m=0.30, probe_speed_mps=0.30, time_headway_s=0.5)
    far = probe_speed_limit(gap_to_leader_m=0.44, **kw)
    near = probe_speed_limit(gap_to_leader_m=0.36, **kw)
    assert far > near > 0.0


def test_probe_speed_is_capped_at_the_configured_probe_speed():
    assert probe_speed_limit(
        gap_to_leader_m=5.0, stop_distance_m=0.30,
        probe_speed_mps=0.30, time_headway_s=0.5,
    ) == pytest.approx(0.30)


def test_probe_stops_at_the_barrier_distance():
    assert probe_speed_limit(
        gap_to_leader_m=0.30, stop_distance_m=0.30,
        probe_speed_mps=0.30, time_headway_s=0.5,
    ) == 0.0


def test_probe_speed_fails_closed():
    assert probe_speed_limit(
        gap_to_leader_m=float("inf"), stop_distance_m=0.30,
        probe_speed_mps=0.30, time_headway_s=0.5,
    ) == 0.0


# ---- Decision stability, §IV-C2 ----

def test_near_equal_scores_keep_the_previous_choice():
    scores = {"pass": 1.02, "follow": 1.00}
    assert stable_choice(
        candidate_scores=scores, threshold=0.10, previous_choice="follow",
    ) == "follow"


def test_a_clear_winner_overrides_hysteresis():
    scores = {"pass": 2.00, "follow": 1.00}
    assert stable_choice(
        candidate_scores=scores, threshold=0.10, previous_choice="follow",
    ) == "pass"


def test_first_decision_takes_the_best():
    scores = {"pass": 1.02, "follow": 1.00}
    assert stable_choice(
        candidate_scores=scores, threshold=0.10, previous_choice=None,
    ) == "pass"
