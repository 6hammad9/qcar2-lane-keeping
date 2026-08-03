"""Focused tests for the simulated ROSbot's asymmetric follower safety."""

import csv
import json
import math

from sim_rosbot import (
    BehaviorReporter,
    FollowerProximityGovernor,
    configured_pause_trigger,
    pack,
    pose_at_arc,
)


def test_qcar_ahead_is_slowed_progressively_before_emergency_stop():
    follower = FollowerProximityGovernor(0.40, 0.90, 0.50)

    assert follower.update(1.00).speed_scale == 1.0

    decision = follower.update(0.65)
    assert decision.state == "SLOW_QCAR_AHEAD"
    assert math.isclose(decision.speed_scale, 0.5)
    assert not decision.hard_stop

    decision = follower.update(0.45)
    assert math.isclose(decision.speed_scale, 0.1)
    assert not decision.hard_stop


def test_emergency_stop_has_clear_distance_hysteresis():
    follower = FollowerProximityGovernor(0.40, 0.90, 0.50)

    assert follower.update(0.39).hard_stop
    assert follower.update(0.45).hard_stop
    assert follower.update(0.50).hard_stop

    decision = follower.update(0.51)
    assert not decision.hard_stop
    assert decision.state == "SLOW_QCAR_AHEAD"


def test_qcar_behind_does_not_slow_rosbot():
    follower = FollowerProximityGovernor(0.40, 0.90, 0.50)
    decision = follower.update(float("inf"))
    assert decision.speed_scale == 1.0
    assert not decision.hard_stop


def test_stale_observation_cannot_release_an_existing_hold():
    follower = FollowerProximityGovernor(0.40, 0.90, 0.50)

    assert follower.update(0.39, observation_fresh=True).hard_stop
    stale = follower.update(float("inf"), observation_fresh=False)
    assert stale.hard_stop
    assert stale.state == "QCAR_STALE_HOLD"
    assert not follower.update(0.70, observation_fresh=True).hard_stop


def test_stale_observation_does_not_create_a_phantom_obstacle():
    follower = FollowerProximityGovernor(0.40, 0.90, 0.50)
    decision = follower.update(float("inf"), observation_fresh=False)
    assert decision.state == "QCAR_UNKNOWN_CRUISE"
    assert decision.speed_scale == 1.0
    assert not decision.hard_stop


def test_rosbot_path_command_has_no_lateral_overtake_offset():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    cumulative = [0.0, 1.0]

    x, y, yaw = pose_at_arc(points, cumulative, 0.5, loop=False)
    assert math.isclose(x, 0.5)
    assert y == 0.0
    assert yaw == 0.0


def test_normal_scenario_ignores_legacy_scripted_pause_arguments():
    cumulative_arc = [0.0, 1.0, 2.0]
    assert configured_pause_trigger(False, 1, None, cumulative_arc) is None
    assert configured_pause_trigger(True, 1, None, cumulative_arc) == 1.0


def test_clear_sim_lane_is_reported_explicitly_not_as_unknown():
    packet = pack(
        "rosbot3",
        1,
        10.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.08,
        [(0.0, 0.0, 0.0)],
        blocked=False,
    )
    message = json.loads(packet)
    assert message["bl"] == 0


def test_behavior_csv_exposes_progressive_speed_and_hard_stop(tmp_path):
    log_path = tmp_path / "rosbot_behavior.csv"
    reporter = BehaviorReporter(node=None, log_path=str(log_path))
    reporter.publish(
        sim_time=12.0,
        state="SLOW_QCAR_AHEAD",
        base_speed=0.07,
        command_speed=0.035,
        speed_scale=0.5,
        hard_stop=False,
        qcar_distance=0.65,
        qcar_fresh=True,
        scheduled_pause=False,
        finished=False,
    )
    reporter.close()

    with log_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["state"] == "SLOW_QCAR_AHEAD"
    assert row["base_speed_mps"] == "0.070"
    assert row["command_speed_mps"] == "0.035"
    assert row["proximity_speed_scale"] == "0.500"
    assert row["hard_stop_requested"] == "0"
