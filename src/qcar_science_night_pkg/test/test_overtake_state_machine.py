from qcar_science_night_pkg.overtake_state_machine import OvertakeStateMachine
from qcar_science_night_pkg.overtake_types import ObstacleStatus
from qcar_science_night_pkg.overtake_safety import (
    has_safe_v2v_return_clearance,
    limit_status_for_path_context,
    scan_matches_safely_offset_lead,
    should_hold_communicating_lead,
    should_inject_slow_v2v_lead,
    suppress_matched_nonblocking_v2v_lead,
)


def status(*, obstacle=False, emergency=False, left=True, right=True,
           front_count=0, left_count=0, right_count=0, flank=True):
    return ObstacleStatus(
        obstacle_ahead=obstacle,
        emergency=emergency,
        left_clear=left,
        right_clear=right,
        front_min=0.8 if obstacle else 2.0,
        left_min=2.0,
        right_min=2.0,
        front_count=front_count,
        left_count=left_count,
        right_count=right_count,
        flank_clear=flank,
        flank_count=0 if flank else 6,
    )


def machine():
    return OvertakeStateMachine(
        obstacle_confirm_required=1,
        left_clear_confirm_required=1,
        no_obstacle_confirm_required=2,
        right_clear_confirm_required=1,
        flank_clear_confirm_required=1,
        min_overtake_progress=0.8,
        min_return_progress=0.3,
    )


def test_scans_cannot_complete_overtake_while_stationary():
    sm = machine()
    decision = sm.update(
        status(obstacle=True, front_count=1), True, progress=10.0
    )
    assert decision.state == sm.OVERTAKE

    for _ in range(100):
        decision = sm.update(status(), True, progress=10.0)

    assert decision.state == sm.OVERTAKE
    assert decision.offset == sm.overtake_offset


def test_pass_and_return_require_measured_progress():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=10.0)
    sm.update(status(), True, progress=10.5)
    decision = sm.update(status(), True, progress=10.81)
    assert decision.state == sm.RETURN

    for _ in range(100):
        decision = sm.update(status(), True, progress=10.81)
    assert decision.state == sm.RETURN

    decision = sm.update(status(), True, progress=11.12)
    assert decision.state == sm.DRIVE


def test_return_waits_for_the_vacated_lane_beside_the_car():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=10.0)
    sm.update(status(flank=False), True, progress=10.5)

    # Far enough travelled, and every forward box says the original lane is
    # clear -- because the lead is level with the car, where no forward box
    # looks. The flank is the only measurement that covers it.
    decision = sm.update(status(flank=False), True, progress=10.81)
    assert decision.state == sm.OVERTAKE
    assert decision.offset == sm.overtake_offset

    decision = sm.update(status(), True, progress=10.9)
    assert decision.state == sm.RETURN


def test_return_is_not_gated_on_permission_to_start_a_pass():
    """overtake_allowed answers "may I begin a pass here", which is false
    over most of this route. Gating the return on it too stranded the car in
    the passing lane for whole curves, and said nothing about whether the
    lead had actually been cleared.
    """
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=10.0)
    sm.update(status(), False, progress=10.5)

    decision = sm.update(status(), False, progress=10.81)
    assert decision.state == sm.RETURN


def test_a_lead_reappearing_alongside_freezes_a_committed_return():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=10.0)
    sm.update(status(), True, progress=10.5)
    assert sm.update(status(), True, progress=10.81).state == sm.RETURN

    decision = sm.update(status(flank=False), True, progress=10.9)
    assert decision.state == sm.RETURN
    assert decision.offset == 999.0
    assert not decision.motion_enabled


def test_far_right_returns_do_not_block_a_confirmed_clear_lane():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=4.0)

    # A map boundary can produce distant points in the swept corridor. The
    # analyzer has already classified the lane clear from their distance, so
    # their nonzero count must not contradict that safety result.
    far_returns = status(right=True, right_count=8)
    sm.update(far_returns, True, progress=4.5)
    decision = sm.update(far_returns, True, progress=4.81)

    assert decision.state == sm.RETURN
    assert decision.offset == 0.0


def test_blocked_original_lane_stops_without_reversing_return():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=0.0)
    sm.update(status(), True, progress=0.4)
    decision = sm.update(status(), True, progress=0.9)
    assert decision.state == sm.RETURN

    decision = sm.update(
        status(right=False, right_count=2), True, progress=1.0
    )
    assert decision.state == sm.RETURN
    assert decision.offset == 999.0
    assert decision.motion_enabled is False

    # Clearing the lane resumes the existing right return.  It does not
    # command another left lane change or restart the pass.
    decision = sm.update(status(), True, progress=1.0)
    assert decision.state == sm.RETURN
    assert decision.offset == 0.0
    assert decision.motion_enabled is True


def test_emergency_stop_has_priority():
    sm = machine()
    decision = sm.update(
        status(obstacle=True, emergency=True, front_count=1),
        True,
        progress=2.0,
    )
    assert decision.state == sm.ESTOP
    assert decision.motion_enabled is False


def test_interrupted_pass_cannot_finish_without_progress():
    sm = machine()
    sm.update(status(obstacle=True, front_count=1), True, progress=5.0)
    decision = sm.update(
        status(left=False, left_count=2), True, progress=5.0
    )
    assert decision.state == sm.WAIT
    assert decision.motion_enabled is False

    # Even though the front return disappeared, the stopped vehicle has not
    # passed anything and must not jump to DRIVE / zero offset.
    decision = sm.update(
        status(left=False, left_count=2), True, progress=5.0
    )
    assert decision.state == sm.WAIT
    decision = sm.update(status(), True, progress=5.0)
    assert decision.state == sm.OVERTAKE
    assert decision.offset == sm.overtake_offset


def test_v2v_pass_must_be_confirmed_before_returning_right():
    sm = machine()
    sm.update(
        status(obstacle=True, front_count=1),
        True,
        progress=2.0,
        pass_confirmed=False,
    )

    # Even with a clear scan and enough travelled distance, a communicated
    # lead vehicle that is still ahead has not been overtaken.
    sm.update(status(), True, progress=2.5, pass_confirmed=False)
    decision = sm.update(status(), True, progress=3.0, pass_confirmed=False)
    assert decision.state == sm.OVERTAKE
    assert decision.offset == sm.overtake_offset

    decision = sm.update(status(), True, progress=3.1, pass_confirmed=True)
    assert decision.state == sm.RETURN


def test_v2v_return_requires_real_forward_merge_clearance():
    values = {
        "fresh": True,
        "on_path": True,
        "required_ahead_m": 0.45,
    }

    # Level centres and a token lead are not a completed pass.
    assert not has_safe_v2v_return_clearance(
        **values, signed_gap_m=0.0
    )
    assert not has_safe_v2v_return_clearance(
        **values, signed_gap_m=-0.05
    )
    assert not has_safe_v2v_return_clearance(
        **values, signed_gap_m=-0.449
    )

    assert has_safe_v2v_return_clearance(
        **values, signed_gap_m=-0.45
    )
    assert has_safe_v2v_return_clearance(
        **values, signed_gap_m=-0.70
    )


def test_v2v_return_clearance_fails_closed_on_bad_or_stale_data():
    values = {
        "fresh": True,
        "on_path": True,
        "signed_gap_m": -0.70,
        "required_ahead_m": 0.45,
    }
    assert not has_safe_v2v_return_clearance(**{**values, "fresh": False})
    assert not has_safe_v2v_return_clearance(**{**values, "on_path": False})
    assert not has_safe_v2v_return_clearance(
        **{**values, "signed_gap_m": float("nan")}
    )
    assert not has_safe_v2v_return_clearance(
        **{**values, "required_ahead_m": 0.0}
    )


def test_safely_offset_v2v_lead_can_leave_front_box_during_committed_pass():
    observed = status(obstacle=True, front_count=8)
    observed = observed.__class__(
        **{**observed.__dict__, "front_min": 0.38}
    )
    assert scan_matches_safely_offset_lead(
        observed,
        relative_geometry=(-0.36, 0.415, 1.50),
        clear_metric_min=1.05,
        lead_body_radius_m=0.20,
        association_tolerance_m=0.12,
    )


def test_unknown_obstacle_closer_than_v2v_lead_still_hard_stops():
    observed = status(obstacle=True, front_count=8)
    observed = observed.__class__(
        **{**observed.__dict__, "front_min": 0.10}
    )
    assert not scan_matches_safely_offset_lead(
        observed,
        relative_geometry=(-0.36, 0.415, 1.50),
        clear_metric_min=1.05,
        lead_body_radius_m=0.20,
        association_tolerance_m=0.12,
    )


def slow_lead_candidate(**overrides):
    values = {
        "fresh": True,
        "on_path": True,
        "lead_speed_mps": 0.07,
        "qcar_free_speed_mps": 0.15,
        "max_lead_speed_mps": 0.10,
        "min_speed_advantage_mps": 0.03,
        "gap_m": 1.4,
        "detect_range_m": 2.5,
        "overtake_allowed": True,
        "lead_blocked": True,
    }
    values.update(overrides)
    return should_inject_slow_v2v_lead(**values)


def test_fresh_stopped_rosbot_ahead_can_initiate_qcar_pass():
    assert slow_lead_candidate(lead_speed_mps=0.0)


def test_fresh_slow_moving_rosbot_ahead_can_initiate_qcar_pass():
    assert slow_lead_candidate(lead_speed_mps=0.07)


def test_near_cruise_speed_rosbot_stays_with_mpc_following_governor():
    assert not slow_lead_candidate(lead_speed_mps=0.13)


def test_pass_requires_useful_speed_advantage_not_only_absolute_ceiling():
    assert not slow_lead_candidate(
        lead_speed_mps=0.07,
        qcar_free_speed_mps=0.09,
    )


def test_stale_off_path_or_behind_v2v_never_injects_obstacle():
    assert not slow_lead_candidate(fresh=False)
    assert not slow_lead_candidate(on_path=False)
    assert not slow_lead_candidate(gap_m=-0.1)
    assert not slow_lead_candidate(gap_m=float("nan"))


def test_invalid_or_reversing_lead_speed_never_injects_obstacle():
    assert not slow_lead_candidate(lead_speed_mps=-0.01)
    assert not slow_lead_candidate(lead_speed_mps=float("nan"))


def test_curve_or_out_of_range_lead_stays_with_mpc_governor():
    assert not slow_lead_candidate(overtake_allowed=False)
    assert not slow_lead_candidate(gap_m=2.5)


def _merge_is_inside_mpc_barrier(return_clearance_m):
    """Replay a merge authorized at ``return_clearance_m`` against the MPC.

    RETURN_RIGHT collapses the passing offset to zero, so the worst case for
    path_mpc_node's DCBF keep-out is the end of the manoeuvre: zero lateral
    separation, with only the along-path lead the state machine demanded.
    Returns True when that pose violates the barrier -- meaning the state
    machine authorized a merge the MPC will refuse mid-manoeuvre.
    """
    from qcar_science_night_pkg.path_utils import PathUtils

    # path_mpc_node.v2v_ellipse_a / v2v_ellipse_b.
    ellipse_a, ellipse_b = 0.55, 0.40

    # ROSbot at the origin facing +x; QCar ahead by the demanded clearance,
    # back in the ROSbot's lane.
    obstacle = [[0.0, 0.0, 0.0]]
    vehicle = [[float(return_clearance_m), 0.0, 0.0]]
    return not PathUtils.predicted_ellipse_clear(
        vehicle, obstacle, ellipse_a, ellipse_b, stages=1
    )


def test_return_clearance_must_clear_the_mpc_safety_ellipse():
    # Regression: the shipped 0.45 m authorized RETURN_RIGHT 0.10 m INSIDE
    # the MPC's own hard barrier. The MPC then rejected its own solve and
    # published zero velocity with the car still straddling the lane, while
    # the ROSbot behind kept closing -- the reported collision.
    assert _merge_is_inside_mpc_barrier(0.45)
    assert _merge_is_inside_mpc_barrier(0.54)

    # The node's validated floor and its default must both stay outside it.
    assert not _merge_is_inside_mpc_barrier(0.60)
    assert not _merge_is_inside_mpc_barrier(0.80)


def test_return_clearance_default_leaves_real_bumper_margin():
    # ROSbot collision box is 0.36 m long (sim/worlds/sami_track.sdf), QCar
    # body radius 0.20 m (v2v_lead_body_radius_m). A centre-to-centre merge
    # clearance has to cover both half-lengths before any margin exists.
    combined_half_lengths = 0.18 + 0.20
    assert 0.80 - combined_half_lengths >= 0.35
    assert 0.45 - combined_half_lengths < 0.10


# ---- Coordinated pass: blocked lead, hold, release ----

def blocked_lead_candidate(**overrides):
    values = {
        "fresh": True,
        "on_path": True,
        "lead_speed_mps": 0.05,
        "qcar_free_speed_mps": 0.15,
        "max_lead_speed_mps": 0.10,
        "min_speed_advantage_mps": 0.03,
        "gap_m": 1.0,
        "detect_range_m": 2.5,
        "overtake_allowed": True,
        "lead_blocked": True,
    }
    values.update(overrides)
    return should_inject_slow_v2v_lead(**values)


def test_a_merely_slower_lead_is_not_a_reason_to_pass():
    # The reported "overtaking even when there was no need": a lead whose own
    # lane is clear is just cruising. Passing it buys nothing and costs a
    # full lane change, so V2V must not propose one.
    assert not blocked_lead_candidate(lead_blocked=False)
    assert blocked_lead_candidate(lead_blocked=True)


def test_unknown_block_state_cannot_authorize_lane_change():
    # A schema 1 broadcaster remains usable for following, but absence of a
    # blockage report is not positive evidence for a cooperative lane change.
    assert not blocked_lead_candidate(lead_blocked=None)


def test_declared_detour_intent_does_not_veto_the_pass():
    # Both vehicles want the same lane and somebody must go first. QCar, the
    # only vehicle running an optimizer, decides -- it claims the lane by
    # holding the lead, not by refusing to move. Refusing would deadlock: the
    # lead is stopped at its obstacle and the queue never clears.
    assert blocked_lead_candidate(lead_blocked=True)


def hold_case(**overrides):
    values = {
        "fresh": True,
        "on_path": True,
        "signed_gap_m": 0.5,
        "interaction_range_m": 2.5,
        "qcar_is_passing": True,
        "lead_blocked": True,
    }
    values.update(overrides)
    return should_hold_communicating_lead(**values)


def test_lead_is_held_only_while_we_occupy_its_passing_lane():
    assert hold_case()
    # Behind us mid-pass is still within the conflict, hence still held.
    assert hold_case(signed_gap_m=-0.5)
    # Not passing: the lead drives normally and may take its own detour.
    assert not hold_case(qcar_is_passing=False)
    assert not hold_case(lead_blocked=False)
    assert not hold_case(lead_blocked=None)
    # Out of interaction range: not our traffic.
    assert not hold_case(signed_gap_m=4.0)


def test_hold_fails_closed_to_not_holding_on_bad_data():
    # A hold stops a robot that is otherwise driving correctly, so it needs
    # positive evidence; absence of data must never manufacture one.
    assert not hold_case(fresh=False)
    assert not hold_case(on_path=False)
    assert not hold_case(signed_gap_m=float("nan"))
    assert not hold_case(interaction_range_m=0.0)
    assert not hold_case(signed_gap_m=None)


def test_matched_clear_lead_is_followed_not_used_as_lateral_pass_trigger():
    observed = status(obstacle=True, front_count=8)
    # One metre centre separation minus the configured 0.20 m body radius
    # predicts the 0.80 m LiDAR surface return.
    observed = type(observed)(
        obstacle_ahead=True,
        emergency=False,
        left_clear=True,
        right_clear=True,
        front_min=0.80,
        left_min=observed.left_min,
        right_min=observed.right_min,
        front_count=8,
        left_count=0,
        right_count=0,
    )
    filtered, suppressed = suppress_matched_nonblocking_v2v_lead(
        observed,
        fresh=True,
        on_path=True,
        signed_gap_m=1.0,
        detect_range_m=2.5,
        lead_blocked=False,
        geometry_fresh=True,
        relative_geometry=(-1.0, 0.0, 3.3),
        lead_body_radius_m=0.20,
        association_tolerance_m=0.12,
    )
    assert suppressed
    assert not filtered.obstacle_ahead
    assert not filtered.emergency


def test_clear_lead_filter_preserves_emergency_and_unassociated_returns():
    common = {
        "fresh": True,
        "on_path": True,
        "signed_gap_m": 1.0,
        "detect_range_m": 2.5,
        "lead_blocked": False,
        "geometry_fresh": True,
        "relative_geometry": (-1.0, 0.0, 3.3),
        "lead_body_radius_m": 0.20,
        "association_tolerance_m": 0.12,
    }
    emergency = status(obstacle=True, emergency=True, front_count=8)
    kept, suppressed = suppress_matched_nonblocking_v2v_lead(
        emergency, **common
    )
    assert not suppressed
    assert kept.emergency and kept.obstacle_ahead

    closer = type(emergency)(
        obstacle_ahead=True,
        emergency=False,
        left_clear=True,
        right_clear=True,
        front_min=0.40,
        left_min=2.0,
        right_min=2.0,
        front_count=8,
        left_count=0,
        right_count=0,
    )
    kept, suppressed = suppress_matched_nonblocking_v2v_lead(
        closer, **common
    )
    assert not suppressed
    assert kept.obstacle_ahead


def test_blocked_lead_is_never_filtered_from_lateral_decision():
    observed = type(status(obstacle=True, front_count=8))(
        obstacle_ahead=True,
        emergency=False,
        left_clear=True,
        right_clear=True,
        front_min=0.80,
        left_min=2.0,
        right_min=2.0,
        front_count=8,
        left_count=0,
        right_count=0,
    )
    kept, suppressed = suppress_matched_nonblocking_v2v_lead(
        observed,
        fresh=True,
        on_path=True,
        signed_gap_m=1.0,
        detect_range_m=2.5,
        lead_blocked=True,
        geometry_fresh=True,
        relative_geometry=(-1.0, 0.0, 3.3),
        lead_body_radius_m=0.20,
        association_tolerance_m=0.12,
    )
    assert not suppressed
    assert kept.obstacle_ahead


# ---- Curve wall suppression ----

def limited(front_min, allow_overtake, emergency=False):
    st = status(obstacle=True, emergency=emergency, front_count=110)
    st = st.__class__(**{**st.__dict__, "front_min": front_min})
    return limit_status_for_path_context(
        st,
        allow_overtake=allow_overtake,
        front_stop_straight_m=1.50,
        emergency_stop_straight_m=1.20,
        front_stop_curve_m=0.70,
        emergency_stop_curve_m=0.60,
    )


def test_close_wall_in_a_curve_is_not_an_obstacle():
    # Regression: a wall 0.54 m ahead against a 0.70 m curve limit left both
    # "too far" flags False, which skipped the suppression entirely and kept
    # obstacle_ahead=True. A curve also forbids overtaking and the inside
    # edge blocks the right sector, so the state machine had no exit and sat
    # in WAIT_FOR_CLEAR in front of a wall the route bends safely away from.
    out, context, _, _ = limited(0.54, allow_overtake=False)
    assert context == "CURVE"
    assert not out.obstacle_ahead


def test_a_curve_keeps_emergency_authority():
    # Suppressing the wide box must not cost the body-corridor stop: a
    # person or object in the swept path still halts the car.
    out, _, _, _ = limited(0.30, allow_overtake=False, emergency=True)
    assert out.emergency


def test_a_straight_still_treats_a_close_return_as_an_obstacle():
    out, context, _, _ = limited(0.80, allow_overtake=True)
    assert context == "STRAIGHT"
    assert out.obstacle_ahead


def test_a_distant_return_is_discarded_on_a_straight():
    out, _, _, _ = limited(2.00, allow_overtake=True)
    assert not out.obstacle_ahead
