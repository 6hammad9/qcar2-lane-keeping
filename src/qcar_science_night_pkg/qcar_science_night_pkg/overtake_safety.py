"""Pure safety helpers shared by LiDAR/V2V behavior and unit tests."""

import math


def has_safe_v2v_return_clearance(
    *,
    fresh,
    on_path,
    signed_gap_m,
    required_ahead_m,
):
    """True once QCar is far enough *ahead* to merge back safely.

    ``signed_gap_m`` is positive while the communicated vehicle is ahead of
    QCar and negative after QCar passes it.  Merely testing ``gap < 0`` lets
    the return start when the vehicle centres are level.  The bodies then
    overlap longitudinally as the lateral offset collapses.  Requiring a
    finite negative clearance keeps the passing lane until a real merge gap
    exists; the LiDAR corridor and MPC safety ellipse remain authoritative.
    """
    try:
        gap = float(signed_gap_m)
        clearance = float(required_ahead_m)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(gap) or not math.isfinite(clearance):
        return False
    if clearance <= 0.0:
        return False
    return bool(fresh and on_path and gap <= -clearance)


def should_inject_slow_v2v_lead(
    *,
    fresh,
    on_path,
    lead_speed_mps,
    qcar_free_speed_mps,
    max_lead_speed_mps,
    min_speed_advantage_mps,
    gap_m,
    detect_range_m,
    overtake_allowed,
    lead_blocked=None,
):
    """Whether V2V may add a slower lead to the LiDAR observation.

    This encodes the intended asymmetric split. A fresh, sufficiently slower
    ROSbot on the shared path can initiate QCar's LiDAR-verified passing
    behaviour on a straight. A ROSbot travelling near QCar's free-flow speed
    remains under the continuous MPC following governor because there is no
    useful speed advantage with which to complete a pass.

    The absolute speed ceiling prevents a configured high QCar cruise speed
    from classifying normal traffic as a passing target. The relative-speed
    margin prevents a marginal (and very long) pass. This helper only permits
    an *additive* virtual observation; physical LiDAR still decides whether
    the passing lane is clear and retains emergency/unknown-obstacle authority.

    ``lead_blocked`` is the lead's own obstacle report (True/False/None for a
    sender that cannot tell us). Being slower is not a reason to pass: if the
    lead's lane is clear it is simply cruising and will keep going, and a pass
    buys nothing while costing the whole risk of a lane change. Only a lead
    that positively reports that it is *blocked* justifies one. Unknown is not
    evidence for a lane change; an older broadcaster therefore remains usable
    for following and collision avoidance, but cannot initiate a cooperative
    pass until it is upgraded to report its obstacle state.

    A lead that has announced it wants the passing lane does NOT veto the
    pass. Both vehicles want the same lane and somebody has to go first;
    QCar, as the only vehicle running an optimizer, is the one that decides.
    It claims the lane by holding the lead (``should_hold_communicating_lead``)
    and then passes. Refusing to pass instead would deadlock: the lead is
    stopped at its obstacle and the queue never clears.
    """
    if lead_blocked is not True:
        return False

    values = (
        float(lead_speed_mps),
        float(qcar_free_speed_mps),
        float(max_lead_speed_mps),
        float(min_speed_advantage_mps),
        float(gap_m),
        float(detect_range_m),
    )
    if not all(math.isfinite(value) for value in values):
        return False

    lead_speed_mps = float(lead_speed_mps)
    qcar_free_speed_mps = float(qcar_free_speed_mps)
    max_lead_speed_mps = float(max_lead_speed_mps)
    min_speed_advantage_mps = float(min_speed_advantage_mps)
    gap_m = float(gap_m)
    detect_range_m = float(detect_range_m)

    if (
        lead_speed_mps < 0.0
        or qcar_free_speed_mps <= 0.0
        or max_lead_speed_mps < 0.0
        or min_speed_advantage_mps < 0.0
        or detect_range_m <= 0.0
    ):
        return False

    useful_pass_speed = min(
        max_lead_speed_mps,
        qcar_free_speed_mps - min_speed_advantage_mps,
    )
    return bool(
        fresh
        and on_path
        and overtake_allowed
        and lead_speed_mps <= useful_pass_speed
        and 0.0 <= gap_m < detect_range_m
    )


def limit_status_for_path_context(
    status,
    *,
    allow_overtake,
    front_stop_straight_m,
    emergency_stop_straight_m,
    front_stop_curve_m,
    emergency_stop_curve_m,
    min_front_narrow_points=2,
    front_backstop_m=None,
):
    """Apply curve/straight lookahead limits to a raw LiDAR status.

    ``allow_overtake`` comes from the MPC: False means the route is too
    curved to change lanes there, which is also the signal that the broad
    front rectangle is now pointing at the outside of a bend.

    On a straight the wide front box is meaningful. In a curve it is not: a
    legitimate road edge enters it while the recorded route bends safely
    away. This used to be handled by discarding ``obstacle_ahead`` outright
    in a curve, which left the narrow emergency corridor as the only
    detector on a route that is curved for most of its length. That is why a
    person (wide enough to always clip the centreline) reliably stopped the
    car and a ROSbot or bottle standing a little off centre did not.

    So a curve now consults ``front_narrow_*`` instead: the corridor the
    analyzer swept along the route's own curvature. It excludes the road
    edge the route bends away from, which was the original goal, without
    also excluding whatever is genuinely in the car's path. A status without
    that measurement (an older publisher, or a hand-built one) reports
    ``front_narrow_count == 0`` and therefore behaves exactly as before.

    Emergency authority is untouched by context, but is now gated on the
    emergency box's own range rather than on ``front_min``, which belongs to
    a different and larger box.

    Returns ``(status, context, effective_front_m, effective_emergency_m)``.
    """
    if allow_overtake:
        effective_front = float(front_stop_straight_m)
        effective_emergency = float(emergency_stop_straight_m)
        context = "STRAIGHT"
    else:
        effective_front = float(front_stop_curve_m)
        effective_emergency = float(emergency_stop_curve_m)
        context = "CURVE"

    emergency_range = (
        status.emergency_min
        if status.emergency_min > 0.0
        else status.front_min
    )
    emergency_too_far = emergency_range > effective_emergency

    front_min = status.front_min
    front_count = status.front_count

    # The corridor the analyzer swept along the route's own curvature is the
    # volume the car will actually occupy, so it is the primary detector in
    # both contexts.  It is not required to also appear in the wide box,
    # because on a tight bend the corridor leaves that rectangle entirely.
    narrow_hit = (
        status.front_narrow_count >= int(min_front_narrow_points)
        and 0.0 < status.front_narrow_min <= effective_front
    )

    if context == "STRAIGHT":
        # The wide box used to be the sole authority here, and it is far too
        # wide to be one.  "Straight" is only the MPC's lane-change
        # permission; it does not mean the road has no edges.  Measured on
        # the course: a wall held at front_min=1.30 m while the corridor the
        # car would sweep was clear to 1.64 m, so the car declared an
        # obstacle and stopped for scenery with nothing in its path.
        #
        # The box is therefore demoted to a mid-range backstop for something
        # the corridor's width misses -- a wide or badly off-centre object.
        # ``front_backstop_m`` sets its reach: below the nearest wall the box
        # can hold on this course, above the emergency range so an off-centre
        # person is met with braking room rather than at their feet. Callers
        # that do not supply it keep the emergency range. Beyond the
        # backstop the corridor decides alone.
        backstop = (
            float(front_backstop_m)
            if front_backstop_m is not None
            else effective_emergency
        )
        wide_hit = (
            status.obstacle_ahead
            and 0.0 < status.front_min <= backstop
        )
        obstacle_ahead = narrow_hit or wide_hit
        use_narrow = narrow_hit and not wide_hit
    else:
        obstacle_ahead = narrow_hit
        use_narrow = narrow_hit

    if use_narrow:
        # The hard-stop check and the logs downstream read front_min, so
        # report the measurement the decision was actually made on.
        front_min = status.front_narrow_min
        front_count = status.front_narrow_count

    limited = type(status)(
        obstacle_ahead=obstacle_ahead,
        emergency=(status.emergency and not emergency_too_far),
        left_clear=status.left_clear,
        right_clear=status.right_clear,
        front_min=front_min,
        left_min=status.left_min,
        right_min=status.right_min,
        front_count=front_count,
        left_count=status.left_count,
        right_count=status.right_count,
        front_narrow_min=status.front_narrow_min,
        front_narrow_count=status.front_narrow_count,
        emergency_min=status.emergency_min,
        emergency_count=status.emergency_count,
        flank_clear=status.flank_clear,
        flank_min=status.flank_min,
        flank_count=status.flank_count,
    )
    return limited, context, effective_front, effective_emergency


def should_hold_communicating_lead(
    *,
    fresh,
    on_path,
    signed_gap_m,
    interaction_range_m,
    qcar_is_passing,
    lead_blocked=None,
    lead_detour_intent=False,
):
    """Whether QCar should command the communicating lead to hold.

    The conflict this resolves: a blocked ROSbot waits ~1.5 s and then
    detours LEFT around its obstacle, having checked that lane only from
    0.2 m behind itself. A QCar overtaking from further back is invisible to
    that check, so both vehicles claim the passing lane at once and neither
    can recover. Nothing in QCar's own MPC can prevent it, because the
    ROSbot's swerve is not predictable from its broadcast horizon.

    So QCar, the only vehicle running an optimizer, sequences the two: it
    holds the lead for the duration of its own pass and releases it
    afterwards, at which point the lead performs its detour into a lane QCar
    has already vacated. The lead never optimizes and never negotiates — it
    obeys a bounded lease (see ``pack_command``), which is what keeps the
    architecture asymmetric.

    Fails closed to "do not hold": a hold is an instruction to stop a robot
    that is otherwise driving correctly, so it requires positive evidence
    both that QCar is passing and that the lead has reported a real blockage.
    """
    try:
        gap = float(signed_gap_m)
        interaction_range = float(interaction_range_m)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(gap) or not math.isfinite(interaction_range):
        return False
    if interaction_range <= 0.0:
        return False

    return bool(
        fresh
        and on_path
        and lead_blocked is True
        and qcar_is_passing
        and abs(gap) <= interaction_range
    )


def suppress_matched_nonblocking_v2v_lead(
    physical_status,
    *,
    fresh,
    on_path,
    signed_gap_m,
    detect_range_m,
    lead_blocked,
    geometry_fresh,
    relative_geometry,
    lead_body_radius_m,
    association_tolerance_m,
):
    """Remove only a positively associated, non-blocked lead as a pass trigger.

    A clear ROSbot is still a physical object and consequently appears in the
    QCar's front LiDAR box.  Feeding that return directly to the lateral
    overtake state machine would start a pass even though V2V explicitly says
    the ROSbot's lane is clear.  The MPC already owns longitudinal following,
    so this helper removes that *matched* non-emergency return from the lateral
    decision only.

    The match is deliberately strict.  A closer or otherwise unassociated
    return remains an obstacle, and an emergency return is never suppressed.
    ``lead_blocked=None`` is treated like unconfirmed/clear for pass initiation:
    without a positive blocked report it may be followed, but not overtaken.
    Returns ``(status, suppressed)``.
    """
    if (
        lead_blocked is True
        or not fresh
        or not on_path
        or not geometry_fresh
        or not physical_status.obstacle_ahead
        or physical_status.emergency
        or relative_geometry is None
    ):
        return physical_status, False

    try:
        gap = float(signed_gap_m)
        detect_range = float(detect_range_m)
        longitudinal, lateral, _metric = (
            float(value) for value in relative_geometry
        )
        body_radius = float(lead_body_radius_m)
        tolerance = float(association_tolerance_m)
        measured_surface = float(physical_status.front_min)
    except (TypeError, ValueError):
        return physical_status, False

    values = (
        gap,
        detect_range,
        longitudinal,
        lateral,
        body_radius,
        tolerance,
        measured_surface,
    )
    if not all(math.isfinite(value) for value in values):
        return physical_status, False
    if (
        detect_range <= 0.0
        or body_radius < 0.0
        or tolerance < 0.0
        or measured_surface <= 0.0
        or not (0.0 <= gap < detect_range)
    ):
        return physical_status, False

    predicted_surface = max(
        0.0,
        math.hypot(longitudinal, lateral) - body_radius,
    )
    if abs(measured_surface - predicted_surface) > tolerance:
        return physical_status, False

    filtered = type(physical_status)(
        obstacle_ahead=False,
        emergency=physical_status.emergency,
        left_clear=physical_status.left_clear,
        right_clear=physical_status.right_clear,
        front_min=physical_status.front_min,
        left_min=physical_status.left_min,
        right_min=physical_status.right_min,
        front_count=0,
        left_count=physical_status.left_count,
        right_count=physical_status.right_count,
        front_narrow_min=physical_status.front_narrow_min,
        front_narrow_count=physical_status.front_narrow_count,
        emergency_min=physical_status.emergency_min,
        emergency_count=physical_status.emergency_count,
        flank_clear=physical_status.flank_clear,
        flank_min=physical_status.flank_min,
        flank_count=physical_status.flank_count,
    )
    return filtered, True


def should_inject_stopped_v2v_lead(
    *,
    fresh,
    on_path,
    speed_mps,
    moving_threshold_mps,
    gap_m,
    detect_range_m,
    overtake_allowed,
):
    """Backward-compatible wrapper for the former stopped-only policy."""
    threshold = float(moving_threshold_mps)
    return should_inject_slow_v2v_lead(
        fresh=fresh,
        on_path=on_path,
        lead_speed_mps=speed_mps,
        qcar_free_speed_mps=threshold,
        max_lead_speed_mps=max(0.0, math.nextafter(threshold, -math.inf)),
        min_speed_advantage_mps=0.0,
        gap_m=gap_m,
        detect_range_m=detect_range_m,
        overtake_allowed=overtake_allowed,
    )


def scan_matches_safely_offset_lead(
    physical_status,
    relative_geometry,
    clear_metric_min,
    lead_body_radius_m,
    association_tolerance_m,
):
    """Associate a front-box return with a V2V lead outside its ellipse."""
    longitudinal, lateral, metric = relative_geometry
    if metric < clear_metric_min:
        return False
    if not physical_status.left_clear or physical_status.emergency:
        return False
    if physical_status.obstacle_ahead and physical_status.front_min > 0.0:
        predicted_surface_range = max(
            0.0,
            math.hypot(longitudinal, lateral) - lead_body_radius_m,
        )
        if (
            physical_status.front_min
            < predicted_surface_range - association_tolerance_m
        ):
            return False
    return True
