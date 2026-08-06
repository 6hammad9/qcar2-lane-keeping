"""Lane-probing and risk gating, after IDEAM (Shu, Zhou & Zhang, T-ITS 2025).

"Agile Decision-Making and Safety-Critical Motion Planning for Emergency
Autonomous Vehicles", IEEE T-ITS 26(9):13750-13766.

What is taken from the paper and what is not
--------------------------------------------
Taken: the three constraint states (LK / LP / LC, §V-A), the lane-probing
behaviour and its exit condition (§V-A2), the relative-speed-dependent risk
gap (Eq. 20), the ellipse barrier and its tangent linearization (Eq. 25-27),
and the progressive-horizon tie-break that stops the decision chattering
(§IV-C2).

Not taken: LSGM's graph search over vehicle groups (§IV). That algorithm
ranks six vehicle groups across three lanes of dense traffic. This vehicle
shares its track with a single ROSbot on a two-lane 38.8 m loop, so C-DFS
would search a graph with one occupied node and return the only answer. It
would be code that cannot be exercised, which is worse than no code.

Everything here is scaled to the QCar. The paper's constants are car-sized
(d_0 = 5 m, d_lat = 2.1 m, l_diag = 3.7 m, 18 m/s) against a 0.39 m vehicle
doing 0.6 m/s in a 0.43 m lane, so they are parameters with QCar defaults
rather than the published values.

Pure functions, no ROS, so the behaviour is testable off the vehicle.
"""

import math


# Constraint states, §V-A.
LANE_KEEPING = "LK"
LANE_PROBING = "LP"
LANE_CHANGING = "LC"


def required_risk_gap(
    *,
    ego_lead_speed,
    follower_speed,
    diagonal_length_m,
    epsilon_m,
    closing_gain,
):
    """Longitudinal gap needed to pass safely, Eq. (20).

    The paper's asymmetry is the point: if the follower in the target lane is
    closing on us it is not enough to be two body-lengths clear, because that
    margin is being eaten. The requirement then grows with the closing speed.

        d_risk = 2*l_diag + eps                    if v_lead,ego > v_follow
        d_risk = 2*l_diag + n*dv + eps             if v_follow > v_lead,ego

    Returns metres. Non-finite input returns +inf, which fails every caller
    closed.
    """
    values = (
        ego_lead_speed,
        follower_speed,
        diagonal_length_m,
        epsilon_m,
        closing_gain,
    )
    try:
        values = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return math.inf
    if not all(math.isfinite(v) for v in values):
        return math.inf

    ego_lead_speed, follower_speed, diagonal, epsilon, gain = values
    if diagonal < 0.0 or epsilon < 0.0 or gain < 0.0:
        return math.inf

    base = 2.0 * diagonal + epsilon
    if follower_speed > ego_lead_speed:
        return base + gain * (follower_speed - ego_lead_speed)
    return base


def spatial_advantage_gained(
    *,
    ego_station_m,
    target_follower_station_m,
    vehicle_length_m,
):
    """The LP -> LC exit condition, s_e - s_f^d >= l/2 (§V-A2).

    Being merely level with the target lane's follower is not enough to merge:
    the bodies still overlap longitudinally while the lateral offset
    collapses. The paper requires half a vehicle length of lead.
    """
    try:
        ego = float(ego_station_m)
        follower = float(target_follower_station_m)
        length = float(vehicle_length_m)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (ego, follower, length)):
        return False
    if length <= 0.0:
        return False

    return (ego - follower) >= (length / 2.0)


def constraint_state(
    *,
    desired_lane_differs,
    advantage_gained,
    lane_change_permitted,
):
    """Select LK / LP / LC, §V-A.

    LK  the desired group is the current group; stay put.
    LP  we want the other lane but have not yet earned the spatial advantage,
        so probe forward instead of stopping.
    LC  advantage earned and the manoeuvre is permitted; commit.

    ``lane_change_permitted`` is the existing curvature gate. Note it does NOT
    suppress probing: a curve forbids changing lanes, not creeping up on the
    obstacle in your own lane, and forbidding both is what left the car
    stationary in WAIT_FOR_CLEAR with a clear passing lane beside it.
    """
    if not desired_lane_differs:
        return LANE_KEEPING
    if not advantage_gained:
        return LANE_PROBING
    if not lane_change_permitted:
        return LANE_PROBING
    return LANE_CHANGING


def ellipse_closest_point(
    *,
    station_m,
    lateral_m,
    center_station_m,
    center_lateral_m,
    semi_major_m,
    semi_minor_m,
    iterations=24,
):
    """Nearest point on an ellipse to (station, lateral), Eq. (25)-(26).

    The paper solves this with Lagrange multipliers. Newton's method on the
    same stationarity condition converges in a handful of steps and needs no
    linear algebra, which suits a controller running at 12 Hz on a Jetson.

    Obstacles are ellipses in Frenet coordinates rather than circles because
    a vehicle is much longer than it is wide; a circumscribing circle around
    a ROSbot would forbid passing at any realistic lane offset.
    """
    a = float(semi_major_m)
    b = float(semi_minor_m)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("ellipse semi-axes must be positive")

    ds = float(station_m) - float(center_station_m)
    dl = float(lateral_m) - float(center_lateral_m)

    if ds == 0.0 and dl == 0.0:
        # Dead centre: no unique nearest point, take the semi-minor axis.
        return float(center_station_m), float(center_lateral_m) + b

    # Solve for the multiplier t. On the valid domain t > -min(a^2, b^2) the
    # residual below falls monotonically from +inf to -1, so bisection always
    # converges. Newton does not: for a point inside the ellipse the step can
    # drive t past the pole at -min(a^2, b^2), after which the cubed
    # denominator overflows. An obstacle barrier must not raise, and points
    # inside the ellipse are exactly the case that matters most.
    def residual(value):
        return (
            ((a * ds) / (value + a * a)) ** 2
            + ((b * dl) / (value + b * b)) ** 2
            - 1.0
        )

    floor = -min(a * a, b * b)
    lo = floor + 1e-12 * max(1.0, a * a, b * b)
    hi = max(a, b) * math.hypot(ds, dl) + max(a * a, b * b)

    # Guarantee the bracket really does straddle the root.
    for _ in range(64):
        if residual(hi) < 0.0:
            break
        hi *= 2.0

    t = hi
    for _ in range(int(iterations) * 3):
        t = 0.5 * (lo + hi)
        if residual(t) > 0.0:
            lo = t
        else:
            hi = t
        if hi - lo < 1e-14:
            break

    s = (a * a * ds) / (t + a * a) + float(center_station_m)
    lat = (b * b * dl) / (t + b * b) + float(center_lateral_m)
    return s, lat


def ellipse_barrier_coefficients(
    *,
    station_m,
    lateral_m,
    center_station_m,
    center_lateral_m,
    semi_major_m,
    semi_minor_m,
):
    """Tangent coefficients (A, B, C) of the linearized DHOCBF, Eq. (27).

        psi_0(s, e_y) = A*s + B*e_y + C
        A = b^2 (s_bar - s_o)
        B = a^2 (e_bar - e_yo)

    where (s_bar, e_bar) is the point on the ellipse nearest the state the
    barrier is linearized about.

    The coefficients are the useful form, not the scalar value: an MPC needs
    a LINEAR CONSTRAINT to stay convex, so it wants A, B and C to hand to the
    solver. Evaluating the barrier at one point tells the optimizer nothing
    about which direction to move.

    Safe by construction: an ellipse lies entirely on one side of any of its
    tangents, so the half-plane psi_0 >= 0 is contained in the true exterior.
    The linearization can refuse a manoeuvre that was in fact safe; it cannot
    admit one that was not.

    Divide all three by hypot(A, B) for conditioning and the residual reads
    directly in metres of clearance, without moving the boundary.
    """
    a = float(semi_major_m)
    b = float(semi_minor_m)
    s_bar, l_bar = ellipse_closest_point(
        station_m=station_m,
        lateral_m=lateral_m,
        center_station_m=center_station_m,
        center_lateral_m=center_lateral_m,
        semi_major_m=a,
        semi_minor_m=b,
    )

    so = float(center_station_m)
    lo = float(center_lateral_m)

    a_cbf = b * b * (s_bar - so)
    b_cbf = a * a * (l_bar - lo)
    c_cbf = -(a_cbf * s_bar + b_cbf * l_bar)

    return a_cbf, b_cbf, c_cbf


def ellipse_barrier(
    *,
    station_m,
    lateral_m,
    center_station_m,
    center_lateral_m,
    semi_major_m,
    semi_minor_m,
):
    """Linearized DHOCBF value psi_0 at the linearization point itself.

    Positive outside the ellipse, zero on it, negative inside. A convenience
    over ``ellipse_barrier_coefficients``; use those directly when building
    an MPC constraint.
    """
    a_cbf, b_cbf, c_cbf = ellipse_barrier_coefficients(
        station_m=station_m,
        lateral_m=lateral_m,
        center_station_m=center_station_m,
        center_lateral_m=center_lateral_m,
        semi_major_m=semi_major_m,
        semi_minor_m=semi_minor_m,
    )
    return a_cbf * float(station_m) + b_cbf * float(lateral_m) + c_cbf


def probe_speed_limit(
    *,
    gap_to_leader_m,
    stop_distance_m,
    probe_speed_mps,
    time_headway_s,
):
    """Speed cap while probing, from the longitudinal DCBF of Eq. (21).

        h_lon = |s - s_i| - T_d*v_x - d_0  >= 0
        =>  v_x <= (gap - d_0) / T_d

    So probing approaches the obstacle at a speed that keeps a headway-scaled
    barrier non-negative, instead of running at cruise until a fixed trigger
    distance and then stopping dead. Returns 0.0 once inside d_0, which is
    the hard stop.
    """
    try:
        gap = float(gap_to_leader_m)
        stop_distance = float(stop_distance_m)
        probe_speed = float(probe_speed_mps)
        headway = float(time_headway_s)
    except (TypeError, ValueError):
        return 0.0

    values = (gap, stop_distance, probe_speed, headway)
    if not all(math.isfinite(v) for v in values):
        return 0.0
    if headway <= 0.0 or probe_speed < 0.0 or stop_distance < 0.0:
        return 0.0
    if gap <= stop_distance:
        return 0.0

    return max(0.0, min(probe_speed, (gap - stop_distance) / headway))


def stable_choice(
    *,
    candidate_scores,
    threshold,
    previous_choice=None,
):
    """Progressive tie-break, §IV-C2.

    When two options score within ``threshold`` the paper extends the
    prediction horizon rather than picking the nominal best, because
    near-equal scores otherwise make the decision flip frame to frame. With
    one obstacle there is no longer horizon to extend to, so the equivalent
    guarantee here is hysteresis: inside the threshold, keep what we chose
    last time.

    ``candidate_scores`` maps option -> score, higher is better.
    """
    if not candidate_scores:
        return previous_choice

    best = max(candidate_scores, key=lambda k: candidate_scores[k])
    if previous_choice is None or previous_choice not in candidate_scores:
        return best

    margin = candidate_scores[best] - candidate_scores[previous_choice]
    if margin <= float(threshold):
        return previous_choice
    return best
