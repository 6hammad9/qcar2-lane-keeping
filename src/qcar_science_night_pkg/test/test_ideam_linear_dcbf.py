"""The tangent linearization behind the MPC's V2V barrier (IDEAM Eq. 25-27).

The property that makes it safe to put a linear constraint in place of the
ellipse: an ellipse lies entirely on one side of any of its tangents, so the
tangent half-plane is contained in the true exterior. The linearization can
refuse a safe manoeuvre; it cannot permit an unsafe one.
"""

import math

from qcar_science_night_pkg.ideam import (
    ellipse_barrier,
    ellipse_barrier_coefficients,
    ellipse_closest_point,
)

A = 0.55   # v2v_ellipse_a
B = 0.40   # v2v_ellipse_b


def coeffs(station, lateral, a=A, b=B):
    return ellipse_barrier_coefficients(
        station_m=station,
        lateral_m=lateral,
        center_station_m=0.0,
        center_lateral_m=0.0,
        semi_major_m=a,
        semi_minor_m=b,
    )


def psi(station, lateral, at_station, at_lateral, a=A, b=B):
    """Barrier linearized at (at_station, at_lateral), evaluated elsewhere."""
    a_lin, b_lin, c_lin = coeffs(at_station, at_lateral, a, b)
    return a_lin * station + b_lin * lateral + c_lin


def test_coefficients_reproduce_the_barrier_value():
    for station, lateral in [(1.2, 0.3), (-0.8, 0.5), (0.1, -1.4)]:
        a_lin, b_lin, c_lin = coeffs(station, lateral)
        expected = ellipse_barrier(
            station_m=station,
            lateral_m=lateral,
            center_station_m=0.0,
            center_lateral_m=0.0,
            semi_major_m=A,
            semi_minor_m=B,
        )
        got = a_lin * station + b_lin * lateral + c_lin
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)


def test_the_tangent_never_admits_a_point_inside_the_ellipse():
    # The safety property. Linearize anywhere outside, then check that no
    # point strictly inside the ellipse satisfies the linear constraint.
    linearization_points = [
        (1.0, 0.0), (0.0, 1.0), (-1.2, 0.4), (0.9, -0.9), (2.0, 0.1),
    ]
    for at_station, at_lateral in linearization_points:
        for i in range(36):
            angle = 2.0 * math.pi * i / 36.0
            # Strictly inside: 90% of each semi-axis.
            inside_s = 0.9 * A * math.cos(angle)
            inside_l = 0.9 * B * math.sin(angle)
            value = psi(inside_s, inside_l, at_station, at_lateral)
            assert value < 0.0, (
                f"linearized at ({at_station}, {at_lateral}), point "
                f"({inside_s:.3f}, {inside_l:.3f}) inside the ellipse "
                f"was admitted with psi={value:.6f}"
            )


def test_the_tangent_is_conservative_relative_to_the_true_exterior():
    # psi >= 0 must imply the exact ellipse metric >= 1. The converse need
    # not hold -- that gap is exactly the conservativeness being bought.
    at_station, at_lateral = 1.1, 0.35
    for i in range(60):
        angle = 2.0 * math.pi * i / 60.0
        for radius in (0.5, 1.0, 1.5, 2.0):
            station = radius * math.cos(angle)
            lateral = radius * math.sin(angle)
            if psi(station, lateral, at_station, at_lateral) >= 0.0:
                metric = (station / A) ** 2 + (lateral / B) ** 2
                assert metric >= 1.0 - 1e-9, (
                    f"({station:.3f}, {lateral:.3f}) passed the linear "
                    f"constraint but sits inside the ellipse, metric={metric}"
                )


def test_the_barrier_is_zero_on_the_ellipse_at_the_tangent_point():
    at_station, at_lateral = 1.3, 0.2
    s_bar, l_bar = ellipse_closest_point(
        station_m=at_station,
        lateral_m=at_lateral,
        center_station_m=0.0,
        center_lateral_m=0.0,
        semi_major_m=A,
        semi_minor_m=B,
    )
    value = psi(s_bar, l_bar, at_station, at_lateral)
    assert abs(value) < 1e-9


def test_normalizing_moves_no_boundary():
    # The MPC divides by hypot(A, B) for conditioning. That must not change
    # which states are admissible, only the scale of the residual.
    at_station, at_lateral = 0.95, -0.6
    a_lin, b_lin, c_lin = coeffs(at_station, at_lateral)
    norm = math.hypot(a_lin, b_lin)
    assert norm > 1e-9

    for i in range(40):
        angle = 2.0 * math.pi * i / 40.0
        station = 1.4 * math.cos(angle)
        lateral = 1.4 * math.sin(angle)
        raw = a_lin * station + b_lin * lateral + c_lin
        scaled = (
            (a_lin / norm) * station
            + (b_lin / norm) * lateral
            + (c_lin / norm)
        )
        assert (raw >= 0.0) == (scaled >= 0.0)
        assert math.isclose(scaled, raw / norm, rel_tol=1e-9, abs_tol=1e-12)


def test_normalized_barrier_reads_as_metres_of_clearance():
    # Straight ahead of the ellipse along the major axis, the normalized
    # barrier should equal the distance to the ellipse surface.
    station = 1.55
    a_lin, b_lin, c_lin = coeffs(station, 0.0)
    norm = math.hypot(a_lin, b_lin)
    value = (a_lin * station + b_lin * 0.0 + c_lin) / norm
    assert math.isclose(value, station - A, rel_tol=1e-6, abs_tol=1e-9)


def test_a_vehicle_on_the_boundary_reads_exactly_zero():
    station = A
    value = psi(station, 0.0, station, 0.0)
    assert abs(value) < 1e-9


def test_degenerate_semi_axes_are_rejected():
    for a, b in [(0.0, B), (A, 0.0), (-1.0, B)]:
        try:
            coeffs(1.0, 1.0, a, b)
        except ValueError:
            continue
        raise AssertionError(f"semi-axes ({a}, {b}) should not be accepted")
