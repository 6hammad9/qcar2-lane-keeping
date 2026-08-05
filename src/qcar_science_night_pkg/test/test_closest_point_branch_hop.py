"""The reference must not hop to a parallel branch of a self-crossing route.

Measured on the physical car: idx jumped 286 -> 685 while the vehicle moved
0.25 m, after which every cycle reported ``position_error=1.49 m`` with a
small yaw error and the tracking safety stop held the car indefinitely.

Cause: when the windowed search found nothing within 0.75 m, the recovery
path searched from ``previous_idx`` to the END of the trajectory. This track's
branches run 0.02-0.26 m apart, so the nearest point by distance sat on a
different branch, and ``max(best, previous_idx)`` made it permanent.
"""

from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.path_utils import PathUtils  # noqa: E402


def overlapping_route():
    """Two straight branches 0.10 m apart, same heading, far apart in index.

    Mirrors the real track: an outbound leg and a return leg sharing a
    corridor, hundreds of waypoints apart along the path.
    """
    n = 300
    xs = np.linspace(0.0, 15.0, n)

    out = np.column_stack([xs, np.zeros(n), np.zeros(n)])
    back = np.column_stack([xs, np.full(n, 0.10), np.zeros(n)])

    # A spacer keeps the two branches far apart in index, as on a real lap.
    gap = np.column_stack([
        np.full(60, 15.0),
        np.linspace(0.0, 0.10, 60),
        np.zeros(60),
    ])
    return np.vstack([out, gap, back])


def test_reference_does_not_hop_to_a_parallel_branch():
    route = overlapping_route()
    previous_idx = 100

    # Car is beside branch A but displaced enough that the windowed search
    # exceeds the 0.75 m recovery threshold -- e.g. after a Cartographer
    # wobble. Branch B is nearer in raw distance, 360 indices away.
    pose = (route[100, 0], 0.90, 0.0)

    idx = PathUtils.closest_point(
        pose,
        route,
        previous_idx,
        window_back=10,
        window_forward=120,
    )

    # It must stay in the neighbourhood, not leap onto the return leg.
    assert idx < 360, f"reference hopped to a parallel branch: idx={idx}"


def test_windowed_search_still_tracks_normal_progress():
    route = overlapping_route()
    pose = (route[105, 0], 0.0, 0.0)

    idx = PathUtils.closest_point(
        pose, route, 100, window_back=10, window_forward=120,
    )
    assert idx == 105


def test_reference_never_regresses():
    route = overlapping_route()
    pose = (route[90, 0], 0.0, 0.0)

    idx = PathUtils.closest_point(
        pose, route, 100, window_back=10, window_forward=120,
    )
    assert idx >= 100


def test_global_search_still_acquires_anywhere():
    route = overlapping_route()
    pose = (route[250, 0], 0.0, 0.0)

    idx = PathUtils.closest_point(
        pose, route, 0, global_search=True,
    )
    assert 245 <= idx <= 255


def test_bounded_recovery_still_crosses_a_real_localization_jump():
    """A genuine jump of a few metres must still be recovered."""
    n = 400
    xs = np.linspace(0.0, 20.0, n)
    route = np.column_stack([xs, np.zeros(n), np.zeros(n)])

    # The stale window reaches only idx 119. The car is at 137, which is
    # 0.90 m beyond it -- past the 0.75 m recovery threshold, and inside the
    # bounded recovery span (previous_idx + 2 * window_forward = 140).
    pose = (route[137, 0], 0.0, 0.0)

    idx = PathUtils.closest_point(
        pose, route, 100, window_back=10, window_forward=20,
    )
    assert idx == 137
