"""Focused tests for safety-critical V2V along-path distances."""

from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.v2v_common import PathProjector  # noqa: E402


def _trajectory(points):
    points = np.asarray(points, dtype=float)
    yaw = np.zeros(len(points), dtype=float)
    return np.column_stack([points, yaw, np.zeros(len(points))])


def test_gap_uses_measured_arc_length_not_nominal_spacing():
    trajectory = _trajectory([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.35, 0.0],
        [0.50, 0.0],
    ])
    projector = PathProjector(trajectory, spacing=0.03, loop=False)

    assert projector.gap_along(0, 3) == 0.50
    assert projector.gap_along(3, 0) == -1.0
    assert projector.signed_gap_along(0, 3) == 0.50
    assert projector.signed_gap_along(3, 0) == -0.50
    assert projector.path_length() == 0.50


def test_loop_gap_includes_the_physical_closing_seam():
    trajectory = _trajectory([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    projector = PathProjector(trajectory, spacing=0.03, loop=True)

    assert projector.path_length() == 4.0
    assert projector.gap_along(3, 0) == 1.0
    assert projector.gap_along(2, 1) == 3.0
    assert projector.signed_gap_along(3, 0) == 1.0
    assert projector.signed_gap_along(0, 3) == -1.0


def test_duplicate_closed_endpoint_wraps_without_extra_distance():
    trajectory = _trajectory([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    projector = PathProjector(trajectory, spacing=0.03, loop=True)

    assert projector.path_length() == 4.0
    assert projector.gap_along(4, 0) == 0.0
    assert projector.gap_along(4, 1) == 1.0
