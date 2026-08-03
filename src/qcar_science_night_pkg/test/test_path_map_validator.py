"""Focused tests for trajectory/map compatibility checks."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.path_map_validator import read_pgm  # noqa: E402
from qcar_science_night_pkg.path_map_validator import (  # noqa: E402
    validate_trajectory_against_map as _validate_trajectory_against_map,
)


def validate_trajectory_against_map(*args, **kwargs):
    # Synthetic maps use one-metre cells; individual tests target map
    # classification rather than the QCar's 0.10 m spacing safety limit.
    kwargs.setdefault("max_waypoint_spacing_m", 2.0)
    return _validate_trajectory_against_map(*args, **kwargs)


def _write_map(tmp_path, pixels, *, pgm_format="P5", origin=(0, 0, 0)):
    pixels = np.asarray(pixels, dtype=np.uint8)
    pgm_path = tmp_path / "map.pgm"
    height, width = pixels.shape

    if pgm_format == "P5":
        header = f"P5\n# generated test map\n{width} {height}\n255\n"
        pgm_path.write_bytes(header.encode("ascii") + pixels.tobytes())
    else:
        rows = [" ".join(str(int(value)) for value in row) for row in pixels]
        content = (
            f"P2\n# generated test map\n{width} {height}\n255\n"
            + "\n".join(rows)
            + "\n"
        )
        pgm_path.write_text(content, encoding="ascii")

    yaml_path = tmp_path / "map.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "image: map.pgm",
                "resolution: 1.0",
                f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path, pgm_path


def _write_trajectory(tmp_path, rows):
    path = tmp_path / "trajectory.npy"
    np.save(path, np.asarray(rows, dtype=float))
    return path


def test_read_pgm_supports_ascii_comments(tmp_path):
    expected = np.asarray([[0, 205, 254], [254, 205, 0]], dtype=np.uint8)
    _, pgm_path = _write_map(tmp_path, expected, pgm_format="P2")

    actual, max_value = read_pgm(pgm_path)

    assert max_value == 255
    np.testing.assert_array_equal(actual, expected)


def test_validator_accepts_free_path_and_flips_map_rows(tmp_path):
    pixels = np.full((3, 3), 254, dtype=np.uint8)
    pixels[0, 0] = 0  # Top-left image pixel is highest-y world cell.
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        max_unknown_fraction=0.0,
        max_clearance_violation_fraction=0.0,
    )

    assert report["ok"] is True
    assert report["free_fraction"] == 1.0
    assert report["occupied_fraction"] == 0.0

    high_path = _write_trajectory(
        tmp_path,
        [
            [0.5, 2.5, 0.0],
            [0.6, 2.5, 0.0],
            [0.7, 2.5, 0.0],
        ],
    )
    high_report = validate_trajectory_against_map(
        high_path,
        map_path,
        clearance_m=0.0,
    )
    assert high_report["occupied_fraction"] == 1.0
    assert high_report["ok"] is False


def test_validator_reports_unknown_occupied_and_outside_points(tmp_path):
    pixels = np.full((4, 4), 254, dtype=np.uint8)
    pixels[3, 1] = 127
    pixels[3, 2] = 0
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        max_unknown_fraction=0.0,
    )

    assert report["ok"] is False
    assert report["free_fraction"] == 0.25
    assert report["unknown_fraction"] == 0.25
    assert report["occupied_fraction"] == 0.25
    assert report["outside_fraction"] == 0.25
    assert report["clearance_violation_fraction"] == 0.75
    assert len(report["failures"]) == 4


def test_validator_rejects_duplicate_points_and_bad_headings(tmp_path):
    pixels = np.full((5, 5), 254, dtype=np.uint8)
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [0.5, 0.5, np.pi],
            [0.5, 0.5, np.pi],
            [1.5, 0.5, np.pi],
            [2.5, 0.5, np.pi],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
    )

    assert report["ok"] is False
    assert report["yaw_tangent_bad_fraction"] > 0.05
    assert any("duplicate" in failure for failure in report["failures"])
    assert any("headings" in failure for failure in report["failures"])


def test_validator_rotated_origin_keeps_free_cells_in_bounds(tmp_path):
    pixels = np.full((3, 3), 254, dtype=np.uint8)
    map_path, _ = _write_map(
        tmp_path, pixels, origin=(0.0, 0.0, np.pi / 2.0)
    )
    path = _write_trajectory(
        tmp_path,
        [
            [-0.5, 0.5, np.pi / 2.0],
            [-0.5, 1.5, np.pi / 2.0],
            [-0.5, 2.5, np.pi / 2.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        max_unknown_fraction=0.0,
        max_clearance_violation_fraction=0.0,
    )

    assert report["ok"] is True
    assert report["free_fraction"] == 1.0


def test_validator_rejects_infeasible_loop_seam(tmp_path):
    pixels = np.full((5, 5), 254, dtype=np.uint8)
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, np.pi / 2.0],
            [1.5, 1.5, math.atan2(-0.9, -1.0)],
            [0.5, 0.6, -np.pi / 2.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        loop_path=True,
    )

    assert report["ok"] is False
    assert report["loop_seam_required_steering_rad"] > 0.58
    assert any("loop seam requires" in item for item in report["failures"])


def test_validator_rotated_origin_classifies_occupied_cell(tmp_path):
    pixels = np.full((3, 3), 254, dtype=np.uint8)
    pixels[2, 0] = 0
    map_path, _ = _write_map(
        tmp_path, pixels, origin=(10.0, 20.0, np.pi / 2.0)
    )
    path = _write_trajectory(
        tmp_path,
        [
            [9.9, 20.1, np.pi / 2.0],
            [9.9, 20.2, np.pi / 2.0],
            [9.9, 20.3, np.pi / 2.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
    )

    assert report["occupied_fraction"] == 1.0
    assert report["outside_fraction"] == 0.0
    assert report["ok"] is False


def test_validator_rejects_unsteerable_loop_seam(tmp_path):
    pixels = np.full((5, 5), 254, dtype=np.uint8)
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [1.0, 1.0, 0.0],
            [1.1, 1.0, 0.0],
            [1.2, 1.0, 0.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        loop_path=True,
        max_loop_gap_m=0.3,
    )

    assert report["loop_seam_required_steering_rad"] > 0.58
    assert report["ok"] is False
    assert any("loop seam" in failure for failure in report["failures"])


def test_validator_rejects_excessive_uniform_spacing(tmp_path):
    pixels = np.full((5, 5), 254, dtype=np.uint8)
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [
            [0.5, 0.5, 0.0],
            [0.7, 0.5, 0.0],
            [0.9, 0.5, 0.0],
        ],
    )

    report = validate_trajectory_against_map(
        path,
        map_path,
        clearance_m=0.0,
        max_waypoint_spacing_m=0.10,
    )

    assert report["ok"] is False
    assert any(
        "maximum waypoint spacing" in item for item in report["failures"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"clearance_m": -0.01},
        {"max_unknown_fraction": 1.01},
        {"max_steering_violation_fraction": -0.01},
    ],
)
def test_validator_rejects_invalid_safety_limits(tmp_path, overrides):
    pixels = np.full((3, 3), 254, dtype=np.uint8)
    map_path, _ = _write_map(tmp_path, pixels)
    path = _write_trajectory(
        tmp_path,
        [[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]],
    )

    with pytest.raises(ValueError):
        validate_trajectory_against_map(path, map_path, **overrides)
