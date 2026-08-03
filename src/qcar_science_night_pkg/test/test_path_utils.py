"""Unit tests for MPC trajectory indexing helpers."""

from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.path_utils import PathUtils  # noqa: E402


def _straight_trajectory(count):
    x = np.arange(count, dtype=float) * 0.03
    return np.column_stack(
        [x, np.zeros(count), np.zeros(count), np.zeros(count)]
    )


def test_startup_search_can_acquire_waypoint_beyond_old_window():
    trajectory = _straight_trajectory(300)
    pose = np.asarray([7.5, 0.01, 0.0])

    closest = PathUtils.closest_point(
        pose, trajectory, previous_idx=0, global_search=True
    )

    assert closest == 250


def test_startup_search_uses_heading_at_self_intersection():
    trajectory = _straight_trajectory(220)
    trajectory[10, :3] = [1.0, 1.0, 0.0]
    trajectory[200, :3] = [1.0, 1.0, np.pi]
    pose = np.asarray([1.0, 1.0, np.pi])

    closest = PathUtils.closest_point(
        pose, trajectory, previous_idx=0, global_search=True
    )

    assert closest == 200


def test_recovery_search_never_regresses_completed_path():
    trajectory = _straight_trajectory(300)
    pose = np.asarray([3.0, 0.0, 0.0])  # Geometrically near index 100.

    closest = PathUtils.closest_point(pose, trajectory, previous_idx=150)

    assert closest >= 150


def test_loop_reset_local_search_does_not_reacquire_path_end():
    trajectory = _straight_trajectory(220)
    trajectory[-1, :3] = [0.01, 0.0, np.pi]
    pose = np.asarray([0.01, 0.0, np.pi])

    closest = PathUtils.closest_point(
        pose, trajectory, previous_idx=0, global_search=False
    )

    assert closest < 120
    assert closest != len(trajectory) - 1


def test_index_zero_does_not_imply_global_search_after_loop_reset():
    trajectory = _straight_trajectory(300)
    pose = np.asarray([8.9, 0.0, 0.0])

    closest = PathUtils.closest_point(
        pose, trajectory, previous_idx=0, global_search=False
    )

    assert closest < 120


def test_load_xy_trajectory_adds_forward_yaw_and_curvature(tmp_path):
    path = tmp_path / "line.npy"
    np.save(path, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))

    trajectory = PathUtils.load_trajectory(path)

    assert trajectory.shape == (3, 4)
    np.testing.assert_allclose(trajectory[:, 2], 0.0)
    np.testing.assert_allclose(trajectory[:, 3], 0.0)


def test_zero_target_speed_freezes_every_reference_stage():
    trajectory = _straight_trajectory(20)

    ref = PathUtils.build_reference(
        trajectory,
        idx=7,
        horizon=8,
        dt=0.08,
        spacing=0.03,
        target_v=0.0,
    ).reshape(-1, 3)

    np.testing.assert_allclose(ref, np.tile(trajectory[7, :3], (8, 1)))
    assert PathUtils.step_indices_for_speed(0.0, 0.08, 0.03) == 0


def test_reference_uses_fractional_arc_length_not_whole_waypoints():
    trajectory = _straight_trajectory(20)

    ref = PathUtils.build_reference(
        trajectory,
        idx=2,
        horizon=5,
        dt=0.10,
        spacing=0.03,
        target_v=0.10,
    ).reshape(-1, 3)

    # Start at x=.06 and advance exactly .01 m per stage.  The old integer
    # stepping advanced .03 m per stage at this speed.
    np.testing.assert_allclose(ref[:, 0], [0.06, 0.07, 0.08, 0.09, 0.10])


def test_closed_reference_wraps_continuously_across_canonical_seam():
    trajectory = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, np.pi / 2.0, 0.0],
        [1.0, 1.0, np.pi, 0.0],
        [0.0, 1.0, -np.pi / 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    ref = PathUtils.build_reference(
        trajectory,
        idx=3,
        horizon=4,
        dt=0.5,
        spacing=1.0,
        target_v=1.0,
        loop_path=True,
    ).reshape(-1, 3)

    np.testing.assert_allclose(
        ref[:, :2],
        [[0.0, 1.0], [0.0, 0.5], [0.0, 0.0], [0.5, 0.0]],
        atol=1e-12,
    )


def test_forward_command_cannot_exceed_target_or_v2v_cap():
    assert PathUtils.enforce_forward_speed_cap(0.30, 0.20, 0.07) == 0.07
    assert PathUtils.enforce_forward_speed_cap(0.30, 0.05, 0.20) == 0.05


def test_zero_cap_defeats_any_minimum_speed_override():
    assert PathUtils.enforce_forward_speed_cap(0.08, 0.20, 0.0) == 0.0
    assert PathUtils.enforce_forward_speed_cap(0.30, 0.0, None) == 0.0


def test_v2v_cap_is_hard_zero_inside_stop_gap():
    cap = PathUtils.v2v_following_cap(
        gap=0.69,
        lead_speed=0.30,
        stop_gap=0.70,
        follow_gap=1.20,
        follow_gain=0.5,
        soft_decel=0.5,
    )

    assert cap == 0.0


def test_hard_v2v_shield_accepts_clear_overtake_horizon():
    vehicle = np.asarray([[0.0, 0.47, 0.0], [0.1, 0.47, 0.0]])
    obstacle = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    assert PathUtils.predicted_ellipse_clear(
        vehicle, obstacle, 0.55, 0.40
    )


def test_hard_v2v_shield_rejects_predicted_collision():
    vehicle = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    obstacle = np.asarray([[0.6, 0.0, 0.0], [0.3, 0.0, 0.0]])

    assert not PathUtils.predicted_ellipse_clear(
        vehicle, obstacle, 0.55, 0.40
    )


def test_hard_v2v_shield_fails_closed_on_nonfinite_data():
    vehicle = np.asarray([[np.nan, 0.0, 0.0]])
    obstacle = np.asarray([[0.0, 0.0, 0.0]])

    assert not PathUtils.predicted_ellipse_clear(
        vehicle, obstacle, 0.55, 0.40
    )


def test_offset_profile_adds_lane_change_heading_then_parallel_lane():
    ref = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.2, 0.2, 0.0],
        [0.3, 0.2, 0.0],
    ]).reshape(-1)

    shifted = PathUtils.apply_offset_profile(
        ref, np.asarray([0.0, 0.1, 0.2, 0.2])
    ).reshape(-1, 3)

    np.testing.assert_allclose(shifted[:, 1], [0.0, 0.2, 0.4, 0.4])
    assert shifted[1, 2] > 0.0
    assert abs(shifted[-1, 2]) < 1e-12
