"""Pure geometry tests for the MPC overtake-curvature gate."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def _stub_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_path_mpc_geometry():
    """Load the production helpers without requiring ROS in pure CI."""
    placeholder = type("Placeholder", (), {})
    stubs = {
        "casadi": _stub_module("casadi"),
        "rclpy": _stub_module("rclpy"),
        "rclpy.node": _stub_module("rclpy.node", Node=object),
        "geometry_msgs": _stub_module("geometry_msgs"),
        "geometry_msgs.msg": _stub_module(
            "geometry_msgs.msg",
            PointStamped=placeholder,
            PoseWithCovarianceStamped=placeholder,
            Twist=placeholder,
        ),
        "nav_msgs": _stub_module("nav_msgs"),
        "nav_msgs.msg": _stub_module("nav_msgs.msg", Path=placeholder),
        "std_msgs": _stub_module("std_msgs"),
        "std_msgs.msg": _stub_module(
            "std_msgs.msg",
            Bool=placeholder,
            Float32=placeholder,
            String=placeholder,
            Int32=placeholder,
        ),
        "tf2_ros": _stub_module(
            "tf2_ros",
            Buffer=placeholder,
            TransformListener=placeholder,
            TransformException=Exception,
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = (
            PACKAGE_ROOT
            / "qcar_science_night_pkg"
            / "path_mpc_node.py"
        )
        spec = spec_from_file_location("_path_mpc_geometry_test", module_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


MPC = _load_path_mpc_geometry()


def test_constant_left_offset_can_be_kinematically_tighter():
    radius = 2.0
    offset = 0.47
    angle = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    trajectory = np.column_stack((
        radius * np.cos(angle),
        radius * np.sin(angle),
        angle + np.pi / 2.0,
        np.full_like(angle, 1.0 / radius),
    ))

    offset_curvature = MPC.offset_path_curvature(
        trajectory, offset, closed_path=True
    )

    np.testing.assert_allclose(
        np.median(offset_curvature),
        1.0 / (radius - offset),
        rtol=2e-3,
    )
    assert np.min(offset_curvature) > 1.0 / radius


def test_offset_preview_blocks_when_nominal_preview_is_straight():
    nominal = np.full(30, 0.10)
    offset = nominal.copy()
    offset[8] = 0.45

    allowed, metrics = MPC.overtake_curvature_preview(
        nominal,
        offset,
        start_idx=4,
        horizon=10,
        closed_path=False,
        max_limit=0.40,
        mean_limit=0.25,
    )

    assert not allowed
    assert metrics["nominal_max"] == 0.10
    assert metrics["offset_max"] == 0.45


def test_offset_preview_mean_curvature_is_also_required():
    nominal = np.full(30, 0.10)
    offset = np.full(30, 0.30)

    allowed, metrics = MPC.overtake_curvature_preview(
        nominal,
        offset,
        start_idx=0,
        horizon=20,
        closed_path=False,
        max_limit=0.40,
        mean_limit=0.25,
    )

    assert not allowed
    assert metrics["offset_max"] < 0.40
    assert metrics["offset_mean"] > 0.25


def test_closed_preview_checks_offset_curvature_across_seam():
    nominal = np.full(20, 0.05)
    offset = nominal.copy()
    offset[0] = 0.50

    allowed, metrics = MPC.overtake_curvature_preview(
        nominal,
        offset,
        start_idx=18,
        horizon=5,
        closed_path=True,
        max_limit=0.40,
        mean_limit=0.25,
    )

    assert not allowed
    assert metrics["offset_max"] == 0.50


def test_both_straight_previews_allow_overtake():
    nominal = np.full(30, 0.08)
    offset = np.full(30, 0.12)

    allowed, metrics = MPC.overtake_curvature_preview(
        nominal,
        offset,
        start_idx=5,
        horizon=15,
        closed_path=False,
        max_limit=0.40,
        mean_limit=0.25,
    )

    assert allowed
    assert metrics["nominal_max"] < metrics["offset_max"]


# ---- Signed curvature for the LiDAR detection corridor ----

def test_signed_preview_keeps_the_direction_of_the_bend():
    # overtake_curvature_preview takes np.abs because a lane change is equally
    # hard either way.  The LiDAR corridor needs the opposite: it has to bend
    # with the road, so a right-hand bend must not read as a left-hand one.
    left = np.full(20, 0.9)
    right = np.full(20, -0.9)

    assert MPC.signed_curvature_preview(left, 0, 10, False) > 0.0
    assert MPC.signed_curvature_preview(right, 0, 10, False) < 0.0


def test_signed_preview_averages_over_the_horizon():
    curvature = np.concatenate([np.zeros(10), np.full(10, 1.0)])

    # Entirely on the straight section.
    assert MPC.signed_curvature_preview(curvature, 0, 5, False) == 0.0
    # Half straight, half bend.
    assert MPC.signed_curvature_preview(curvature, 5, 10, False) == 0.5


def test_signed_preview_wraps_only_on_a_closed_path():
    curvature = np.concatenate([np.full(5, 1.0), np.zeros(15)])

    # Closed: the horizon runs off the end and back onto the leading bend.
    wrapped = MPC.signed_curvature_preview(curvature, 18, 5, True)
    assert wrapped > 0.0

    # Open: the last sample is held instead, so the bend is never reached.
    assert MPC.signed_curvature_preview(curvature, 18, 5, False) == 0.0


def test_signed_preview_falls_back_to_straight_on_unusable_input():
    assert MPC.signed_curvature_preview(np.array([]), 0, 5, False) == 0.0
    assert MPC.signed_curvature_preview(np.full(5, 0.4), 0, 0, False) == 0.0
    assert MPC.signed_curvature_preview(np.full(5, np.nan), 0, 5, False) == 0.0
