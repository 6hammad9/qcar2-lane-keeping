"""Unit tests for trajectory recording and initial-pose message helpers."""

import importlib
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import time

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def _install_ros_import_stubs():
    try:
        importlib.import_module("rclpy")
        return
    except ModuleNotFoundError:
        pass

    rclpy = ModuleType("rclpy")
    rclpy_executors = ModuleType("rclpy.executors")
    rclpy_node = ModuleType("rclpy.node")
    rclpy_time = ModuleType("rclpy.time")
    geometry_msgs = ModuleType("geometry_msgs")
    geometry_msgs_msg = ModuleType("geometry_msgs.msg")
    tf2_ros = ModuleType("tf2_ros")

    class PoseWithCovarianceStamped:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id="")
            position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
            pose = SimpleNamespace(position=position, orientation=orientation)
            self.pose = SimpleNamespace(pose=pose, covariance=[0.0] * 36)

    class ExternalShutdownException(Exception):
        pass

    rclpy_executors.ExternalShutdownException = ExternalShutdownException
    rclpy_node.Node = object
    rclpy_time.Time = object
    geometry_msgs_msg.PoseWithCovarianceStamped = PoseWithCovarianceStamped
    geometry_msgs.msg = geometry_msgs_msg
    tf2_ros.Buffer = object
    tf2_ros.TransformListener = object
    tf2_ros.TransformException = RuntimeError

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.executors"] = rclpy_executors
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["rclpy.time"] = rclpy_time
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg
    sys.modules["tf2_ros"] = tf2_ros


_install_ros_import_stubs()

from qcar_science_night_pkg.trajectory_initial_pose import (  # noqa: E402
    TrajectoryInitialPose,
)
from qcar_science_night_pkg.trajectory_recorder import (  # noqa: E402
    TrajectoryRecorder,
    quaternion_to_yaw,
    resample_trajectory,
)


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(("info", message))

    def error(self, message, **_kwargs):
        self.messages.append(("error", message))


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _FakeTimePoint:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def __sub__(self, other):
        return _FakeTimePoint(self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        return self.nanoseconds


def test_quaternion_to_yaw_handles_quarter_turn():
    yaw = math.pi / 2.0
    quaternion = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )

    assert math.isclose(quaternion_to_yaw(quaternion), yaw, abs_tol=1e-12)


def test_resample_trajectory_produces_uniform_tangent_yaw():
    points = np.asarray(
        [
            [0.0, 0.0, 1.2],
            [0.04, 0.0, -2.0],
            [0.15, 0.0, 0.4],
            [0.30, 0.0, 2.4],
        ]
    )

    trajectory = resample_trajectory(points, spacing=0.05)

    spacing = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-12)
    np.testing.assert_allclose(trajectory[:, 2], 0.0, atol=1e-12)


def test_recorder_save_writes_resampled_npy_raw_npy_and_csv(tmp_path):
    node = object.__new__(TrajectoryRecorder)
    node.output_file = str(tmp_path / "recorded.npy")
    node.output_spacing = 0.1
    node.points = [
        [0.0, 0.0, 3.10],
        [0.07, 0.0, -3.10],
        [0.20, 0.0, -3.00],
        [0.30, 0.0, -2.90],
    ]
    node.saved = False
    logger = _FakeLogger()
    node.get_logger = lambda: logger

    node.save()

    trajectory = np.load(node.output_file)
    assert trajectory.shape == (4, 3)
    spacing = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    np.testing.assert_allclose(spacing, 0.1, atol=1e-12)
    assert (tmp_path / "recorded_raw.npy").exists()
    assert (tmp_path / "recorded.csv").exists()
    assert node.saved is True


def test_initial_pose_builds_normalized_orientation_and_covariance():
    node = object.__new__(TrajectoryInitialPose)
    node.started_at = time.monotonic() - 1.0
    node.timeout_sec = 30.0
    node.wait_for_odom = True
    node.x = 1.25
    node.y = -0.5
    node.yaw = -2.7
    node.index = 17
    node.position_variance = 0.25
    node.yaw_variance = 0.0685
    node.done = False
    node.success = False
    node.timer = _FakeTimer()
    logger = _FakeLogger()
    node.get_logger = lambda: logger
    clock = SimpleNamespace(now=lambda: _FakeTimePoint(1_000_000_000))
    node.get_clock = lambda: clock

    class Publisher:
        def __init__(self):
            self.messages = []

        def get_subscription_count(self):
            return 1

        def publish(self, message):
            self.messages.append(message)

    class Buffer:
        @staticmethod
        def lookup_transform(*_args):
            return object()

    node.publisher = Publisher()
    node.tf_buffer = Buffer()

    node._try_publish()

    assert node.success is True
    assert node.done is True
    assert node.timer.cancelled is True
    assert len(node.publisher.messages) == 1
    message = node.publisher.messages[0]
    assert message.header.frame_id == "map"
    assert math.isclose(message.pose.pose.position.x, node.x)
    assert math.isclose(message.pose.pose.position.y, node.y)
    assert math.isclose(message.pose.pose.orientation.z, math.sin(-2.7 / 2.0))
    assert math.isclose(message.pose.pose.orientation.w, math.cos(-2.7 / 2.0))
    assert message.pose.covariance[0] == node.position_variance
    assert message.pose.covariance[7] == node.position_variance
    assert message.pose.covariance[35] == node.yaw_variance
