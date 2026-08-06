#!/usr/bin/env python3

import hashlib
import math
import os
import time
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32, String, Int32
from tf2_ros import Buffer, TransformListener, TransformException
from qcar_science_night_pkg.path_map_validator import (
    format_report,
    validate_trajectory_against_map,
)
from qcar_science_night_pkg.path_utils import PathUtils


def path_curvature_from_xy(xy, closed_path=False):
    """Return curvature magnitude, using periodic derivatives for a loop."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 3:
        raise ValueError("xy must contain at least three finite 2D points")
    xy = xy[:, :2]
    if not np.all(np.isfinite(xy)):
        raise ValueError("xy must contain at least three finite 2D points")

    if closed_path:
        previous = np.roll(xy, 1, axis=0)
        following = np.roll(xy, -1, axis=0)
        derivative = 0.5 * (following - previous)
        second_derivative = following - 2.0 * xy + previous
    else:
        derivative = np.gradient(xy, axis=0, edge_order=2)
        second_derivative = np.gradient(
            derivative, axis=0, edge_order=2
        )

    numerator = np.abs(
        derivative[:, 0] * second_derivative[:, 1]
        - derivative[:, 1] * second_derivative[:, 0]
    )
    denominator = np.power(
        np.sum(derivative * derivative, axis=1), 1.5
    )
    curvature = np.full(len(xy), np.inf, dtype=float)
    valid = denominator > 1e-9
    curvature[valid] = numerator[valid] / denominator[valid]
    return curvature


def offset_path_curvature(trajectory, offset_left, closed_path=False):
    """Precompute curvature of a constant left-offset reference path."""
    trajectory = np.asarray(trajectory, dtype=float)
    if (
        trajectory.ndim != 2
        or trajectory.shape[1] < 3
        or len(trajectory) < 3
        or not np.all(np.isfinite(trajectory[:, :3]))
        or not np.isfinite(offset_left)
    ):
        raise ValueError("trajectory and offset must be finite")

    yaw = trajectory[:, 2]
    offset_xy = trajectory[:, :2].copy()
    offset_xy[:, 0] -= np.sin(yaw) * float(offset_left)
    offset_xy[:, 1] += np.cos(yaw) * float(offset_left)
    return path_curvature_from_xy(offset_xy, closed_path=closed_path)


def overtake_curvature_preview(
        nominal_curvature,
        offset_curvature,
        start_idx,
        horizon,
        closed_path,
        max_limit,
        mean_limit):
    """Gate overtaking on both nominal- and passing-lane geometry."""
    nominal = np.asarray(nominal_curvature, dtype=float).reshape(-1)
    offset = np.asarray(offset_curvature, dtype=float).reshape(-1)
    if len(nominal) == 0 or len(nominal) != len(offset):
        raise ValueError("nominal and offset curvature must have equal size")
    if horizon <= 0:
        raise ValueError("curvature preview horizon must be positive")

    start_idx = int(np.clip(start_idx, 0, len(nominal) - 1))
    indices = start_idx + np.arange(int(horizon), dtype=int)
    if closed_path:
        indices %= len(nominal)
    else:
        indices = np.minimum(indices, len(nominal) - 1)

    nominal_preview = np.abs(nominal[indices])
    offset_preview = np.abs(offset[indices])
    metrics = {
        "nominal_max": float(np.max(nominal_preview)),
        "nominal_mean": float(np.mean(nominal_preview)),
        "offset_max": float(np.max(offset_preview)),
        "offset_mean": float(np.mean(offset_preview)),
    }
    allowed = (
        np.all(np.isfinite(list(metrics.values())))
        and metrics["nominal_max"] < max_limit
        and metrics["nominal_mean"] < mean_limit
        and metrics["offset_max"] < max_limit
        and metrics["offset_mean"] < mean_limit
    )
    return bool(allowed), metrics


def signed_curvature_preview(
        nominal_curvature,
        start_idx,
        horizon,
        closed_path):
    """Mean *signed* curvature of the route just ahead, in 1/m.

    ``overtake_curvature_preview`` takes ``np.abs`` because a lane change is
    equally hard in either direction. The LiDAR corridor needs the opposite:
    it has to know *which way* the road bends so it can bend with it, so this
    keeps the sign. Positive is a left turn, matching the y-left-positive
    convention in ``LidarSectorAnalyzer.scan_to_xy``.

    Returns 0.0 for empty or non-finite input; a straight corridor is the
    correct fallback because it is what the analyzer used before curvature
    was available.
    """
    nominal = np.asarray(nominal_curvature, dtype=float).reshape(-1)
    if len(nominal) == 0 or horizon <= 0:
        return 0.0

    start_idx = int(np.clip(start_idx, 0, len(nominal) - 1))
    indices = start_idx + np.arange(int(horizon), dtype=int)
    if closed_path:
        indices %= len(nominal)
    else:
        indices = np.minimum(indices, len(nominal) - 1)

    preview = nominal[indices]
    preview = preview[np.isfinite(preview)]
    if len(preview) == 0:
        return 0.0

    return float(np.mean(preview))


class QCar2PathMPC(Node):

    def __init__(self):
        super().__init__("qcar2_path_mpc_node")

        default_workspace = os.path.expanduser("~/ros2_ws")
        self.declare_parameter(
            "trajectory_file",
            os.path.join(default_workspace, "recorded_path_amcl_final_long.npy"),
        )
        self.declare_parameter(
            "reverse_trajectory_file",
            os.path.join(default_workspace, "recorded_path_reverse27.npy"),
        )
        self.declare_parameter(
            "map_file",
            os.path.join(default_workspace, "track_map_new.yaml"),
        )
        self.declare_parameter("require_map_validation", True)
        self.declare_parameter("allow_unsafe_path", False)
        self.declare_parameter("path_clearance_m", 0.10)
        self.declare_parameter("max_unknown_fraction", 0.05)
        self.declare_parameter("max_clearance_violation_fraction", 0.01)
        self.declare_parameter("max_speed", 0.20)
        self.declare_parameter("curve_speed", 0.12)
        self.declare_parameter("startup_speed", 0.10)
        self.declare_parameter("maneuver_speed", 0.10)
        # Policy ceiling on max_speed/curve_speed. Default preserves the
        # previous hardcoded 0.60 limit; raise it deliberately, per route.
        self.declare_parameter("speed_limit_ceiling", 0.60)
        # Braking-aware speed limiting. brake_lookahead_m MUST grow with
        # max_speed -- see the check at the assignment site.
        self.declare_parameter("max_decel", 0.8)
        self.declare_parameter("brake_lookahead_m", 0.35)
        self.declare_parameter("curve_kappa_threshold", 0.3)
        # Waypoint search window, in waypoints. Narrow the forward window on
        # a self-crossing route so the reference cannot hop to a parallel
        # branch -- see PathUtils.closest_point.
        self.declare_parameter("search_window_back", 10)
        self.declare_parameter("search_window_forward", 120)
        self.declare_parameter("max_steer", 0.50)
        self.declare_parameter("wheelbase", 0.256)
        self.declare_parameter("path_spacing", 0.03)
        self.declare_parameter("loop_path", False)
        self.declare_parameter("target_laps", 1)
        self.declare_parameter("enable_reverse", False)
        # Lane centering and overtaking both shift the waypoint reference.
        # Enable that behavior by default, but constrain every requested
        # shift with max_reference_offset below.
        self.declare_parameter("enable_reference_offsets", True)
        self.declare_parameter("max_reference_offset", 0.65)
        self.declare_parameter("behavior_timeout_sec", 1.0)
        self.declare_parameter("overtake_max_curvature", 0.40)
        self.declare_parameter("overtake_mean_curvature", 0.25)
        # Lane centering. See the assignment site for why 0.04 m could never
        # work and why the curvature gate is now a backstop, not the gate.
        self.declare_parameter("max_lane_offset_m", 0.12)
        self.declare_parameter("min_lane_offset_m", 0.04)
        self.declare_parameter("lane_confidence_floor", 0.25)
        self.declare_parameter("lane_centering_curve_limit", 1.20)
        self.declare_parameter("solver_timeout_sec", 0.20)
        self.declare_parameter("lane_change_distance_m", 0.90)
        self.declare_parameter("enable_v2v", False)
        self.declare_parameter("max_start_position_error", 0.35)
        self.declare_parameter("max_start_yaw_error_deg", 35.0)
        self.declare_parameter("max_tracking_error", 0.75)
        self.declare_parameter("max_tracking_yaw_error_deg", 75.0)
        self.declare_parameter("require_amcl_quality", True)
        self.declare_parameter("max_amcl_position_variance", 0.10)
        self.declare_parameter("max_amcl_yaw_variance", 0.10)
        self.declare_parameter(
            "tracking_log_file",
            os.path.join(default_workspace, "mpc_tracking_log.csv"),
        )

        self.forward_trajectory_file = os.path.abspath(os.path.expanduser(
            self.get_parameter("trajectory_file").value
        ))
        self.reverse_trajectory_file = os.path.abspath(os.path.expanduser(
            self.get_parameter("reverse_trajectory_file").value
        ))
        self.map_file = os.path.abspath(os.path.expanduser(
            self.get_parameter("map_file").value
        ))
        self.require_map_validation = bool(
            self.get_parameter("require_map_validation").value
        )
        self.allow_unsafe_path = bool(self.get_parameter("allow_unsafe_path").value)
        self.path_clearance_m = float(self.get_parameter("path_clearance_m").value)
        self.max_unknown_fraction = float(
            self.get_parameter("max_unknown_fraction").value
        )
        self.max_clearance_violation_fraction = float(
            self.get_parameter("max_clearance_violation_fraction").value
        )
        self.enable_reverse = bool(self.get_parameter("enable_reverse").value)
        self.enable_reference_offsets = bool(
            self.get_parameter("enable_reference_offsets").value
        )
        self.max_reference_offset = float(
            self.get_parameter("max_reference_offset").value
        )
        if not 0.05 <= self.max_reference_offset <= 1.0:
            raise ValueError("max_reference_offset must be in [0.05, 1.0] m")
        self.behavior_timeout_sec = float(
            self.get_parameter("behavior_timeout_sec").value
        )
        if not 0.2 <= self.behavior_timeout_sec <= 5.0:
            raise ValueError("behavior_timeout_sec must be in [0.2, 5.0] s")

        self.N = 25
        self.dt = 0.08
        self.L = float(self.get_parameter("wheelbase").value)

        # Hard stop above which no parameter can take the vehicle. Beyond
        # this the MPC's 0.08 s control period and the LiDAR box stop being
        # a meaningful safety margin at all, so it is not configurable.
        self.absolute_speed_ceiling = 2.0

        self.v_min = 0.0
        self.v_max = float(self.get_parameter("max_speed").value)
        self.v_curve_min = float(self.get_parameter("curve_speed").value)
        requested_maneuver_speed = float(
            self.get_parameter("maneuver_speed").value
        )

        # The 0.60 default is a policy limit, not a limit of the car -- manual
        # teleop drives it far faster. It exists because what degrades first
        # under autonomy is not grip: it is solve latency (the MPC already
        # logs "Slow MPC solve" near its 0.08 s period), localization update
        # rate, and the distance the LiDAR emergency box buys you. Raising it
        # is legitimate on a validated route, but it must be a deliberate
        # opt-in rather than something a stray parameter can do quietly.
        speed_ceiling = float(self.get_parameter("speed_limit_ceiling").value)
        if not 0.0 < speed_ceiling <= self.absolute_speed_ceiling:
            raise ValueError(
                "speed_limit_ceiling must be in (0, "
                f"{self.absolute_speed_ceiling}] m/s"
            )
        if not 0.0 < self.v_curve_min <= self.v_max <= speed_ceiling:
            raise ValueError(
                "Require 0 < curve_speed <= max_speed <= "
                f"{speed_ceiling} m/s (raise speed_limit_ceiling to go faster)"
            )
        # Maneuver speed is the lane-change speed. It stays proportionally
        # capped rather than fixed, so raising the ceiling does not silently
        # authorize fast lane changes -- those are a different risk.
        maneuver_ceiling = min(0.30, 0.5 * speed_ceiling)
        if not 0.0 < requested_maneuver_speed <= maneuver_ceiling:
            raise ValueError(
                f"Require 0 < maneuver_speed <= {maneuver_ceiling:.2f} m/s"
            )
        self.v_maneuver = min(requested_maneuver_speed, self.v_max)
        if self.L <= 0.0:
            raise ValueError("wheelbase must be positive")

        self.reverse_speed = -0.10
        self.reverse_done_distance = 0.12

        self.max_steer = float(self.get_parameter("max_steer").value)
        if not 0.05 <= self.max_steer <= 0.58:
            raise ValueError("max_steer must be in [0.05, 0.58] rad")
        self.publish_steering_angle = True

        self.reverse_mode = False
        self.mission_done = False

        # Loop/lap settings. This imitates stopping and restarting
        # the MPC node at the end of each lap, but does it automatically.
        self.loop_path = bool(self.get_parameter("loop_path").value)
        self.target_laps = max(1, int(self.get_parameter("target_laps").value))
        self.completed_laps = 0
        self.loop_reset_idx_margin = 10
        self.loop_end_tolerance_m = 0.12

        # Mission-specific indices from the older 1873-point trajectory are
        # intentionally not applied to arbitrary/new paths.
        self.stop_points = []

        self.stop_active = False
        self.stop_start_time = None
        self.active_stop = None
        self.stop_tolerance_idx = 3
        self.pickup_done = False
        self.pickup_active = False
        self.pickup_start_time = None
        self.pickup_tolerance_idx = 3

        self.overtake_curve_limit = float(
            self.get_parameter("overtake_max_curvature").value
        )
        self.overtake_mean_curve_limit = float(
            self.get_parameter("overtake_mean_curvature").value
        )
        self.solver_timeout_sec = float(
            self.get_parameter("solver_timeout_sec").value
        )
        self.lane_change_distance_m = float(
            self.get_parameter("lane_change_distance_m").value
        )
        if not (
            0.05 <= self.overtake_mean_curve_limit
            <= self.overtake_curve_limit <= 1.0
        ):
            raise ValueError(
                "Require 0.05 <= overtake_mean_curvature <= "
                "overtake_max_curvature <= 1.0"
            )
        if not 0.05 <= self.solver_timeout_sec <= 1.0:
            raise ValueError("solver_timeout_sec must be in [0.05, 1.0] s")
        if not 0.50 <= self.lane_change_distance_m <= 2.0:
            raise ValueError("lane_change_distance_m must be in [0.50, 2.0] m")
        self.was_avoiding = False
        self.return_blend_active = False
        self.return_blend_threshold = 0.02
        self.return_decay = 0.92

        # Curvature above which lane centering was switched off entirely.
        # The default 0.45 is BELOW this route's median curvature of 0.56, so
        # lane centering was inactive on more than half the lap -- one of
        # three reasons it could not affect the car at all. It is retained as
        # a coarse backstop but is no longer the primary gate; confidence is.
        # See lane_confidence_callback.
        self.lane_centering_curve_limit = float(
            self.get_parameter("lane_centering_curve_limit").value
        )

        self.current_max_curvature = 999.0
        self.current_mean_curvature = 999.0
        # Signed; 0.0 (straight) until the first preview runs.
        self.current_signed_curvature = 0.0
        self.current_offset_max_curvature = 999.0
        self.current_offset_mean_curvature = 999.0

        # Lane-centering authority.
        #
        # This was hardcoded at 0.04 m, which cannot do the job it exists
        # for. The lane is 0.42 m wide and the car about 0.19 m, so the car
        # can sit (0.42-0.19)/2 = 0.115 m off centre before a wheel reaches a
        # line. A 0.04 m budget can nudge but can never recover a car that
        # has drifted, nor correct a recorded route that runs off-centre --
        # which is the whole point of centering on the painted lane rather
        # than on the map.
        #
        # The ceiling is therefore the physical margin. Authority is scaled
        # by the detector's confidence between base and ceiling, so a weak or
        # single-boundary observation moves the car very little and a
        # rejected frame moves it not at all. That scaling is only defensible
        # because the bird's-eye fit reports a real residual-based
        # confidence; the previous perspective-frame detector reported
        # valid=True whenever it fitted anything, so a flat small cap was the
        # only safe choice there.
        self.max_lane_offset = float(
            self.get_parameter("max_lane_offset_m").value
        )
        self.min_lane_offset = float(
            self.get_parameter("min_lane_offset_m").value
        )
        self.lane_confidence_floor = float(
            self.get_parameter("lane_confidence_floor").value
        )
        if not 0.0 < self.min_lane_offset <= self.max_lane_offset <= 0.20:
            raise ValueError(
                "require 0 < min_lane_offset_m <= max_lane_offset_m <= 0.20; "
                "0.20 m exceeds the half-lane margin and would steer the car "
                "onto the line"
            )
        # 0.0 until the detector reports; treated as no authority.
        self.lane_confidence = 0.0

        self.w_pos = 200.0
        self.w_yaw = 80.0
        self.w_delta_rate = 35.0
        self.w_speed_rate = 0.5
        self.w_speed_tracking = 90.0
        self.w_control = 3.0
        self.path_spacing = float(self.get_parameter("path_spacing").value)
        if self.path_spacing <= 0.0:
            raise ValueError("path_spacing must be positive")

        # Braking-aware speed limiting: prevents accelerating to v_max
        # on a straight that's too short to brake back down from before
        # the next curve. Tune max_decel to your car's real achievable
        # deceleration (start conservative, e.g. 0.6-1.0 m/s^2 for a
        # small RC-scale car; too low brakes early/conservatively, too
        # high won't catch short straights).
        self.max_decel = float(self.get_parameter("max_decel").value)
        self.brake_lookahead_m = float(
            self.get_parameter("brake_lookahead_m").value
        )
        self.curve_kappa_threshold = float(
            self.get_parameter("curve_kappa_threshold").value
        )
        if self.max_decel <= 0.0 or self.brake_lookahead_m <= 0.0:
            raise ValueError("max_decel and brake_lookahead_m must be positive")

        self.search_window_back = max(
            1, int(self.get_parameter("search_window_back").value)
        )
        self.search_window_forward = max(
            2, int(self.get_parameter("search_window_forward").value)
        )
        advance = self.v_max * self.dt / max(self.path_spacing, 1e-6)
        if self.search_window_forward < 3.0 * advance:
            self.get_logger().warn(
                f"search_window_forward={self.search_window_forward} is tight "
                f"for {advance:.1f} waypoints/step at {self.v_max:.2f} m/s"
            )

        # brake_lookahead_m has to cover the distance needed to shed speed
        # from max_speed down to curve_speed, or the car accelerates onto a
        # straight it cannot slow down from before the next bend. Observed
        # on hardware: at 1.0 m/s the 0.35 m default is just sufficient
        # (0.32 m required) and tracking held at 0.06 m; at 2.0 m/s the
        # requirement is 2.19 m, the car reached a corner 0.4 s after going
        # to full speed, saturated steering at 0.50 rad and tripped the
        # tracking safety stop. Warn rather than refuse -- a route with no
        # tight corners legitimately needs less.
        needed = max(
            0.0,
            (self.v_max ** 2 - self.v_curve_min ** 2) / (2.0 * self.max_decel),
        )
        if self.brake_lookahead_m < needed:
            self.get_logger().error(
                f"brake_lookahead_m={self.brake_lookahead_m:.2f} m is below "
                f"the {needed:.2f} m needed to brake from "
                f"{self.v_max:.2f} to {self.v_curve_min:.2f} m/s at "
                f"{self.max_decel:.2f} m/s^2. The car will enter curves too "
                "fast. Raise brake_lookahead_m or max_decel, or lower "
                "max_speed."
            )

        # Final-approach stopping: decelerate smoothly to a near-stop
        # speed before the hard end-of-path trigger fires, so the car
        # doesn't coast/overshoot into whatever is at the end (e.g. a
        # pickup arm). end_brake_distance is how far before the literal
        # last waypoint to start ramping speed down; end_stop_margin is
        # how much earlier than the very last point counts as "done"
        # (gives slack for tracking lag so the trigger fires before the
        # physical bumper/arm contact point, not after).
        self.end_brake_distance = 1.0
        self.end_stop_margin = 0.05
        self.end_stop_speed = 0.06
        self.end_stop_tolerance_m = 0.08

        

        self.closest_idx = 0
        self.start_alignment_checked = False
        self.global_path_acquisition = True
        self.last_solution = None

        self.prev_delta = 0.0
        self.prev_v = 0.0

        self.motion_enabled = False
        self.drive_state = "DRIVE"
        self.last_drive_state_time = None
        self.last_avoidance_offset_time = None
        self.depth_emergency = False

        self.avoidance_offset = 0.0
        self.avoidance_offset_filtered = 0.0
        self.overtake_transition_start_s = None
        self.return_transition_start_s = None
        self.return_transition_offset = 0.0
        self.return_transition_target = None

        self.lane_offset = 0.0
        self.lane_valid = False
        self.lane_offset_filtered = 0.0
        self.lane_alpha = 0.08

        self.startup_align_steps = 30
        self.startup_counter = 0
        self.startup_v_max = float(self.get_parameter("startup_speed").value)
        if not 0.0 < self.startup_v_max <= self.v_max:
            raise ValueError("Require 0 < startup_speed <= max_speed")
        
        self.localization_ready = False
        self.localization_counter = 0
        self.required_stable_cycles = 40   # 40 * 0.08 = ~3.2 sec

        self.prev_pose = None
        self.max_start_position_error = float(
            self.get_parameter("max_start_position_error").value
        )
        self.max_start_yaw_error = math.radians(float(
            self.get_parameter("max_start_yaw_error_deg").value
        ))
        self.max_tracking_error = float(
            self.get_parameter("max_tracking_error").value
        )
        self.max_tracking_yaw_error = math.radians(float(
            self.get_parameter("max_tracking_yaw_error_deg").value
        ))
        self.require_amcl_quality = bool(
            self.get_parameter("require_amcl_quality").value
        )
        self.max_amcl_position_variance = float(
            self.get_parameter("max_amcl_position_variance").value
        )
        self.max_amcl_yaw_variance = float(
            self.get_parameter("max_amcl_yaw_variance").value
        )
        if (
            self.max_amcl_position_variance <= 0.0
            or self.max_amcl_yaw_variance <= 0.0
        ):
            raise ValueError("AMCL variance limits must be positive")
        self.amcl_covariance = None

        # ---- V2V cooperative layer (ROSbot ahead on the track) ----
        # Fail-safe contract: with no fresh /v2v/* data (receiver not
        # running, link down, ROSbot not localized) every gate below stays
        # inactive and this node behaves exactly as it did without V2V.
        self.v2v_enable = bool(self.get_parameter("enable_v2v").value)
        self.v2v_stale_sec = 1.0    # ignore V2V older than this

        # Discrete CBF: elliptical keep-out around the ROSbot's predicted
        # pose, enforced over the first v2v_ndho stages of the horizon with
        # per-stage slack so the NLP can never become infeasible (a violated
        # barrier costs heavily instead of aborting the solve, which would
        # trigger the exception handler and hard-stop the car).
        self.v2v_ndho = 12          # constrained stages (< N), ~1 s ahead
        self.v2v_gamma = 0.35       # DCBF decay rate, in (0, 1]
        self.v2v_ellipse_a = 0.55   # semi-axis along ROSbot heading [m]
        self.v2v_ellipse_b = 0.40   # semi-axis lateral [m]
        self.v2v_w_slack = 400.0    # quadratic slack weight
        # A safety barrier must not be buyable by the optimizer. Keeping the
        # variable in the NLP preserves its structure, but a zero upper bound
        # turns the V2V DCBF into a hard constraint. Infeasibility falls back
        # to the controller's zero command.
        self.v2v_max_barrier_slack = 0.0
        self.v2v_far = 50.0         # placeholder obstacle distance when idle

        # Speed governor: car-following cap on target_v when the ROSbot is
        # ahead on our reference path. Produces slow -> follow -> stop
        # behavior; overtaking stays with the LiDAR state machine.
        self.v2v_stop_gap = 0.70    # park this far behind a stopped ROSbot
        self.v2v_follow_gap = 1.20  # steady-state following distance [m]
        self.v2v_follow_k = 0.5     # follow-gap proportional gain
        self.v2v_slow_start = 3.0   # governor engages below this gap [m]
        self.v2v_soft_decel = 0.5   # governed braking rate [m/s^2]
        # Below this cap the governor issues a hard stop instead of just
        # lowering target_v — see the note at the enforcement site.
        self.v2v_stop_cap_threshold = 0.04

        self.v2v_alive = False
        self.v2v_last_rx_time = None
        self.v2v_gap = -1.0
        self.v2v_on_path = False
        self.v2v_speed = 0.0
        self.v2v_pred = None        # np.ndarray (M, 3): x, y, yaw
        self.v2v_cap = -1.0         # last governor cap, for logging

        self.tracking_log_path = os.path.abspath(os.path.expanduser(
            self.get_parameter("tracking_log_file").value
        ))
        os.makedirs(os.path.dirname(self.tracking_log_path), exist_ok=True)
        self.tracking_log = open(self.tracking_log_path, "w")
        self.tracking_log.write(
            "time,mode,idx,x,y,yaw,ref_x,ref_y,ref_yaw,"
            "track_error,yaw_error_deg,target_v,v,delta,drive_state,"
            "lane_offset,avoidance_offset,lane_active,mean_curvature,"
            "v2v_active,v2v_gap,v2v_cap\n"
        )
        self.tracking_log.flush()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.allow_overtake_pub = self.create_publisher(Bool, "/allow_overtake", 10)
        # Signed mean curvature of the route ahead. lidar_overtake bends its
        # detection corridor along this so that a curve no longer has to
        # disable the front box wholesale. See overtake_safety.
        self.path_curvature_pub = self.create_publisher(
            Float32,
            "/path_curvature",
            10,
        )
        self.idx_pub = self.create_publisher(Int32, "/current_path_idx", 10)
        self.overtake_yaw_stable_pub = self.create_publisher(
            Bool,
            "/overtake_yaw_stable",
            10,
        )
        self.v2v_relative_pub = self.create_publisher(
            PointStamped,
            "/v2v/relative_geometry",
            10,
        )

        self.sound_pub = self.create_publisher(String, "/qcar2/sound_event", 10)


        self.create_subscription(Bool, "/motion_enable", self.motion_callback, 10)
        self.create_subscription(Float32, "/avoidance_offset", self.avoidance_callback, 10)
        self.create_subscription(String, "/drive_state", self.drive_state_callback, 10)
        self.create_subscription(Float32, "/lane_center_offset", self.lane_offset_callback, 10)
        self.create_subscription(Bool, "/lane_center_valid", self.lane_valid_callback, 10)
        self.create_subscription(
            Float32, "/lane_center_confidence", self.lane_confidence_callback, 10
        )
        self.create_subscription(Bool, "/depth_emergency_stop", self.depth_emergency_callback, 10)
        self.create_subscription(Bool, "/mission_restart", self.mission_restart_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_pose_callback,
            10,
        )

        # V2V inputs (all published locally by v2v_receiver_node).
        self.create_subscription(Bool, "/v2v/alive", self.v2v_alive_callback, 10)
        self.create_subscription(Float32, "/v2v/gap", self.v2v_gap_callback, 10)
        self.create_subscription(Bool, "/v2v/on_path", self.v2v_on_path_callback, 10)
        self.create_subscription(Float32, "/v2v/rosbot_speed", self.v2v_speed_callback, 10)
        self.create_subscription(Path, "/v2v/predicted_path", self.v2v_pred_callback, 10)


        self.trajectory = PathUtils.load_trajectory(self.forward_trajectory_file)
        self.overtake_offset_curvature = offset_path_curvature(
            self.trajectory,
            self.max_reference_offset,
            closed_path=self.loop_path,
        )
        finite_offset_curvature = self.overtake_offset_curvature[
            np.isfinite(self.overtake_offset_curvature)
        ]
        if len(finite_offset_curvature) != len(self.trajectory):
            self.get_logger().error(
                "Maximum-offset path contains degenerate curvature; "
                "overtaking will fail closed near those points"
            )
        else:
            self.get_logger().info(
                f"Maximum-offset (+{self.max_reference_offset:.3f} m) "
                "curvature precomputed: "
                f"min={np.min(finite_offset_curvature):.3f}, "
                f"max={np.max(finite_offset_curvature):.3f} 1/m"
            )

        with open(self.forward_trajectory_file, "rb") as trajectory_stream:
            trajectory_hash = hashlib.sha256(trajectory_stream.read()).hexdigest()[:12]

        measured_spacing = np.linalg.norm(
            np.diff(self.trajectory[:, :2], axis=0), axis=1
        )
        self.path_cumulative_m = np.concatenate(
            ([0.0], np.cumsum(measured_spacing))
        )
        self.path_open_length_m = float(self.path_cumulative_m[-1])
        median_spacing = float(np.median(measured_spacing))
        configured_spacing = self.path_spacing
        if abs(configured_spacing - median_spacing) > 0.25 * median_spacing:
            self.get_logger().warn(
                f"Configured path_spacing={configured_spacing:.4f} m differs "
                f"from trajectory median={median_spacing:.4f} m; using the "
                "measured spacing"
            )
        self.path_spacing = median_spacing
        self.get_logger().info(
            f"Trajectory file={self.forward_trajectory_file}, "
            f"sha256={trajectory_hash}, shape={self.trajectory.shape}, "
            f"median_spacing={self.path_spacing:.4f} m"
        )

        self.path_validation_ok = True
        if self.require_map_validation:
            try:
                report = validate_trajectory_against_map(
                    self.forward_trajectory_file,
                    self.map_file,
                    clearance_m=self.path_clearance_m,
                    max_outside_fraction=0.0,
                    max_occupied_fraction=0.0,
                    max_unknown_fraction=self.max_unknown_fraction,
                    max_clearance_violation_fraction=(
                        self.max_clearance_violation_fraction
                    ),
                    wheelbase_m=self.L,
                    max_steer_rad=self.max_steer,
                    loop_path=self.loop_path,
                )
                self.path_validation_ok = bool(report["ok"])
                log_method = (
                    self.get_logger().info
                    if self.path_validation_ok
                    else self.get_logger().error
                )
                for line in format_report(report).splitlines():
                    log_method(f"Path/map validation: {line}")
            except Exception as error:
                self.path_validation_ok = False
                self.get_logger().error(f"Path/map validation failed: {error}")

        if not self.path_validation_ok and self.allow_unsafe_path:
            self.get_logger().error(
                "UNSAFE OVERRIDE ACTIVE: allow_unsafe_path=true; "
                "use only for a wheels-raised diagnostic"
            )
            self.path_validation_ok = True

        # Check whether the recorded path is clean enough for simple loop reset.
        first_point = self.trajectory[0]
        last_point = self.trajectory[-1]

        loop_gap = math.hypot(
            last_point[0] - first_point[0],
            last_point[1] - first_point[1],
        )

        loop_yaw_gap = abs(
            PathUtils.wrap_angle(last_point[2] - first_point[2])
        )

        self.get_logger().warn(
            f"Loop check: last-to-first distance={loop_gap:.3f} m, "
            f"yaw_diff={math.degrees(loop_yaw_gap):.1f} deg"
        )

        if loop_gap > 0.15:
            self.get_logger().warn(
                "Path may not be a clean loop: last and first points are far apart."
            )

        if loop_yaw_gap > math.radians(25.0):
            self.get_logger().warn(
                "Path yaw difference is high. Loop may still work, but steering may jump slightly."
            )

        # Loop mode is safe only for a genuinely closed, heading-continuous
        # trajectory.  Never invent a diagonal final-to-first connector or
        # rely on a pose/index reset across a large seam.
        if self.loop_path and (
            loop_gap > 0.15 or loop_yaw_gap > math.radians(25.0)
        ):
            self.path_validation_ok = False
            self.get_logger().error(
                "Motion blocked: loop_path=true requires a canonical closed "
                "trajectory (seam <= 0.15 m and yaw mismatch <= 25 deg)."
            )

        if len(self.trajectory) < self.N + 2:
            raise RuntimeError(
                f"Trajectory too short. Need {self.N + 2}, got {len(self.trajectory)}"
            )

        self._setup_optimizer()
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f"QCar MPC ready: loop_path={self.loop_path}, "
            f"target_laps={self.target_laps}, higher normal speed"
        )

    def motion_callback(self, msg):
        self.motion_enabled = bool(msg.data)
        
    def mission_restart_callback(self, msg):
        # Only act on a True pulse, and only if mission is actually done
        if not msg.data:
            return
        if not self.mission_done:
            self.get_logger().warn("Restart requested but mission not done — ignoring.")
            return

        self.get_logger().warn("Manual restart received. Restarting lap loop.")

        self.mission_done = False
        self.completed_laps = 0
        self.closest_idx = 0
        self.start_alignment_checked = False
        self.global_path_acquisition = True

        self.reverse_mode = False
        self.drive_state = "DRIVE"

        self.stop_active = False
        self.stop_start_time = None
        self.active_stop = None
        self.reset_stop_points()

        self.reset_mpc_memory()


    def avoidance_callback(self, msg):
        self.avoidance_offset = float(msg.data)
        self.last_avoidance_offset_time = self.get_clock().now()

    def drive_state_callback(self, msg):
        new_state = str(msg.data)
        if new_state != self.drive_state:
            current_s = float(self.path_cumulative_m[self.closest_idx])
            if new_state == "OVERTAKE_LEFT":
                # Latch one geometric transition origin.  Recomputing the
                # ramp from zero on every receding horizon would make the
                # adjacent lane a target that forever moves away.
                if self.drive_state != "WAIT_FOR_CLEAR":
                    self.overtake_transition_start_s = current_s
                self.return_transition_start_s = None
            elif new_state == "RETURN_RIGHT":
                self.return_transition_start_s = current_s
                self.return_transition_offset = max(
                    abs(self.avoidance_offset_filtered),
                    min(abs(self.avoidance_offset), self.max_reference_offset),
                )
                self.return_transition_target = 0.0
            elif new_state == "DRIVE":
                self.overtake_transition_start_s = None
                self.return_transition_start_s = None
                self.return_transition_target = None
        self.drive_state = new_state
        self.last_drive_state_time = self.get_clock().now()

    def behavior_data_fresh(self):
        """Require a live LiDAR behavior publisher before applying offsets."""
        if (
            self.last_drive_state_time is None
            or self.last_avoidance_offset_time is None
        ):
            return False
        now = self.get_clock().now()
        ages = [
            (now - self.last_drive_state_time).nanoseconds * 1e-9,
            (now - self.last_avoidance_offset_time).nanoseconds * 1e-9,
        ]
        return all(0.0 <= age < self.behavior_timeout_sec for age in ages)

    def lane_offset_callback(self, msg):
        self.lane_offset = float(msg.data)

    def lane_valid_callback(self, msg):
        self.lane_valid = bool(msg.data)

    def lane_confidence_callback(self, msg):
        value = float(msg.data)
        if not math.isfinite(value):
            self.lane_confidence = 0.0
            return
        self.lane_confidence = min(1.0, max(0.0, value))

    def lane_offset_authority(self):
        """How far the camera may move the reference, in metres.

        Scales from min_lane_offset_m to max_lane_offset_m with detector
        confidence, and is zero below lane_confidence_floor. A detector that
        cannot say how sure it is gets the minimum, which is the old
        hardcoded behaviour and the correct fallback.
        """
        if self.lane_confidence < self.lane_confidence_floor:
            return 0.0
        span = self.max_lane_offset - self.min_lane_offset
        return self.min_lane_offset + span * self.lane_confidence

    def depth_emergency_callback(self, msg):
        self.depth_emergency = bool(msg.data)

    def amcl_pose_callback(self, msg):
        covariance = msg.pose.covariance
        if len(covariance) >= 36:
            self.amcl_covariance = (
                float(covariance[0]),
                float(covariance[7]),
                float(covariance[35]),
            )

    def amcl_quality_ok(self):
        if not self.require_amcl_quality:
            return True
        if self.amcl_covariance is None:
            return False
        variance_x, variance_y, variance_yaw = self.amcl_covariance
        return (
            np.all(np.isfinite(self.amcl_covariance))
            and max(variance_x, variance_y) <= self.max_amcl_position_variance
            and variance_yaw <= self.max_amcl_yaw_variance
            and min(variance_x, variance_y, variance_yaw) >= 0.0
        )

    def v2v_alive_callback(self, msg):
        self.v2v_alive = bool(msg.data)
        self.v2v_last_rx_time = self.get_clock().now()

    def v2v_gap_callback(self, msg):
        self.v2v_gap = float(msg.data)

    def v2v_on_path_callback(self, msg):
        self.v2v_on_path = bool(msg.data)

    def v2v_speed_callback(self, msg):
        self.v2v_speed = float(msg.data)

    def v2v_pred_callback(self, msg):
        pts = []
        for ps in msg.poses:
            q = ps.pose.orientation
            pts.append([
                ps.pose.position.x,
                ps.pose.position.y,
                self.quat_to_yaw(q.x, q.y, q.z, q.w),
            ])
        self.v2v_pred = np.array(pts, dtype=float) if pts else None

    def v2v_data_fresh(self):
        """True only when the receiver reported fresh, localized ROSbot data
        recently. Every V2V behavior gates on this — stale data can never
        keep a constraint or a speed cap active."""
        if not self.v2v_enable or self.v2v_last_rx_time is None:
            return False
        age = (
            self.get_clock().now() - self.v2v_last_rx_time
        ).nanoseconds * 1e-9
        return self.v2v_alive and age < self.v2v_stale_sec

    def build_v2v_obstacle_params(self):
        """Per-stage obstacle poses for the DCBF parameter block.

        With fresh data: the ROSbot's predicted pose at each MPC stage
        (clamped to its last prediction). Without: a fixed point v2v_far
        meters away, which keeps the barrier trivially satisfied so the
        solve is bit-identical in effect to the pre-V2V controller.
        """
        n = self.v2v_ndho + 1
        if (
            self.v2v_data_fresh()
            and self.v2v_pred is not None
            and len(self.v2v_pred) > 0
        ):
            pred = self.v2v_pred
            rows = [pred[min(k, len(pred) - 1)] for k in range(n)]
            return np.array(rows, dtype=float).flatten()

        far = np.zeros((n, 3), dtype=float)
        far[:, 0] = self.v2v_far
        far[:, 1] = self.v2v_far
        return far.flatten()

    def publish_v2v_relative_geometry(self, pose):
        """Publish the live QCar position in the ROSbot heading frame.

        Point.x is longitudinal separation, point.y lateral separation, and
        point.z the dimensionless safety-ellipse metric.  Keeping all three
        in one stamped standard message lets LiDAR associate the known V2V
        vehicle with its scan return without weakening unknown-obstacle
        handling.
        """
        if (
            not self.v2v_data_fresh()
            or self.v2v_pred is None
            or len(self.v2v_pred) == 0
        ):
            return
        ox, oy, oyaw = np.asarray(self.v2v_pred[0], dtype=float)
        if not np.all(np.isfinite([*pose, ox, oy, oyaw])):
            return
        dx = float(pose[0] - ox)
        dy = float(pose[1] - oy)
        longitudinal = math.cos(oyaw) * dx + math.sin(oyaw) * dy
        lateral = -math.sin(oyaw) * dx + math.cos(oyaw) * dy
        metric = (
            (longitudinal / self.v2v_ellipse_a) ** 2
            + (lateral / self.v2v_ellipse_b) ** 2
        )

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rosbot_predicted"
        msg.point.x = longitudinal
        msg.point.y = lateral
        msg.point.z = metric
        self.v2v_relative_pub.publish(msg)

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        self.prev_v = 0.0
        self.prev_delta = 0.0

    def reset_mpc_memory(self):
        self.last_solution = np.zeros(self.nx + self.nu + self.ns)
        self.avoidance_offset_filtered = 0.0
        self.lane_offset_filtered = 0.0
        self.prev_v = 0.0
        self.prev_delta = 0.0
        self.startup_counter = 0

    def reset_stop_points(self):
        for stop_point in self.stop_points:
            stop_point["done"] = False

        self.pickup_done = False
        self.pickup_active = False
        self.pickup_start_time = None

    def switch_to_reverse(self):
        self.get_logger().warn("Forward complete. Switching to reverse trajectory.")

        try:
            reverse_trajectory = PathUtils.load_trajectory(
                self.reverse_trajectory_file
            )
            if self.require_map_validation:
                report = validate_trajectory_against_map(
                    self.reverse_trajectory_file,
                    self.map_file,
                    clearance_m=self.path_clearance_m,
                    max_outside_fraction=0.0,
                    max_occupied_fraction=0.0,
                    max_unknown_fraction=self.max_unknown_fraction,
                    max_clearance_violation_fraction=(
                        self.max_clearance_violation_fraction
                    ),
                    wheelbase_m=self.L,
                    max_steer_rad=self.max_steer,
                    loop_path=False,
                )
                for line in format_report(report).splitlines():
                    log_method = (
                        self.get_logger().info
                        if report["ok"]
                        else self.get_logger().error
                    )
                    log_method(f"Reverse path/map validation: {line}")
                if not report["ok"] and not self.allow_unsafe_path:
                    raise RuntimeError("reverse trajectory/map validation failed")
        except Exception as error:
            self.path_validation_ok = False
            self.mission_done = True
            self.stop()
            self.get_logger().error(
                f"Reverse motion blocked: {error}"
            )
            return

        self.trajectory = reverse_trajectory
        self.path_spacing = float(np.median(np.linalg.norm(
            np.diff(self.trajectory[:, :2], axis=0), axis=1
        )))

        if len(self.trajectory) < self.N + 2:
            raise RuntimeError(
                f"Reverse trajectory too short. Need {self.N + 2}, got {len(self.trajectory)}"
            )

        self.closest_idx = 0
        self.start_alignment_checked = False
        self.global_path_acquisition = False
        self.reverse_mode = True
        self.drive_state = "DRIVE"

        self.reset_mpc_memory()

    def publish_overtake_disabled(self):
        msg = Bool()
        msg.data = False
        self.allow_overtake_pub.publish(msg)

        # Keep /path_curvature fresh, or lidar_overtake's staleness watchdog
        # falls back to a straight corridor while reversing.
        curvature_msg = Float32()
        curvature_msg.data = 0.0
        self.path_curvature_pub.publish(curvature_msg)

    def handle_pickup_stop(self):
        if self.reverse_mode:
            return False

        if self.stop_active:
            elapsed = (
                self.get_clock().now() - self.stop_start_time
            ).nanoseconds * 1e-9

            self.stop()

            if elapsed >= self.active_stop["wait_sec"]:
                resume_sound = self.active_stop.get("resume_sound", "resume")
                if resume_sound:
                    self.sound_pub.publish(String(data=resume_sound))

                self.stop_active = False
                self.stop_start_time = None
                self.active_stop = None

                self.reset_mpc_memory()
                self.get_logger().warn("Timed stop complete. Continuing path.")

            return True

        for stop_point in self.stop_points:
            if stop_point["done"]:
                continue

            if abs(self.closest_idx - stop_point["idx"]) <= self.stop_tolerance_idx:
                self.stop_active = True
                self.active_stop = stop_point
                self.stop_start_time = self.get_clock().now()
                stop_point["done"] = True

                self.stop()

                start_sound = stop_point.get("start_sound", "pickup")
                if start_sound:
                    self.sound_pub.publish(String(data=start_sound))

                self.get_logger().warn(
                    f"Timed stop started at idx={self.closest_idx}, "
                    f"target_idx={stop_point['idx']}, "
                    f"wait={stop_point['wait_sec']} sec"
                )

                return True

        return False
    
    def localization_stable(self, pose):

        if self.prev_pose is None:
            self.prev_pose = pose.copy()
            return False

        dx = pose[0] - self.prev_pose[0]
        dy = pose[1] - self.prev_pose[1]

        dpos = math.hypot(dx, dy)

        dyaw = abs(
            PathUtils.wrap_angle(
                pose[2] - self.prev_pose[2]
            )
        )

        self.prev_pose = pose.copy()

        if dpos < 0.005 and dyaw < math.radians(0.5):
            self.localization_counter += 1
        else:
            self.localization_counter = 0

        return self.localization_counter > self.required_stable_cycles

    def log_tracking_error(
        self,
        pose,
        closest_point,
        tracking_error,
        yaw_error,
        target_v,
        v,
        delta,
    ):
        if not hasattr(self, "tracking_log"):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        mode = "REVERSE" if self.reverse_mode else "FORWARD"
        lane_active = (
            self.current_mean_curvature < self.lane_centering_curve_limit
            and self.lane_valid
            and self.drive_state == "DRIVE"
        )

        self.tracking_log.write(
            f"{now:.3f},"
            f"{mode},"
            f"{self.closest_idx},"
            f"{pose[0]:.4f},"
            f"{pose[1]:.4f},"
            f"{pose[2]:.4f},"
            f"{closest_point[0]:.4f},"
            f"{closest_point[1]:.4f},"
            f"{closest_point[2]:.4f},"
            f"{tracking_error:.4f},"
            f"{math.degrees(yaw_error):.2f},"
            f"{target_v:.4f},"
            f"{v:.4f},"
            f"{delta:.4f},"
            f"{self.drive_state},"
            f"{self.lane_offset_filtered:.4f},"
            f"{self.avoidance_offset_filtered:.4f},"
            f"{lane_active},"
            f"{self.current_mean_curvature:.4f},"
            f"{int(self.v2v_data_fresh())},"
            f"{self.v2v_gap:.3f},"
            f"{self.v2v_cap:.3f}\n"
        )

        self.tracking_log.flush()

    def _get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
            )

            tr = t.transform.translation
            q = t.transform.rotation
            yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)

            return np.array([tr.x, tr.y, yaw], dtype=float)

        except TransformException as e:
            self.get_logger().warn(f"TF unavailable: {e}")
            return None

    def _setup_optimizer(self):
        X = ca.SX.sym("X", 3, self.N + 1)
        U = ca.SX.sym("U", 2, self.N)

        # Parameter layout: [x0(3), prev_delta, prev_v, target_v,
        #                    ref(3N), v2v_obstacle(3*(ndho+1))]
        P = ca.SX.sym(
            "P",
            3 + 2 + 1 + self.N * 3 + 3 * (self.v2v_ndho + 1),
        )

        cost = 0
        g = []

        g.append(X[:, 0] - P[0:3])

        prev_delta = P[3]
        prev_v = P[4]
        target_v = P[5]

        for k in range(self.N):
            st = X[:, k]
            delta_k = U[0, k]
            v_k = U[1, k]

            idx = 6 + 3 * k

            x_ref = P[idx]
            y_ref = P[idx + 1]
            yaw_ref = P[idx + 2]

            pos_err = (
                (st[0] - x_ref) ** 2
                + (st[1] - y_ref) ** 2
            )

            cost += self.w_pos * ca.log(1.0 + pos_err)

            yaw_err = ca.atan2(
                ca.sin(st[2] - yaw_ref),
                ca.cos(st[2] - yaw_ref),
            )

            cost += self.w_yaw * yaw_err ** 2

            if k == 0:
                ddelta = delta_k - prev_delta
                dv = v_k - prev_v
            else:
                ddelta = delta_k - U[0, k - 1]
                dv = v_k - U[1, k - 1]

            cost += self.w_delta_rate * ddelta ** 2
            cost += self.w_speed_rate * dv ** 2
            cost += self.w_speed_tracking * (v_k - target_v) ** 2
            cost += self.w_control * delta_k ** 2

            st_next = ca.vertcat(
                st[0] + self.dt * v_k * ca.cos(st[2]),
                st[1] + self.dt * v_k * ca.sin(st[2]),
                st[2] + self.dt * (v_k / self.L) * ca.tan(delta_k),
            )

            g.append(X[:, k + 1] - st_next)

        # ---- V2V discrete control barrier function ----
        # Elliptical keep-out around the ROSbot's predicted pose, rotated to
        # its heading, enforced as h_{k+1} >= (1-gamma) h_k over the first
        # v2v_ndho stages. Per-stage slack S_k >= 0 keeps the NLP feasible
        # under any data; the slack cost makes violation expensive.
        S = ca.SX.sym("S", self.v2v_ndho)
        obs_base = 6 + 3 * self.N

        def v2v_barrier(k):
            ox = P[obs_base + 3 * k]
            oy = P[obs_base + 3 * k + 1]
            oyaw = P[obs_base + 3 * k + 2]
            dx = X[0, k] - ox
            dy = X[1, k] - oy
            lon = ca.cos(oyaw) * dx + ca.sin(oyaw) * dy
            lat = -ca.sin(oyaw) * dx + ca.cos(oyaw) * dy
            return (
                (lon / self.v2v_ellipse_a) ** 2
                + (lat / self.v2v_ellipse_b) ** 2
                - 1.0
            )

        for k in range(self.v2v_ndho):
            h_k = v2v_barrier(k)
            h_next = v2v_barrier(k + 1)
            g.append(h_next - (1.0 - self.v2v_gamma) * h_k + S[k])
            cost += self.v2v_w_slack * S[k] ** 2 + 10.0 * S[k]

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(S, -1, 1),
        )

        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": ca.vertcat(*g),
            "p": P,
        }

        self.solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                # A controller that spends seconds searching for a solution
                # is no longer a controller.  The command bridge stops after
                # 0.5 s without a fresh command; these limits guarantee that
                # this callback returns well before that watchdog expires.
                "ipopt.max_iter": 40,
                "ipopt.max_cpu_time": self.solver_timeout_sec,
                "ipopt.max_wall_time": self.solver_timeout_sec,
                "ipopt.tol": 1e-3,
            },
        )

        self.nx = 3 * (self.N + 1)
        self.nu = 2 * self.N
        self.ns = self.v2v_ndho

        self.lbx = []
        self.ubx = []

        for _ in range(self.N + 1):
            self.lbx += [-ca.inf, -ca.inf, -ca.inf]
            self.ubx += [ca.inf, ca.inf, ca.inf]

        for _ in range(self.N):
            self.lbx += [-self.max_steer, self.v_min]
            self.ubx += [self.max_steer, self.v_max]

        # V2V barrier variables are hard-bounded. A future configurable
        # relaxation may use a small audited value, but never infinity.
        self.lbx += [0.0] * self.ns
        self.ubx += [self.v2v_max_barrier_slack] * self.ns

        # Equality constraints (dynamics), then V2V barrier inequalities.
        self.lbg = [0.0] * (3 * (self.N + 1)) + [0.0] * self.ns
        self.ubg = [0.0] * (3 * (self.N + 1)) + [ca.inf] * self.ns

        self.last_solution = np.zeros(self.nx + self.nu + self.ns)

    def shift(self, sol):
        X = sol[:self.nx].reshape(self.N + 1, 3)
        U = sol[self.nx:self.nx + self.nu].reshape(self.N, 2)
        S = sol[self.nx + self.nu:]

        X = np.vstack([X[1:], X[-1]])
        U = np.vstack([U[1:], U[-1]])
        if len(S) > 1:
            S = np.concatenate([S[1:], S[-1:]])

        return np.concatenate([X.flatten(), U.flatten(), S])

    def publish_overtake_permission(self):
        if self.reverse_mode:
            self.publish_overtake_disabled()
            return False, 0.0, 0.0

        allow_overtake, metrics = overtake_curvature_preview(
            self.trajectory[:, 3],
            self.overtake_offset_curvature,
            self.closest_idx,
            self.N,
            self.loop_path,
            self.overtake_curve_limit,
            self.overtake_mean_curve_limit,
        )
        self.current_max_curvature = metrics["nominal_max"]
        self.current_mean_curvature = metrics["nominal_mean"]
        self.current_offset_max_curvature = metrics["offset_max"]
        self.current_offset_mean_curvature = metrics["offset_mean"]

        # Return conservative combined values so existing diagnostics reflect
        # whichever lane geometry actually blocked the maneuver.
        max_curvature = max(
            self.current_max_curvature,
            self.current_offset_max_curvature,
        )
        mean_curvature = max(
            self.current_mean_curvature,
            self.current_offset_mean_curvature,
        )

        msg = Bool()
        msg.data = bool(allow_overtake)
        self.allow_overtake_pub.publish(msg)

        return allow_overtake, max_curvature, mean_curvature

    def publish_path_curvature(self):
        """Publish the signed route curvature, unconditionally.

        This deliberately does NOT live in publish_overtake_permission or
        anywhere else downstream of the control loop's gates. Everything
        there is reachable only once the car is validated, localized,
        aligned and about to drive, and the corridor is needed before all of
        that: lidar_overtake bends its detection boxes with this topic, so a
        silent absence means curve detection degrades to the straight boxes
        exactly while the car is starting up. It is also the pre-flight
        check the RUNBOOK tells you to run, which has to work on a
        stationary car.

        Depends only on the loaded route and the last known index, both of
        which exist from __init__ onwards.
        """
        try:
            self.current_signed_curvature = signed_curvature_preview(
                self.trajectory[:, 3],
                self.closest_idx,
                self.N,
                self.loop_path,
            )
        except (AttributeError, IndexError, TypeError):
            # Route not loaded yet. Straight is the safe report: it is what
            # the corridor falls back to anyway.
            self.current_signed_curvature = 0.0

        curvature_msg = Float32()
        curvature_msg.data = float(self.current_signed_curvature)
        self.path_curvature_pub.publish(curvature_msg)

    def _progress_since(self, start_s):
        if start_s is None:
            return 0.0
        current_s = float(self.path_cumulative_m[self.closest_idx])
        progress = current_s - float(start_s)
        if self.loop_path and progress < 0.0:
            progress += self.path_open_length_m
        return max(0.0, progress)

    def _profiled_reference(self, ref, start_s, start_offset, end_offset):
        """Build a distance-based S-curve between two lane offsets."""
        points = np.asarray(ref, dtype=float).reshape(-1, 3)
        stage_arc = np.zeros(len(points), dtype=float)
        if len(points) > 1:
            stage_arc[1:] = np.cumsum(
                np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
            )
        phase = np.clip(
            (self._progress_since(start_s) + stage_arc)
            / self.lane_change_distance_m,
            0.0,
            1.0,
        )
        blend = phase * phase * (3.0 - 2.0 * phase)
        offsets = start_offset + (end_offset - start_offset) * blend
        return PathUtils.apply_offset_profile(ref, offsets), float(offsets[0])

    def apply_state_machine_reference(self, ref, target_v):
        if self.reverse_mode:
            return ref, self.reverse_speed, False

        if (
            not self.motion_enabled
            or self.depth_emergency
            or self.drive_state in [
                "EMERGENCY_STOP",
                "WAIT_FOR_CLEAR",
                "WAIT_FOR_LEFT_CLEAR",
                "STARTUP_WAIT",
                "LIDAR_TIMEOUT",
                "SENSOR_WAIT",
            ]
            or self.avoidance_offset >= 999.0
            or not np.isfinite(self.avoidance_offset)
        ):
            return None, 0.0, True

        if not self.enable_reference_offsets:
            if self.drive_state != "DRIVE":
                self.get_logger().warn(
                    f"Behavior state {self.drive_state!r} received while "
                    "enable_reference_offsets=false; stopping",
                    throttle_duration_sec=1.0,
                )
                return None, 0.0, True
            return ref, target_v, False

        if self.drive_state in ["OBSTACLE_SLOW", "OVERTAKE_LEFT", "RETURN_RIGHT"]:

            if self.drive_state == "OBSTACLE_SLOW":
                self.avoidance_offset_filtered = 0.0
                self.lane_offset_filtered = 0.0

                # A behavior state may lower speed, but must never undo an
                # end-of-path, curvature, or V2V speed limit.
                target_v = min(target_v, self.v_max)

                return ref, target_v, False

            # Active avoidance.  Overtake and return use a measured-distance
            # S-curve with a matching yaw profile.  The older parallel jump
            # kept the nominal yaw and moved only 0.13 m sideways before the
            # safety envelope was reached in simulation.
            self.was_avoiding = True
            self.return_blend_active = True

            if self.drive_state == "OVERTAKE_LEFT":
                requested_offset = float(np.clip(
                    self.avoidance_offset,
                    -self.max_reference_offset,
                    self.max_reference_offset,
                ))
                if self.overtake_transition_start_s is None:
                    self.overtake_transition_start_s = float(
                        self.path_cumulative_m[self.closest_idx]
                    )
                ref, current_offset = self._profiled_reference(
                    ref,
                    self.overtake_transition_start_s,
                    0.0,
                    requested_offset,
                )
                # Log the offset applied at the vehicle, not merely the final
                # adjacent-lane target at the end of the S-curve.
                self.avoidance_offset_filtered = current_offset
            else:
                # RETURN_RIGHT may temporarily command the passing-lane
                # offset when LiDAR sees the original lane become blocked.
                # Track that command instead of blindly continuing toward
                # zero; when it changes, start a new measured-distance
                # S-curve from the offset currently applied at the vehicle.
                requested_offset = float(np.clip(
                    self.avoidance_offset,
                    -self.max_reference_offset,
                    self.max_reference_offset,
                ))
                if abs(requested_offset) < self.return_blend_threshold:
                    requested_offset = 0.0

                target_changed = (
                    self.return_transition_target is None
                    or abs(
                        requested_offset - self.return_transition_target
                    ) > 1e-3
                )
                if self.return_transition_start_s is None or target_changed:
                    self.return_transition_start_s = float(
                        self.path_cumulative_m[self.closest_idx]
                    )
                    self.return_transition_offset = float(
                        self.avoidance_offset_filtered
                    )
                    self.return_transition_target = requested_offset
                ref, current_offset = self._profiled_reference(
                    ref,
                    self.return_transition_start_s,
                    self.return_transition_offset,
                    self.return_transition_target,
                )
                self.avoidance_offset_filtered = current_offset

            self.lane_offset_filtered = 0.0

            # Give the Ackermann model enough travelled time to settle on the
            # adjacent-lane centre before it becomes longitudinally level
            # with the lead. At full straight-line speed the measured offset
            # lagged the +0.47 m reference by 6 cm and correctly tripped the
            # hard safety ellipse during the pass.
            target_v = min(target_v, self.v_maneuver)

            return ref, target_v, False

        if self.drive_state == "DRIVE":

            # Smooth return after avoidance, even if LiDAR state jumps
            # directly from OVERTAKE_LEFT to DRIVE.
            if self.return_blend_active:
                self.avoidance_offset_filtered *= self.return_decay

                if abs(self.avoidance_offset_filtered) > self.return_blend_threshold:
                    ref = PathUtils.apply_offset(
                     ref,
                     self.N,
                     self.avoidance_offset_filtered,
                    )

                    self.lane_offset_filtered = 0.0

                    target_v = min(target_v, self.v_max)

                    return ref, target_v, False

                self.avoidance_offset_filtered = 0.0
                self.return_blend_active = False
                self.was_avoiding = False

            else:
                self.avoidance_offset_filtered = 0.0

            # Coarse backstop only. The real gate is detector confidence,
            # applied through lane_offset_authority(): a curve degrades the
            # observation, and the detector says so, so gating twice on
            # curvature just switched centering off across most of the lap.
            is_straight = (
                self.current_mean_curvature
                < self.lane_centering_curve_limit
            )
            authority = self.lane_offset_authority()

            if is_straight and self.lane_valid and authority > 0.0:
                self.lane_offset_filtered = (
                    (1.0 - self.lane_alpha)
                    * self.lane_offset_filtered
                    + self.lane_alpha
                    * self.lane_offset
                )

                self.lane_offset_filtered = float(
                    np.clip(
                        self.lane_offset_filtered,
                        -authority,
                        authority,
                    )
                )

                if abs(self.lane_offset_filtered) < 0.005:
                    self.lane_offset_filtered = 0.0

                if abs(self.lane_offset_filtered) > 0.005:
                    ref = PathUtils.apply_offset(
                        ref,
                        self.N,
                        self.lane_offset_filtered,
                    )
            else:
                self.lane_offset_filtered *= 0.90

            return ref, target_v, False

        return None, 0.0, True
    
    def _control_loop(self):
        # Before every gate below. See publish_path_curvature for why.
        self.publish_path_curvature()

        if self.mission_done:
            self.stop()
            return

        if not self.path_validation_ok:
            self.stop()
            self.get_logger().error(
                "Motion blocked: trajectory/map validation failed. "
                "Record/choose a trajectory made on the active map.",
                throttle_duration_sec=2.0,
            )
            return

        if self.motion_enabled and not self.behavior_data_fresh():
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().error(
                "Motion blocked: LiDAR behavior heartbeat is missing or stale",
                throttle_duration_sec=1.0,
            )
            return

        if self.depth_emergency:
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().warn(
                "Immediate stop: depth emergency",
                throttle_duration_sec=0.5,
            )
            return

        pose = self._get_pose()

        if pose is None:
            self.stop()
            return

        self.publish_v2v_relative_geometry(pose)

        if not self.amcl_quality_ok():
            self.stop()
            self.localization_ready = False
            self.localization_counter = 0
            self.prev_pose = None
            self.start_alignment_checked = False
            covariance_text = (
                "not received"
                if self.amcl_covariance is None
                else (
                    f"x={self.amcl_covariance[0]:.4f}, "
                    f"y={self.amcl_covariance[1]:.4f}, "
                    f"yaw={self.amcl_covariance[2]:.4f}"
                )
            )
            self.get_logger().warn(
                "Motion blocked: AMCL covariance is unavailable or above "
                f"limits (position={self.max_amcl_position_variance:.3f}, "
                f"yaw={self.max_amcl_yaw_variance:.3f}); {covariance_text}",
                throttle_duration_sec=1.0,
            )
            return
        
        if not self.localization_ready:

            if self.localization_stable(pose):
                self.localization_ready = True
                self.get_logger().info(
                    "Localization transform is stable; checking path alignment. "
                    "Motion still requires /motion_enable=true."
                )
            else:
                self.stop()
                return

        # While a lateral offset is commanded -- an overtake, a return, a
        # lane-centering correction -- the car is deliberately NOT on the
        # recorded route, so matching the raw pose against it is wrong twice
        # over. The measured error includes the offset even when tracking is
        # perfect, and on this self-crossing route (branches 0.02-0.26 m
        # apart) a 0.30 m offset can put the car nearer a PARALLEL branch
        # than its own, so the index search follows it there. Observed as
        # "Tracking safety stop: position_error=1.35 m, yaw_error=3.2 deg"
        # in the middle of an otherwise healthy OVERTAKE_LEFT.
        #
        # So search from where the car would be with the offset removed:
        # slide it back across its own heading by the commanded amount. The
        # lateral unit vector for heading psi is (-sin psi, cos psi).
        commanded_offset = (
            self.avoidance_offset_filtered + self.lane_offset_filtered
        )
        if not math.isfinite(commanded_offset):
            commanded_offset = 0.0

        search_pose = (
            pose[0] + commanded_offset * math.sin(pose[2]),
            pose[1] - commanded_offset * math.cos(pose[2]),
            pose[2],
        )

        self.closest_idx = PathUtils.closest_point(
            search_pose,
            self.trajectory,
            self.closest_idx,
            global_search=self.global_path_acquisition,
            window_back=self.search_window_back,
            window_forward=self.search_window_forward,
        )

        if self.handle_pickup_stop():
            return

        closest_point = self.trajectory[self.closest_idx]

        # Measure the error against the reference the car was actually asked
        # to follow, not the unoffset route.
        tracking_error = math.hypot(
            search_pose[0] - closest_point[0],
            search_pose[1] - closest_point[1],
        )

        yaw_error = PathUtils.wrap_angle(
            pose[2] - closest_point[2]
        )

        if not self.start_alignment_checked:
            if (
                tracking_error > self.max_start_position_error
                or abs(yaw_error) > self.max_start_yaw_error
            ):
                self.stop()
                self.get_logger().error(
                    "Motion blocked: start pose does not match trajectory. "
                    f"nearest_idx={self.closest_idx}, "
                    f"position_error={tracking_error:.3f} m "
                    f"(limit {self.max_start_position_error:.3f}), "
                    f"yaw_error={math.degrees(yaw_error):.1f} deg "
                    f"(limit {math.degrees(self.max_start_yaw_error):.1f}). "
                    "Set AMCL from the selected trajectory waypoint.",
                    throttle_duration_sec=1.0,
                )
                return

            self.start_alignment_checked = True
            self.global_path_acquisition = False
            self.get_logger().info(
                f"Start alignment accepted at idx={self.closest_idx}: "
                f"position_error={tracking_error:.3f} m, "
                f"yaw_error={math.degrees(yaw_error):.1f} deg"
            )

        if (
            tracking_error > self.max_tracking_error
            or abs(yaw_error) > self.max_tracking_yaw_error
        ):
            self.stop()
            self.reset_mpc_memory()
            self.get_logger().error(
                "Tracking safety stop: "
                f"position_error={tracking_error:.3f} m, "
                f"yaw_error={math.degrees(yaw_error):.1f} deg",
                throttle_duration_sec=0.5,
            )
            return

        end_point = self.trajectory[-1]
        distance_to_end_physical = math.hypot(
            pose[0] - end_point[0],
            pose[1] - end_point[1],
        )

        if (
            not self.reverse_mode
            and self.closest_idx >= len(self.trajectory) - self.loop_reset_idx_margin
            and distance_to_end_physical <= self.loop_end_tolerance_m
        ):
            if self.loop_path:
                self.completed_laps += 1

                self.get_logger().warn(
                    f"Lap {self.completed_laps}/{self.target_laps} complete."
                )

                if self.completed_laps >= self.target_laps:
                    self.get_logger().warn(
                        "Target laps complete. Stopping. Press motion_enable to restart."
                    )
                    self.mission_done = True
                    self.stop()
                    return

                # Soft MPC restart at the loop boundary.
                # This imitates stopping/restarting the MPC node.
                self.closest_idx = 0
                # This is a validated, heading-continuous seam, not a new
                # localization acquisition. Requiring the nominal start pose
                # again can deadlock a legitimate RETURN_RIGHT that crosses
                # the seam with a temporary lane offset.
                self.start_alignment_checked = True
                self.global_path_acquisition = False
                self.reset_mpc_memory()
                self.reset_stop_points()

                return

            if self.enable_reverse:
                self.switch_to_reverse()
                return

            self.get_logger().warn(
                "Forward trajectory complete. Reverse disabled. Stopping."
            )
            self.mission_done = True
            self.stop()
            return

        if (
            self.reverse_mode
            and self.closest_idx >= len(self.trajectory) - 5
            and distance_to_end_physical <= self.end_stop_tolerance_m
        ):
            self.get_logger().warn("Reverse trajectory complete. Stopping.")
            self.mission_done = True
            self.stop()
            return

        allow_overtake, max_curvature, mean_curvature = (
            self.publish_overtake_permission()
        )

        if self.reverse_mode:
            target_v = self.reverse_speed
        else:
            target_v = PathUtils.target_speed(
                self.trajectory,
                self.closest_idx,
                self.N,
                self.v_max,
                self.v_curve_min,
                dt=self.dt,
                spacing=self.path_spacing,
                lookahead_speed=self.v_max,
                max_decel=self.max_decel,
                brake_lookahead_m=self.brake_lookahead_m,
                curve_kappa_threshold=self.curve_kappa_threshold,
            )

        startup_active = (
            not self.reverse_mode
            and self.startup_counter < self.startup_align_steps
        )

        if startup_active:
            target_v = min(target_v, self.startup_v_max)
            self.startup_counter += 1

        if not self.reverse_mode and not self.loop_path:
            # Remaining progress must be measured along the route. This
            # trajectory is geometrically closed even when a one-shot mission
            # uses loop_path=false, so Euclidean distance to its final point
            # is only ~5 cm at startup and used to trigger false end braking.
            distance_to_end = max(
                0.0,
                self.path_open_length_m
                - float(self.path_cumulative_m[self.closest_idx]),
            )

            if distance_to_end < self.end_brake_distance:
                v_end_limited = math.sqrt(
                    max(
                        2.0 * self.max_decel * max(
                            distance_to_end - self.end_stop_margin,
                            0.0,
                        ),
                        0.0,
                    )
                )
                v_end_limited = max(v_end_limited, self.end_stop_speed)
                target_v = min(target_v, v_end_limited)

        # ---- V2V speed governor (slow / follow / stop behind the ROSbot).
        # Only ever LOWERS target_v, only in normal forward driving, and only
        # on fresh data. Overtaking decisions stay with the LiDAR state
        # machine (assisted by V2V early warning in lidar_overtake_node).
        self.v2v_cap = -1.0
        if (
            self.v2v_data_fresh()
            and not self.reverse_mode
            and self.drive_state == "DRIVE"
            and self.v2v_on_path
            and 0.0 <= self.v2v_gap < self.v2v_slow_start
        ):
            gap = self.v2v_gap
            self.v2v_cap = PathUtils.v2v_following_cap(
                gap=gap,
                lead_speed=self.v2v_speed,
                stop_gap=self.v2v_stop_gap,
                follow_gap=self.v2v_follow_gap,
                follow_gain=self.v2v_follow_k,
                soft_decel=self.v2v_soft_decel,
            )
            target_v = min(target_v, self.v2v_cap)

            # A near-zero cap means STOP, and it must be enforced directly.
            # Lowering target_v alone does NOT stop the car: build_reference
            # with target_v=0 collapses the whole horizon onto the single
            # closest waypoint, and if the car is off-track the position cost
            # (w_pos*log(1+e), ~30 at 0.4 m) dwarfs the speed cost
            # (w_speed_tracking*(v-0)^2, ~2 at 0.15 m/s), so the solver
            # sprints to that point at v_max. Observed in sim: gap 0.05 m,
            # target_v 0.00, commanded v 0.15 -> it drove into the ROSbot.
            if (
                gap <= self.v2v_stop_gap
                or self.v2v_cap <= self.v2v_stop_cap_threshold
            ):
                self.stop()
                self.get_logger().warn(
                    f"V2V hold: {self.v2v_gap:.2f} m to "
                    f"{'stopped' if self.v2v_speed < 0.05 else 'slower'} "
                    "vehicle ahead",
                    throttle_duration_sec=1.0,
                )
                return

        ref = PathUtils.build_reference(
            self.trajectory,
            self.closest_idx,
            self.N,
            self.dt,
            self.path_spacing,
            abs(target_v),
            loop_path=(self.loop_path and not self.reverse_mode),
        )

        ref, target_v, should_stop = self.apply_state_machine_reference(
            ref,
            target_v,
        )

        if should_stop:
            self.stop()
            if self.drive_state not in [
                "WAIT_FOR_CLEAR",
                "EMERGENCY_STOP",
                "LIDAR_TIMEOUT",
                "SENSOR_WAIT",
                "STARTUP_WAIT",
            ]:
                self.reset_mpc_memory()
            return

        # Zero target means a hard hold.  It is never left to the optimizer's
        # soft speed cost, whose path-error term could otherwise command
        # motion while trying to return to the reference.
        if not self.reverse_mode and target_v <= 1e-6:
            self.stop()
            return

        active_v2v_cap = self.v2v_cap if self.v2v_cap >= 0.0 else None
        speed_ceiling = self.v_max
        if not self.reverse_mode:
            speed_ceiling = PathUtils.enforce_forward_speed_cap(
                command_v=self.v_max,
                target_v=target_v,
                active_cap=active_v2v_cap,
            )

        p = np.concatenate(
            [
                pose,
                np.array([
                    self.prev_delta,
                    self.prev_v,
                    target_v,
                ]),
                ref,
                self.build_v2v_obstacle_params(),
            ]
        )

        try:
            solver_lbx = list(self.lbx)
            solver_ubx = list(self.ubx)
            if self.reverse_mode:
                for k in range(self.N):
                    speed_index = self.nx + 2 * k + 1
                    solver_lbx[speed_index] = self.reverse_speed
                    solver_ubx[speed_index] = 0.0
            else:
                # Apply the same hard ceiling to every predicted control, so
                # V2V is part of the optimization constraints as well as the
                # final actuator gate below.
                for k in range(self.N):
                    solver_ubx[self.nx + 2 * k + 1] = speed_ceiling

            v2v_was_fresh = self.v2v_data_fresh()
            solve_started = time.monotonic()
            sol = self.solver(
                x0=self.last_solution,
                lbx=solver_lbx,
                ubx=solver_ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p,
            )
            solve_elapsed = time.monotonic() - solve_started

            if solve_elapsed > self.dt:
                self.get_logger().warn(
                    f"Slow MPC solve: {solve_elapsed:.3f} s "
                    f"(control period {self.dt:.3f} s)",
                    throttle_duration_sec=1.0,
                )

            # The single-threaded executor cannot process safety callbacks
            # while IPOPT runs.  Never apply a solution computed from a V2V
            # or LiDAR snapshot that expired during that solve.
            if not self.behavior_data_fresh():
                self.get_logger().error(
                    "Discarding MPC result: behavior heartbeat became stale "
                    "during solve"
                )
                self.stop()
                self.reset_mpc_memory()
                return
            if v2v_was_fresh and not self.v2v_data_fresh():
                self.get_logger().error(
                    "Discarding MPC result: V2V snapshot became stale during "
                    "solve"
                )
                self.stop()
                self.reset_mpc_memory()
                return

            solx = sol["x"].full().flatten()

            if not np.all(np.isfinite(solx)):
                self.get_logger().error("MPC returned non-finite solution")
                self.stop()
                self.reset_mpc_memory()
                return

            if (
                self.v2v_data_fresh()
                and self.v2v_pred is not None
                and len(self.v2v_pred) > 0
            ):
                predicted_vehicle = solx[:self.nx].reshape(self.N + 1, 3)
                predicted_obstacle = np.asarray([
                    self.v2v_pred[min(k, len(self.v2v_pred) - 1)]
                    for k in range(self.v2v_ndho + 1)
                ])
                if not PathUtils.predicted_ellipse_clear(
                    predicted_vehicle,
                    predicted_obstacle,
                    self.v2v_ellipse_a,
                    self.v2v_ellipse_b,
                    stages=self.v2v_ndho + 1,
                ):
                    self.get_logger().error(
                        "V2V safety shield rejected the predicted command "
                        "horizon; publishing zero velocity",
                        throttle_duration_sec=0.5,
                    )
                    self.stop()
                    self.reset_mpc_memory()
                    return

            self.last_solution = self.shift(solx)

            u = solx[
                self.nx:self.nx + self.nu
            ].reshape(
                self.N,
                2,
            )

            delta = float(
                np.clip(
                    u[0, 0],
                    -self.max_steer,
                    self.max_steer,
                )
            )

            if self.reverse_mode:
                v = float(np.clip(u[0, 1], self.reverse_speed, 0.0))
            else:
                v = float(np.clip(u[0, 1], self.v_min, self.v_max))
            if self.drive_state in ["OVERTAKE_LEFT", "RETURN_RIGHT", "OBSTACLE_SLOW"]:
                v = float(np.clip(v, 0.0, 0.3))

            if not self.reverse_mode:
                v = max(v, 0.0)
            if startup_active:
                v = float(np.clip(v, 0.0, self.startup_v_max))

            if (
                not startup_active
                and not self.reverse_mode
                and self.drive_state in ["DRIVE", "OVERTAKE_LEFT", "RETURN_RIGHT"]
                and target_v > 0.05
                and v < 0.08
            ):
                v = 0.08

            if not self.reverse_mode:
                # Last-line actuator guarantee.  This executes after every
                # behavior/minimum-speed adjustment, so no later override can
                # exceed an active V2V cap or turn a zero target into motion.
                v = PathUtils.enforce_forward_speed_cap(
                    command_v=v,
                    target_v=target_v,
                    active_cap=active_v2v_cap,
                )
                
            self.prev_delta = delta
            self.prev_v = v

            self.log_tracking_error(
                pose=pose,
                closest_point=closest_point,
                tracking_error=tracking_error,
                yaw_error=yaw_error,
                target_v=target_v,
                v=v,
                delta=delta,
            )
            
            yaw_stable_msg = Bool()
            yaw_stable_msg.data = abs(yaw_error) < math.radians(8.0)
            self.overtake_yaw_stable_pub.publish(yaw_stable_msg)

            idx_msg = Int32()
            idx_msg.data = int(self.closest_idx)
            self.idx_pub.publish(idx_msg)

            cmd = Twist()
            cmd.linear.x = v

            if self.publish_steering_angle:
                cmd.angular.z = delta
            else:
                cmd.angular.z = v / self.L * math.tan(delta)

            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"mode={'REVERSE' if self.reverse_mode else 'FORWARD'}, "
                f"lap={self.completed_laps}/{self.target_laps}, "
                f"state={self.drive_state}, "
                f"idx={self.closest_idx}, "
                f"track_err={tracking_error:.3f}, "
                f"yaw_err={math.degrees(yaw_error):.1f}, "
                f"v={v:.2f}, "
                f"target_v={target_v:.2f}, "
                f"delta={delta:.2f}, "
                f"lane={self.lane_offset_filtered:.3f}, "
                f"avoidance={self.avoidance_offset_filtered:.3f}, "
                f"lane_valid={self.lane_valid}, "
                f"lane_active={self.current_mean_curvature < self.lane_centering_curve_limit}, "
                f"allow_overtake={allow_overtake}, "
                f"max_curv={max_curvature:.3f}, "
                f"mean_curv={mean_curvature:.3f}, "
                f"v2v={'ON' if self.v2v_data_fresh() else 'off'}, "
                f"v2v_gap={self.v2v_gap:.2f}, "
                f"v2v_cap={self.v2v_cap:.2f}, "
                f"reverse_enabled={self.enable_reverse}",
                throttle_duration_sec=1.0,
            )

        except Exception as e:
            self.get_logger().error(f"MPC failed: {e}")
            self.stop()
            self.reset_mpc_memory()


def main():
    rclpy.init()

    node = QCar2PathMPC()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()

    if hasattr(node, "tracking_log"):
        node.tracking_log.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
