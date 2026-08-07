#!/usr/bin/env python3

import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Bool, String, Int32

from qcar_science_night_pkg.lidar_sector_analyzer import LidarSectorAnalyzer
from qcar_science_night_pkg.overtake_state_machine import OvertakeStateMachine
from qcar_science_night_pkg.overtake_types import OvertakeDecision
from qcar_science_night_pkg.overtake_safety import (
    has_safe_v2v_return_clearance,
    limit_status_for_path_context,
    scan_matches_safely_offset_lead,
    should_hold_communicating_lead,
    should_inject_slow_v2v_lead,
    suppress_matched_nonblocking_v2v_lead,
)


class LidarOvertakeNode(Node):
    def __init__(self):
        super().__init__("lidar_overtake_node")

        # Sensor mounting differs between the physical QCar (the legacy scan
        # is rotated 180 degrees) and the Gazebo model (zero degrees).  Keep
        # the hardware-compatible default and override it in the sim runner.
        self.declare_parameter("front_offset_deg", 180.0)
        self.declare_parameter("sensor_timeout_sec", 0.7)
        # Opt-in, and deliberately matching path_mpc_node's enable_v2v default
        # (False).  V2V fusion is the half that STARTS overtakes; the MPC half
        # is the DCBF keep-out and speed governor that make them safe.  With
        # this defaulted True, any launch that forgot enable_v2v:=true -- both
        # science_night launch files do -- got V2V-initiated passes with no
        # barrier and no governor behind them.  Enable the two together.
        self.declare_parameter("v2v_fusion_enable", False)
        self.declare_parameter("v2v_detect_range", 2.5)
        self.declare_parameter("v2v_qcar_free_speed", 0.15)
        self.declare_parameter("v2v_slow_lead_speed_max", 0.10)
        self.declare_parameter("v2v_min_speed_advantage", 0.03)
        self.declare_parameter("v2v_stale_sec", 1.0)
        self.declare_parameter("path_spacing", 0.05)
        self.declare_parameter("path_point_count", 540)
        self.declare_parameter("max_progress_step_points", 20)
        # Path distance the car must cover before a return may even be
        # considered.  It has to span the whole encounter, not just the
        # approach: the commit distance (overtake_start_min_distance_m, the
        # range to the lead's NEAR face), plus the lead's own length, plus the
        # 0.22 m of QCar that trails the scanner, plus margin.  At the former
        # 0.80 m the return unlocked while the scanner was still short of the
        # lead's rear face -- the car reached its widest point upstream of the
        # obstacle and then merged back through it.
        self.declare_parameter("min_overtake_progress_m", 1.85)
        # Must cover the MPC's return_change_distance_m (1.60 m) so the state
        # machine holds RETURN_RIGHT -- and with it maneuver speed and the
        # swept-corridor checks -- for the whole merge S-curve.
        self.declare_parameter("min_return_progress_m", 1.70)
        # Extra distance past a completed pass after which a persistent flank
        # return is treated as scenery (this course has walls inside the
        # flank band) and loses its merge veto.  See OvertakeStateMachine.
        self.declare_parameter("flank_override_progress_m", 1.20)
        # One lane (0.43 m) plus margin.  At 0.40 m a centred ROSbot cleared
        # the QCar's flank by 0.40 - 0.096 - 0.118 = 0.19 m before tracking
        # lag, and rather less after it.
        self.declare_parameter("overtake_offset_m", 0.62)
        # Flank corridor: the lane being vacated, sampled beside and behind.
        # x_min is negative because the chassis is 0.425 m long with the
        # scanner at its centre.
        self.declare_parameter("flank_corridor_x_min_m", -0.35)
        self.declare_parameter("flank_corridor_x_max_m", 0.70)
        self.declare_parameter("flank_half_width_m", 0.14)
        self.declare_parameter("flank_body_half_width_m", 0.12)
        self.declare_parameter("min_flank_points", 2)
        self.declare_parameter("v2v_geometry_stale_sec", 0.5)
        self.declare_parameter("v2v_lead_body_radius_m", 0.20)
        self.declare_parameter("v2v_scan_association_tolerance_m", 0.12)
        self.declare_parameter("v2v_clear_metric_min", 1.05)
        # Signed along-path lead QCar must hold before RETURN_RIGHT is
        # authorized.  This MUST exceed path_mpc_node's v2v_ellipse_a (0.55 m),
        # the hard DCBF keep-out at zero lateral offset: the return collapses
        # the lateral offset to zero, so a merge authorized inside that radius
        # is one the MPC's own barrier then refuses.  See the note at the
        # arming site below for what that costs.
        self.declare_parameter("v2v_return_clearance_m", 0.80)
        self.declare_parameter("return_corridor_x_min_m", 0.16)
        self.declare_parameter("return_corridor_x_max_m", 0.70)
        self.declare_parameter("return_transition_length_m", 0.54)
        self.declare_parameter("return_outer_margin_m", 0.12)
        self.declare_parameter("return_inner_margin_m", 0.20)
        # Obstacle/emergency lookahead. See the note at the assignment site.
        self.declare_parameter("front_stop_straight_m", 1.45)
        self.declare_parameter("emergency_stop_straight_m", 0.60)
        # Lowered from 0.90/0.65.  At 0.6 m/s the car needs 0.12 m to stop
        # plus ~0.06 m of reaction, so 0.75 m still carries 4x margin over
        # the 0.40 m hard stop, and the curve corridor -- which bends up to
        # 0.62 m sideways over its length -- stops meeting scenery early.
        self.declare_parameter("front_stop_curve_m", 0.75)
        self.declare_parameter("emergency_stop_curve_m", 0.55)
        self.declare_parameter("overtake_start_min_distance_m", 1.20)
        # Range at which LANE_PROBE stops creeping closer.  0.0 means "derive
        # it", which is the only setting that cannot go wrong: the probe must
        # halt while a pass is still committable, so its floor belongs just
        # ABOVE overtake_start_min_distance_m, never below it.
        #
        # This was the deadlock.  The probe's own default was 0.50 m against a
        # 1.20 m commit distance, so an obstacle declared at 1.45 m put the car
        # into LANE_PROBE, the probe walked it in to 0.70 m, and the pass gate
        # then refused forever because 0.70 < 1.20.  Observed live as
        # "allow_raw=True | enough_dist=False" with progress frozen: the car
        # wanted to pass, was parked, and could never again satisfy the range
        # it had just driven through.
        self.declare_parameter("probe_min_gap_m", 0.0)
        self.declare_parameter("hard_stop_front_distance_m", 0.40)
        # How far the adjacent lane must be clear before a pass is allowed.
        # Previously hardcoded at the analyzer's 0.75 m default.
        self.declare_parameter("lane_clear_distance_m", 0.45)
        # Detection-box extents. These are the hard limit on what can be
        # seen at all; scale them with speed. Defaults preserve the previous
        # hardcoded values, which suit roughly 1.0 m/s and below.
        self.declare_parameter("lidar_max_range_m", 2.60)
        self.declare_parameter("front_box_max_m", 2.20)
        self.declare_parameter("side_box_max_m", 0.90)
        self.declare_parameter("emergency_box_max_m", 0.70)
        # Detection-box WIDTHS. The emergency half-width is what decides
        # whether an off-centre object is seen at all -- see the note at the
        # analyzer construction.
        self.declare_parameter("lane_width_m", 0.43)
        self.declare_parameter("emergency_half_width_m", 0.15)
        # Curvature-following corridor. This is what detects an obstacle in a
        # curve, where the wide straight-ahead rectangle is meaningless. Keep
        # it wider than the car body and narrower than the lane half-width
        # (0.215 m at the default lane_width_m) so the road edge stays out.
        # 0.16 was barely the body: a person standing slightly off the path
        # centreline fell outside it and was only caught by the close-in
        # layers, so the car drew up to their feet before stopping.
        self.declare_parameter("front_narrow_half_width_m", 0.20)
        # How far out the WIDE front box may still declare an obstacle on a
        # straight. The corridor is the primary detector; this backstop
        # exists for what the corridor's width misses (a wide or well
        # off-centre object). It must sit below the nearest wall the wide box
        # can hold on this course (measured 1.29 m) or the wall becomes an
        # obstacle again, and above the emergency range or an off-centre
        # person is met at 0.70 m.
        # Lowered 1.10 -> 0.65 against a wall measured live at 0.70 m.  The
        # log that forced it: front=0.70 fc=76 with narrow=-1.00 nc=0 --
        # the wide box held a wall while the corridor the car would actually
        # sweep was completely empty, and the car sat parked in front of
        # scenery.  0.65 is still above emergency_stop_straight_m (0.60), so
        # a genuinely wide or off-centre object is met with braking room;
        # past 0.65 the corridor decides alone, which is what detects the
        # ROSbot out to front_stop_straight_m.
        self.declare_parameter("front_wide_backstop_m", 0.65)
        self.declare_parameter("min_front_narrow_points", 2)
        # /path_curvature older than this falls back to a straight corridor,
        # which is the pre-existing behaviour.
        self.declare_parameter("path_curvature_timeout_s", 1.0)

        gp = lambda name: self.get_parameter(name).value

        self.last_drive_state = None
        self.last_behavior_snapshot = None
        self.allow_overtake = False
        # Widest drivable passing offset from the MPC. Starts at the
        # configured maximum so a missing publisher keeps the old behaviour.
        self.allowed_offset_m = float(gp("overtake_offset_m"))
        # Signed curvature of the route ahead, from the MPC. 0.0 (straight)
        # until the first message; see current_path_curvature().
        self.path_curvature = 0.0
        self.last_path_curvature_time = None
        self.last_scan_time = None
        self.safety_armed = False
        self.sensor_timeout_sec = float(gp("sensor_timeout_sec"))
        self.overtake_offset_m = float(gp("overtake_offset_m"))
        if not 0.20 <= self.overtake_offset_m <= 0.65:
            raise ValueError("overtake_offset_m must be in [0.20, 0.65]")
        self.last_commanded_offset = 999.0
        self.return_shift_estimate = self.overtake_offset_m
        self.return_progress_anchor = None

        self.current_path_idx = 0
        self.previous_path_idx = None
        self.path_progress_m = 0.0
        self.path_spacing = float(gp("path_spacing"))
        self.path_point_count = max(2, int(gp("path_point_count")))
        self.max_progress_step_points = max(
            1, int(gp("max_progress_step_points"))
        )
        self.yaw_stable = False

        # ---- V2V early-warning fusion ----
        # The V2V receiver reports the ROSbot's along-path gap. When it is
        # sufficiently slower than QCar in our lane, inject it as a virtual
        # front observation so a LiDAR-verified pass can start early. Traffic
        # near QCar's free-flow speed stays under the smooth MPC governor.
        # LiDAR keeps full authority over lane-clear checks and emergency
        # stops; without fresh V2V data this node behaves exactly as before.
        self.v2v_fusion_enable = bool(gp("v2v_fusion_enable"))
        self.v2v_detect_range = float(gp("v2v_detect_range"))
        self.v2v_qcar_free_speed = float(gp("v2v_qcar_free_speed"))
        self.v2v_slow_lead_speed_max = float(
            gp("v2v_slow_lead_speed_max")
        )
        self.v2v_min_speed_advantage = float(
            gp("v2v_min_speed_advantage")
        )
        self.v2v_stale_sec = float(gp("v2v_stale_sec"))
        self.v2v_alive = False
        self.v2v_last_rx_time = None
        self.v2v_gap = -1.0
        self.v2v_on_path = False
        self.v2v_speed = 0.0
        self.v2v_pass_required = False
        # None until the lead tells us. Unknown is not "clear": a schema 1
        # broadcaster never publishes the topic at all, and that must leave
        # the pre-coordination behavior intact rather than assert a free lane.
        self.v2v_blocked = None
        self.v2v_detour_intent = False
        self.v2v_relative_geometry = None
        self.v2v_relative_geometry_time = None
        self.v2v_geometry_stale_sec = float(gp("v2v_geometry_stale_sec"))
        self.v2v_lead_body_radius_m = float(gp("v2v_lead_body_radius_m"))
        self.v2v_scan_association_tolerance_m = float(
            gp("v2v_scan_association_tolerance_m")
        )
        self.v2v_clear_metric_min = float(gp("v2v_clear_metric_min"))
        self.v2v_return_clearance_m = float(
            gp("v2v_return_clearance_m")
        )
        # Lower bound is the MPC ellipse semi-axis plus the combined vehicle
        # half-lengths (QCar ~0.20 m, ROSbot 0.18 m from its 0.36 m collision
        # box), so no configuration can authorize a merge that the MPC barrier
        # will reject mid-manoeuvre.
        if not 0.60 <= self.v2v_return_clearance_m <= 2.0:
            raise ValueError(
                "v2v_return_clearance_m must be in [0.60, 2.0] m"
            )

        # After this idx, normal overtaking is disabled.
        # Front stop/emergency safety still remains active.
        self.disable_obstacle_after_idx = 1600

        # Distance behavior. Tunable because the right values depend on the
        # track: a tight indoor loop puts real walls inside a lookahead that
        # is perfectly sensible on an open straight, and the only way to tell
        # a wall from an obstacle at that range is how far the ROUTE bends
        # away from it. Retune per track rather than editing this file.
        #
        # On straights, look farther ahead.
        self.front_stop_straight_m = float(gp("front_stop_straight_m"))
        self.emergency_stop_straight_m = float(
            gp("emergency_stop_straight_m")
        )

        # On curves, look less far ahead because the car is turning. In a
        # curve the broad front rectangle is ignored (see
        # limit_status_by_path_context) and the curvature-following corridor
        # decides instead, so front_stop_curve_m is the range along the route
        # rather than along the straight-ahead axis.
        self.front_stop_curve_m = float(gp("front_stop_curve_m"))
        self.emergency_stop_curve_m = float(gp("emergency_stop_curve_m"))
        self.front_wide_backstop_m = float(gp("front_wide_backstop_m"))
        if not (
            self.emergency_stop_straight_m
            <= self.front_wide_backstop_m
            <= self.front_stop_straight_m
        ):
            raise ValueError(
                "front_wide_backstop_m must lie between "
                "emergency_stop_straight_m and front_stop_straight_m"
            )

        self.min_front_narrow_points = max(
            1,
            int(gp("min_front_narrow_points")),
        )
        self.path_curvature_timeout_s = float(gp("path_curvature_timeout_s"))
        if self.path_curvature_timeout_s <= 0.0:
            raise ValueError("path_curvature_timeout_s must be positive")

        # Overtake only starts if obstacle is far enough.
        # If obstacle is closer than hard_stop_front_distance, stop.
        self.overtake_start_min_distance = float(
            gp("overtake_start_min_distance_m")
        )
        self.hard_stop_front_distance = float(gp("hard_stop_front_distance_m"))

        # LANE_PROBE must stop creeping while the pass is still committable.
        self.probe_min_gap_m = float(gp("probe_min_gap_m"))
        if self.probe_min_gap_m <= 0.0:
            self.probe_min_gap_m = self.overtake_start_min_distance + 0.05
        if self.probe_min_gap_m <= self.overtake_start_min_distance:
            raise ValueError(
                f"probe_min_gap_m ({self.probe_min_gap_m:.2f}) must exceed "
                "overtake_start_min_distance_m "
                f"({self.overtake_start_min_distance:.2f}), or LANE_PROBE "
                "drives the car below the range a pass may commit from and "
                "the maneuver deadlocks with the obstacle still ahead."
            )

        # These four must stay ordered or the pass is geometrically doomed
        # before it begins:
        #
        #   front_box_max_m           what the sensor may EVER report
        #     > front_stop_straight_m   where an obstacle is declared
        #       > overtake_start_min_distance_m   where the pass may commit
        #         > lane_change_distance_m (0.90 m, in path_mpc)
        #
        # The last gap is the one that matters and the one that was wrong.
        # The lateral S-curve needs lane_change_distance_m of travel to
        # finish, so committing at 1.00 m against a 0.90 m ramp left the car
        # still moving sideways as it drew level with the obstacle -- closest
        # to it at the moment it was least displaced. That is the "it does
        # not get far enough away from the thing it is crossing" behaviour,
        # and no increase in overtake_offset_m fixes it, because the offset
        # is never reached in time.
        LANE_CHANGE_DISTANCE_M = 0.90     # mirrors path_mpc; different node
        settle = self.overtake_start_min_distance - LANE_CHANGE_DISTANCE_M
        if settle < 0.30:
            self.get_logger().warn(
                "overtake_start_min_distance_m="
                f"{self.overtake_start_min_distance:.2f} m leaves only "
                f"{settle:.2f} m to settle after the {LANE_CHANGE_DISTANCE_M:.2f} m "
                "lane-change ramp. The car will still be moving sideways as "
                "it passes. Raise it to at least "
                f"{LANE_CHANGE_DISTANCE_M + 0.30:.2f} m."
            )
        if self.front_stop_straight_m <= self.overtake_start_min_distance:
            raise ValueError(
                f"front_stop_straight_m ({self.front_stop_straight_m:.2f}) must "
                "exceed overtake_start_min_distance_m "
                f"({self.overtake_start_min_distance:.2f}): an obstacle cannot "
                "be committed to before it has been detected"
            )

        for name, value in (
            ("front_stop_straight_m", self.front_stop_straight_m),
            ("emergency_stop_straight_m", self.emergency_stop_straight_m),
            ("front_stop_curve_m", self.front_stop_curve_m),
            ("emergency_stop_curve_m", self.emergency_stop_curve_m),
            ("overtake_start_min_distance_m", self.overtake_start_min_distance),
            ("hard_stop_front_distance_m", self.hard_stop_front_distance),
        ):
            if not 0.05 <= value <= 3.0:
                raise ValueError(f"{name} must be in [0.05, 3.0] m")

        # A pass is only over once the car's TAIL has cleared the lead's nose.
        # Measured from the scanner that is the commit range plus the lead's
        # length plus the QCar's rear overhang; below that the return unlocks
        # while the two are still level.  The flank corridor is the primary
        # guard, but it is a sensor and this is the arithmetic backstop, so
        # say so loudly rather than failing closed on a live vehicle.
        self.min_overtake_progress_m = float(gp("min_overtake_progress_m"))
        minimum_safe_progress = self.overtake_start_min_distance + 0.60
        if self.min_overtake_progress_m < minimum_safe_progress:
            self.get_logger().error(
                "min_overtake_progress_m=%.2f m is shorter than the encounter "
                "it must span: overtake_start_min_distance_m=%.2f m + 0.60 m "
                "of lead length and QCar rear overhang = %.2f m. The return "
                "will unlock while the lead is still alongside."
                % (
                    self.min_overtake_progress_m,
                    self.overtake_start_min_distance,
                    minimum_safe_progress,
                )
            )

        self.analyzer = LidarSectorAnalyzer(
            front_offset_deg=float(gp("front_offset_deg")),
            # These bound what the node can EVER see. The front_stop_* and
            # emergency_stop_* parameters only shrink the effective distance
            # inside these boxes -- they cannot extend past them. So the
            # boxes, not those parameters, are what must grow with speed.
            #
            # Stopping distance is v^2/(2a) plus roughly 0.3 m of sensing and
            # control latency: 0.35 m at 0.5 m/s, 0.63 m at 1.0, 1.65 m at
            # 2.0 (using max_decel 1.5). The 0.70 m default emergency box is
            # therefore fine to ~1.0 m/s and too short above it.
            max_range=float(gp("lidar_max_range_m")),
            min_range=0.05,
            lane_width=float(gp("lane_width_m")),

            # Keep analyzer boxes large enough.
            # We reduce the effective distance later based on curve/straight.
            front_x_min=0.10,
            front_x_max=float(gp("front_box_max_m")),

            side_x_min=0.10,
            side_x_max=float(gp("side_box_max_m")),

            emergency_x_min=0.03,
            emergency_x_max=float(gp("emergency_box_max_m")),
            # QCar body is roughly 25 cm wide.  Keep a direct vehicle-width
            # collision corridor while allowing a curved road edge to leave
            # the straight-ahead box without becoming a false obstacle.
            #
            # The 0.12 m default is HALF the body width, which is why a
            # person (wide, always clipping the centreline) triggers a stop
            # but a bottle or another robot sitting 0.20 m off centre does
            # not. Widen toward the true half-width plus a margin to catch
            # off-centre objects; the cost is that road edges enter the box
            # sooner on curves.
            emergency_half_width=float(gp("emergency_half_width_m")),
            emergency_center_y=0.0,

            # Both the emergency corridor and this one bend along
            # /path_curvature, so an object in a bend is measured against the
            # route the car is about to drive rather than against straight
            # ahead.
            front_narrow_half_width=float(gp("front_narrow_half_width_m")),

            # DETECTION DISTANCES. These were never passed, so the analyzer
            # silently used its own defaults of 0.75 m and no amount of
            # raising front_stop_straight_m had any effect: obstacle_ahead
            # simply could not fire beyond 0.75 m, and
            # limit_status_for_path_context can only narrow that further,
            # never widen it. Braking from 0.75 m left the car at about
            # 0.4 m, below overtake_start_min_distance_m, so it could see the
            # obstacle but never had room to commit to a pass -- the operator
            # had to push the car forward by hand every time.
            overtake_start_distance=float(gp("front_stop_straight_m")),
            emergency_distance=float(gp("emergency_stop_straight_m")),

            # Distance the adjacent lane must be clear for before a pass is
            # allowed. Also previously unpassed and stuck at 0.75 m, which is
            # why L=False was reported with an empty passing lane once
            # side_box_max_m was widened: a bigger box sees more, and every
            # return inside 0.75 m counted as blocking.
            lane_clear_distance=float(gp("lane_clear_distance_m")),

            # Keep low so legs/feet/thin objects are not missed.
            min_front_points=1,
            min_side_points=1,
            min_emergency_points=1,
            return_x_min=float(gp("return_corridor_x_min_m")),
            return_x_max=float(gp("return_corridor_x_max_m")),
            return_transition_length=float(
                gp("return_transition_length_m")
            ),
            return_outer_margin=float(gp("return_outer_margin_m")),
            return_inner_margin=float(gp("return_inner_margin_m")),
            flank_x_min=float(gp("flank_corridor_x_min_m")),
            flank_x_max=float(gp("flank_corridor_x_max_m")),
            flank_half_width=float(gp("flank_half_width_m")),
            flank_body_half_width=float(gp("flank_body_half_width_m")),
            min_flank_points=int(gp("min_flank_points")),
        )

        self.state_machine = OvertakeStateMachine(
            overtake_offset=self.overtake_offset_m,
            obstacle_confirm_required=1,
            left_clear_confirm_required=1,
            no_obstacle_confirm_required=5,
            right_clear_confirm_required=3,
            flank_clear_confirm_required=4,
            return_confirm_required=10,
            min_overtake_steps=70,
            min_overtake_progress=self.min_overtake_progress_m,
            min_return_progress=float(gp("min_return_progress_m")),
            flank_override_progress=float(gp("flank_override_progress_m")),
            probe_min_gap_m=self.probe_min_gap_m,
        )

        self.offset_pub = self.create_publisher(Float32, "/avoidance_offset", 10)
        self.motion_pub = self.create_publisher(Bool, "/motion_enable", 10)
        self.state_pub = self.create_publisher(String, "/drive_state", 10)
        self.behavior_state_pub = self.create_publisher(
            String,
            "/v2v/qcar/behavior_state",
            10,
        )
        self.behavior_command_pub = self.create_publisher(
            String,
            "/v2v/qcar/command",
            10,
        )
        # Consumed by v2v_receiver_node, which owns the socket and puts this
        # on the wire to the ROSbot.
        self.hold_request_pub = self.create_publisher(
            Bool,
            "/v2v/hold_request",
            10,
        )

        self.sound_pub = self.create_publisher(
            String,
            "/qcar2/sound_event",
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.allow_overtake_sub = self.create_subscription(
            Bool,
            "/allow_overtake",
            self.allow_overtake_callback,
            10,
        )

        self.allowed_offset_sub = self.create_subscription(
            Float32,
            "/overtake_offset_allowed",
            self.allowed_offset_callback,
            10,
        )

        self.path_curvature_sub = self.create_subscription(
            Float32,
            "/path_curvature",
            self.path_curvature_callback,
            10,
        )

        self.idx_sub = self.create_subscription(
            Int32,
            "/current_path_idx",
            self.idx_callback,
            10,
        )

        self.yaw_sub = self.create_subscription(
            Bool,
            "/overtake_yaw_stable",
            self.yaw_callback,
            10,
        )

        self.v2v_alive_sub = self.create_subscription(
            Bool,
            "/v2v/alive",
            self.v2v_alive_callback,
            10,
        )

        self.v2v_gap_sub = self.create_subscription(
            Float32,
            "/v2v/gap",
            self.v2v_gap_callback,
            10,
        )

        self.v2v_on_path_sub = self.create_subscription(
            Bool,
            "/v2v/on_path",
            self.v2v_on_path_callback,
            10,
        )

        self.v2v_blocked_sub = self.create_subscription(
            Bool,
            "/v2v/rosbot_blocked",
            self.v2v_blocked_callback,
            10,
        )

        self.v2v_detour_sub = self.create_subscription(
            Bool,
            "/v2v/rosbot_detour_intent",
            self.v2v_detour_callback,
            10,
        )

        self.v2v_speed_sub = self.create_subscription(
            Float32,
            "/v2v/rosbot_speed",
            self.v2v_speed_callback,
            10,
        )
        self.v2v_relative_sub = self.create_subscription(
            PointStamped,
            "/v2v/relative_geometry",
            self.v2v_relative_callback,
            10,
        )

        self.watchdog_timer = self.create_timer(0.1, self.watchdog)

        self.publish_decision(
            OvertakeDecision("STARTUP_WAIT", 999.0, False)
        )

        self.get_logger().info(
            "LiDAR overtake node ready: front_offset=%.1f deg, "
            "timeout=%.2f s, overtake_offset=%.2f m, "
            "slow lead<=%.2f m/s, speed advantage>=%.2f m/s, "
            "progress-based pass completion"
            % (
                float(gp("front_offset_deg")),
                self.sensor_timeout_sec,
                self.overtake_offset_m,
                self.v2v_slow_lead_speed_max,
                self.v2v_min_speed_advantage,
            )
        )

    def idx_callback(self, msg):
        new_idx = int(msg.data)
        if not 0 <= new_idx < self.path_point_count:
            self.get_logger().warn(
                f"Ignoring out-of-range path index {new_idx}",
                throttle_duration_sec=2.0,
            )
            return

        if self.previous_path_idx is not None:
            raw_delta = new_idx - self.previous_path_idx
            if (
                raw_delta < 0
                and self.previous_path_idx > int(0.8 * self.path_point_count)
                and new_idx < int(0.2 * self.path_point_count)
            ):
                # Genuine lap wrap.
                delta = self.path_point_count - self.previous_path_idx + new_idx
            elif raw_delta >= 0:
                delta = raw_delta
            else:
                # Nearest-waypoint jitter can move one or two indices
                # backwards.  It is not negative travelled distance.
                delta = 0

            # Projection jumps are not physical motion.  Reject them rather
            # than allowing one bad localization sample to finish a pass.
            if delta <= self.max_progress_step_points:
                self.path_progress_m += delta * self.path_spacing

        self.previous_path_idx = new_idx
        self.current_path_idx = new_idx

    def yaw_callback(self, msg):
        self.yaw_stable = bool(msg.data)

    def v2v_alive_callback(self, msg):
        self.v2v_alive = bool(msg.data)
        self.v2v_last_rx_time = self.get_clock().now()

    def v2v_gap_callback(self, msg):
        self.v2v_gap = float(msg.data)

    def v2v_on_path_callback(self, msg):
        self.v2v_on_path = bool(msg.data)

    def v2v_speed_callback(self, msg):
        self.v2v_speed = float(msg.data)

    def v2v_blocked_callback(self, msg):
        self.v2v_blocked = bool(msg.data)

    def v2v_detour_callback(self, msg):
        self.v2v_detour_intent = bool(msg.data)

    def v2v_relative_callback(self, msg):
        values = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        if all(np.isfinite(value) for value in values):
            self.v2v_relative_geometry = values
            self.v2v_relative_geometry_time = self.get_clock().now()

    def v2v_geometry_fresh(self):
        if self.v2v_relative_geometry_time is None:
            return False
        age = (
            self.get_clock().now() - self.v2v_relative_geometry_time
        ).nanoseconds * 1e-9
        return 0.0 <= age < self.v2v_geometry_stale_sec

    def known_lead_is_safely_offset(self, physical_status):
        """True only when a close scan return matches the safe V2V lead.

        This exception is restricted to a committed pass or its return.  A
        return that is materially closer than the predicted lead surface, or
        enters the V2V ellipse, remains unsafe and keeps hard-stop authority.
        """
        if not (
            self.v2v_pass_required
            and self.state_machine.state
            in [self.state_machine.OVERTAKE, self.state_machine.RETURN]
            and self.v2v_fresh()
            and self.v2v_geometry_fresh()
            and self.v2v_relative_geometry is not None
        ):
            return False

        return scan_matches_safely_offset_lead(
            physical_status,
            self.v2v_relative_geometry,
            self.v2v_clear_metric_min,
            self.v2v_lead_body_radius_m,
            self.v2v_scan_association_tolerance_m,
        )

    def v2v_fresh(self):
        if not self.v2v_fusion_enable or self.v2v_last_rx_time is None:
            return False
        age = (
            self.get_clock().now() - self.v2v_last_rx_time
        ).nanoseconds * 1e-9
        return self.v2v_alive and 0.0 <= age < self.v2v_stale_sec

    def inject_v2v_obstacle(self, status):
        """Fuse the V2V-reported ROSbot in as a virtual front obstacle.

        Strictly additive: it can set obstacle_ahead and shorten front_min,
        never clear them. left/right/emergency stay LiDAR-only, so a lane
        change still requires the physical sensor to confirm clear space.
        Returns (status, injected).
        """
        # On a curve, the MPC governor keeps following/braking. Injecting a
        # far virtual obstacle there would enter WAIT and deadlock before the
        # next safe passing section. Physical LiDAR authority remains active.
        if not should_inject_slow_v2v_lead(
            fresh=self.v2v_fresh(),
            on_path=self.v2v_on_path,
            lead_speed_mps=self.v2v_speed,
            qcar_free_speed_mps=self.v2v_qcar_free_speed,
            max_lead_speed_mps=self.v2v_slow_lead_speed_max,
            min_speed_advantage_mps=self.v2v_min_speed_advantage,
            gap_m=self.v2v_gap,
            detect_range_m=self.v2v_detect_range,
            overtake_allowed=self.allow_overtake,
            lead_blocked=self.v2v_blocked,
        ):
            return status, False

        front_min = status.front_min
        if front_min <= 0.0 or self.v2v_gap < front_min:
            front_min = self.v2v_gap

        fused = type(status)(
            obstacle_ahead=True,
            emergency=status.emergency,
            left_clear=status.left_clear,
            right_clear=status.right_clear,
            front_min=front_min,
            left_min=status.left_min,
            right_min=status.right_min,
            front_count=max(status.front_count, 1),
            left_count=status.left_count,
            right_count=status.right_count,
            front_narrow_min=status.front_narrow_min,
            front_narrow_count=status.front_narrow_count,
            emergency_min=status.emergency_min,
            emergency_count=status.emergency_count,
        )
        return fused, True

    def allow_overtake_callback(self, msg):
        self.allow_overtake = bool(msg.data)

    def allowed_offset_callback(self, msg):
        """Widest passing offset the MPC says is drivable here.

        Adopted only while a pass has not yet been committed.  Changing the
        offset mid-pass would restart the MPC's S-curve from a new target and
        swing the car sideways alongside the obstacle, so once OVERTAKE_LEFT
        is entered the committed width is frozen until the car is back in
        its own lane.
        """
        value = float(msg.data)
        if not math.isfinite(value) or value < 0.0:
            return
        self.allowed_offset_m = min(value, self.overtake_offset_m)
        if self.state_machine.state in (
            self.state_machine.DRIVE,
            self.state_machine.PROBE,
            self.state_machine.WAIT,
        ):
            self.state_machine.overtake_offset = self.allowed_offset_m

    def path_curvature_callback(self, msg):
        value = float(msg.data)
        if not math.isfinite(value):
            return
        self.path_curvature = value
        self.last_path_curvature_time = self.get_clock().now()

    def current_path_curvature(self):
        """Signed route curvature for the detection corridor, 0.0 if stale.

        Falling back to straight is the conservative choice only in the sense
        that it restores the previous behaviour; it does mean a curve is once
        again detected by the emergency corridor alone. The MPC publishes this
        every control cycle, so a timeout here means the MPC is gone, and the
        node already stops the car for that reason.
        """
        if self.last_path_curvature_time is None:
            return 0.0

        age = (
            self.get_clock().now() - self.last_path_curvature_time
        ).nanoseconds * 1e-9
        if age > self.path_curvature_timeout_s:
            return 0.0

        return self.path_curvature

    def is_fresh(self, stamp):
        if stamp is None:
            return False

        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        # A negative age means /clock jumped backwards (world reset).  Do not
        # certify stale pre-reset data as fresh; the next scan will re-arm it.
        return 0.0 <= age < self.sensor_timeout_sec

    def watchdog(self):
        if not self.safety_armed:
            self.publish_decision(
                OvertakeDecision("STARTUP_WAIT", 999.0, False)
            )
            return

        if not self.is_fresh(self.last_scan_time):
            self.publish_decision(
                OvertakeDecision("LIDAR_TIMEOUT", 999.0, False)
            )

    def publish_sound_if_needed(self, decision):
        current_state = str(decision.state)

        obstacle_states = [
            "OVERTAKE_LEFT",
            "EMERGENCY_STOP",
        ]

        if (
            current_state in obstacle_states
            and self.last_drive_state != current_state
        ):
            self.sound_pub.publish(String(data="obstacle"))

        self.last_drive_state = current_state

    def limit_status_by_path_context(self, status):
        """Thin wrapper; the logic lives in overtake_safety so it is
        testable without ROS. See limit_status_for_path_context."""
        return limit_status_for_path_context(
            status,
            allow_overtake=self.allow_overtake,
            front_stop_straight_m=self.front_stop_straight_m,
            emergency_stop_straight_m=self.emergency_stop_straight_m,
            front_stop_curve_m=self.front_stop_curve_m,
            emergency_stop_curve_m=self.emergency_stop_curve_m,
            min_front_narrow_points=self.min_front_narrow_points,
            front_backstop_m=self.front_wide_backstop_m,
        )

    def force_no_overtake_zone(self, status):
        """
        After disable_obstacle_after_idx, do not overtake.

        But do NOT disable obstacle_ahead or emergency.
        This keeps people/front safety active.
        """

        if self.current_path_idx >= self.disable_obstacle_after_idx:
            status = type(status)(
                obstacle_ahead=status.obstacle_ahead,
                emergency=status.emergency,
                left_clear=True,
                right_clear=True,
                front_min=status.front_min,
                left_min=status.left_min,
                right_min=status.right_min,
                front_count=status.front_count,
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

        return status

    def swept_return_lateral_shift(self):
        """Estimated remaining right shift for return-corridor analysis.

        During the committed pass the full lane offset is inspected.  Once a
        return begins, the estimate contracts with measured path progress so
        the fixed road boundary remains outside the corridor as the vehicle
        moves back right.  A blocked return stops motion in the state machine,
        so progress (and therefore this estimate) naturally freezes without
        feeding the previous raw offset command back into the classifier.
        """
        state = self.state_machine.state
        if state == self.state_machine.OVERTAKE:
            self.return_shift_estimate = self.overtake_offset_m
            self.return_progress_anchor = self.path_progress_m
            return self.return_shift_estimate

        if state != self.state_machine.RETURN:
            self.return_shift_estimate = self.overtake_offset_m
            self.return_progress_anchor = None
            return None

        if self.return_progress_anchor is None:
            self.return_progress_anchor = (
                self.state_machine.return_start_progress
                if self.state_machine.return_start_progress is not None
                else self.path_progress_m
            )

        distance = max(
            0.0,
            self.path_progress_m - self.return_progress_anchor,
        )
        transition_distance = max(
            self.state_machine.min_return_progress,
            1e-6,
        )
        fraction = min(1.0, distance / transition_distance)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        self.return_shift_estimate = (
            self.overtake_offset_m * (1.0 - blend)
        )
        return self.return_shift_estimate

    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()

        if not self.safety_armed:
            self.safety_armed = True
            self.get_logger().warn("Safety armed: LiDAR-only mode")

        return_shift = self.swept_return_lateral_shift()
        path_curvature = self.current_path_curvature()

        # Shift the emergency corridor to the lateral offset currently
        # commanded, so it sweeps with a lane change instead of staring
        # straight ahead through it. 999.0 is the node's "no offset" sentinel.
        lateral_intent = 0.0
        if abs(self.last_commanded_offset) < 900.0:
            lateral_intent = float(self.last_commanded_offset)

        status = self.analyzer.analyze(
            msg,
            return_lateral_shift_m=return_shift,
            path_curvature=path_curvature,
            lateral_intent_m=lateral_intent,
            # Same displacement, opposite question: return_lateral_shift_m
            # asks whether the lane ahead is clear to merge into, this asks
            # whether the lead is still beside us.  None outside a pass, which
            # correctly leaves the flank unmeasured and reported clear.
            flank_lateral_shift_m=return_shift,
        )

        status, path_context, effective_front_stop_m, effective_emergency_stop_m = (
            self.limit_status_by_path_context(status)
        )

        status = self.force_no_overtake_zone(status)
        physical_status = status

        # A clear (or unconfirmed) communicating lead remains under the MPC's
        # longitudinal following governor.  Its positively associated LiDAR
        # return must not independently trigger a lateral pass.  This filter
        # cannot suppress emergency or unmatched/closer physical returns.
        status, followed_lead_suppressed = (
            suppress_matched_nonblocking_v2v_lead(
                status,
                fresh=self.v2v_fresh(),
                on_path=self.v2v_on_path,
                signed_gap_m=self.v2v_gap,
                detect_range_m=self.v2v_detect_range,
                lead_blocked=self.v2v_blocked,
                geometry_fresh=self.v2v_geometry_fresh(),
                relative_geometry=self.v2v_relative_geometry,
                lead_body_radius_m=self.v2v_lead_body_radius_m,
                association_tolerance_m=(
                    self.v2v_scan_association_tolerance_m
                ),
            )
        )

        status, v2v_injected = self.inject_v2v_obstacle(status)

        # If obstacle is too close, stop immediately.
        # Do not start an overtake when there is not enough room.
        safe_known_lead = self.known_lead_is_safely_offset(physical_status)
        if (
            status.obstacle_ahead
            and status.front_min < self.hard_stop_front_distance
            and not safe_known_lead
        ):
            decision = OvertakeDecision(
                "EMERGENCY_STOP",
                999.0,
                False,
            )

            self.publish_sound_if_needed(decision)
            self.publish_decision(decision)

            self.get_logger().warn(
                f"Hard stop: obstacle too close for overtake | "
                f"idx={self.current_path_idx} | "
                f"front_min={status.front_min:.2f} m | "
                f"hard_stop={self.hard_stop_front_distance:.2f} m | "
                f"context={path_context}"
            )

            return

        # Only allow overtake if obstacle is far enough.
        if status.obstacle_ahead:
            enough_distance_to_overtake = (
                status.front_min >= self.overtake_start_min_distance
            )
        else:
            enough_distance_to_overtake = True

        # Overtake requires:
        # 1. MPC says path is straight enough
        # 2. obstacle is far enough
        # 3. still before disable index
        overtake_allowed = (
            self.allow_overtake
            and enough_distance_to_overtake
            and self.current_path_idx < self.disable_obstacle_after_idx
        )

        # Arm the relative-clearance gate for ANY pass begun while a
        # communicating vehicle is known to be ahead on our path -- not only
        # for passes that V2V itself injected.  A pass started by physical
        # LiDAR (the ROSbot inside the 0.9 m front box, or moving faster than
        # v2v_slow_lead_speed_max so no injection happens) would otherwise
        # return on the QCar's own 0.80 m of travel plus a forward-only
        # corridor (return_x_min 0.16 m .. return_x_max 0.70 m).  A vehicle
        # alongside or behind sits at x <= 0.16 m and is invisible to that
        # corridor, so the state machine merges back with no evidence it ever
        # got past -- which is how a pass that never needed to happen ends in
        # contact.  The signed V2V gap is the only measurement that covers the
        # alongside/behind region.
        v2v_lead_ahead_on_path = (
            self.v2v_fresh()
            and self.v2v_on_path
            and 0.0 <= self.v2v_gap < self.v2v_detect_range
        )
        if (
            (v2v_injected or v2v_lead_ahead_on_path)
            and self.state_machine.state
            in [self.state_machine.DRIVE, self.state_machine.WAIT]
        ):
            self.v2v_pass_required = True

        pass_confirmed = (
            not self.v2v_pass_required
            or has_safe_v2v_return_clearance(
                fresh=self.v2v_fresh(),
                on_path=self.v2v_on_path,
                signed_gap_m=self.v2v_gap,
                required_ahead_m=self.v2v_return_clearance_m,
            )
        )

        decision = self.state_machine.update(
            status=status,
            overtake_allowed=overtake_allowed,
            yaw_stable=self.yaw_stable,
            progress=self.path_progress_m,
            pass_confirmed=pass_confirmed,
        )

        if decision.state == self.state_machine.DRIVE:
            self.v2v_pass_required = False

        # Sequence the lead for exactly as long as we occupy its passing lane.
        # RETURN_RIGHT is included deliberately: the merge is when the two
        # bodies are closest, and releasing the lead there is what let it
        # start its own detour into the lane we are still vacating.
        qcar_is_passing = decision.state in (
            self.state_machine.OVERTAKE,
            self.state_machine.RETURN,
        )
        hold_lead = should_hold_communicating_lead(
            fresh=self.v2v_fresh(),
            on_path=self.v2v_on_path,
            signed_gap_m=self.v2v_gap,
            interaction_range_m=self.v2v_detect_range,
            qcar_is_passing=qcar_is_passing,
            lead_blocked=self.v2v_blocked,
            lead_detour_intent=self.v2v_detour_intent,
        )
        self.hold_request_pub.publish(Bool(data=bool(hold_lead)))

        self.publish_sound_if_needed(decision)
        self.publish_decision(decision)

        self.get_logger().info(
            f"idx={self.current_path_idx} | "
            f"progress={self.path_progress_m:.2f} m | "
            f"context={path_context} | "
            f"state={decision.state} | "
            f"obs={status.obstacle_ahead} | "
            f"emg={status.emergency} | "
            f"L={status.left_clear} | "
            f"R={status.right_clear} | "
            f"FLANK={status.flank_clear} | "
            f"flank_n={status.flank_count} | "
            f"allow_raw={self.allow_overtake} | "
            f"allow_final={overtake_allowed} | "
            f"yaw_stable={self.yaw_stable} | "
            f"enough_dist={enough_distance_to_overtake} | "
            f"front={status.front_min:.2f} | "
            f"narrow={status.front_narrow_min:.2f} | "
            f"nc={status.front_narrow_count} | "
            f"emg_min={status.emergency_min:.2f} | "
            f"kappa={path_curvature:+.2f} | "
            f"left={status.left_min:.2f} | "
            f"right={status.right_min:.2f} | "
            f"front_limit={effective_front_stop_m:.2f} | "
            f"emg_limit={effective_emergency_stop_m:.2f} | "
            f"hard_stop={self.hard_stop_front_distance:.2f} | "
            f"overtake_min={self.overtake_start_min_distance:.2f} | "
            f"fc={status.front_count} | "
            f"lc={status.left_count} | "
            f"rc={status.right_count} | "
            f"v2v={'INJ' if v2v_injected else ('on' if self.v2v_fresh() else 'off')} | "
            f"v2v_follow={'MATCHED' if followed_lead_suppressed else 'no'} | "
            f"v2v_gap={self.v2v_gap:.2f} | "
            f"return_gap={self.v2v_return_clearance_m:.2f} | "
            f"pass_ok={pass_confirmed} | "
            f"v2v_geom_clear={safe_known_lead} | "
            f"return_sweep={-1.0 if return_shift is None else return_shift:.2f} | "
            f"offset={decision.offset:.2f} | "
            f"motion={decision.motion_enabled}",
            throttle_duration_sec=0.5,
        )

    def publish_decision(self, decision):
        offset_msg = Float32()
        offset_msg.data = float(decision.offset)

        motion_msg = Bool()
        motion_msg.data = bool(decision.motion_enabled)

        state_msg = String()
        state_msg.data = str(decision.state)

        self.offset_pub.publish(offset_msg)
        self.motion_pub.publish(motion_msg)
        self.state_pub.publish(state_msg)
        self.last_commanded_offset = float(decision.offset)

        v2v_is_fresh = self.v2v_fresh()
        if not v2v_is_fresh:
            lead_mode = "NO_FRESH_LEAD"
        elif not self.v2v_on_path or self.v2v_gap < 0.0:
            lead_mode = "NO_LEAD_AHEAD"
        elif self.v2v_blocked is not True:
            lead_mode = (
                "CLEAR_LEAD_MPC_FOLLOW"
                if self.v2v_blocked is False
                else "UNCONFIRMED_LEAD_MPC_FOLLOW"
            )
        elif 0.0 <= self.v2v_speed <= min(
            self.v2v_slow_lead_speed_max,
            self.v2v_qcar_free_speed - self.v2v_min_speed_advantage,
        ):
            lead_mode = (
                "STOPPED_LEAD_PASS_CANDIDATE"
                if self.v2v_speed < 0.02
                else "SLOW_MOVING_LEAD_PASS_CANDIDATE"
            )
        else:
            lead_mode = "MOVING_LEAD_MPC_FOLLOW"

        behavior = {
            "role": "QCAR_REACTIVE",
            "state": str(decision.state),
            "lead_mode": lead_mode,
            "v2v_fresh": v2v_is_fresh,
            "lead_on_path": bool(self.v2v_on_path),
            "lead_gap_m": round(float(self.v2v_gap), 3),
            "required_return_clearance_m": round(
                float(self.v2v_return_clearance_m), 3
            ),
            "lead_speed_mps": round(float(self.v2v_speed), 3),
            "v2v_pass_required": bool(self.v2v_pass_required),
            "rosbot_v2v_overtake": False,
        }
        command = {
            "motion_enabled": bool(decision.motion_enabled),
            "avoidance_offset_m": (
                None if float(decision.offset) >= 999.0
                else round(float(decision.offset), 3)
            ),
            "stop_requested": not bool(decision.motion_enabled),
            "drive_state": str(decision.state),
        }
        behavior_json = json.dumps(behavior, separators=(",", ":"))
        command_json = json.dumps(command, separators=(",", ":"))
        self.behavior_state_pub.publish(String(data=behavior_json))
        self.behavior_command_pub.publish(String(data=command_json))

        snapshot = (behavior_json, command_json)
        if snapshot != self.last_behavior_snapshot:
            self.get_logger().info(
                f"V2V_BEHAVIOR {behavior_json} CMD {command_json}"
            )
            self.last_behavior_snapshot = snapshot


def main():
    rclpy.init()

    node = LidarOvertakeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_decision(
        OvertakeDecision("SHUTDOWN_STOP", 999.0, False)
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
