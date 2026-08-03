#!/usr/bin/env python3
"""V2V receiver — runs on the QCar.

Listens for UDP state datagrams from the ROSbot broadcaster, validates them,
transforms them into the QCar's map frame, projects the ROSbot onto the QCar's
reference trajectory, and republishes everything as local ROS topics for the
MPC and the LiDAR overtake node:

    /v2v/alive           std_msgs/Bool      fresh, localized data available
    /v2v/rosbot_pose     geometry_msgs/PoseStamped   (map frame)
    /v2v/predicted_path  nav_msgs/Path      predicted ROSbot poses (map frame)
    /v2v/rosbot_speed    std_msgs/Float32   [m/s]
    /v2v/gap             std_msgs/Float32   signed along-path gap QCar ->
                                            ROSbot [m] (positive=ahead,
                                            negative=behind, NaN=unknown)
    /v2v/on_path         std_msgs/Bool      ROSbot within lane_half_width of
                                            the QCar's reference path
    /v2v/stats           std_msgs/String    1 Hz JSON diagnostics

Fail-safe contract: if this node is not running, the link is down, packets
are stale, malformed, or the ROSbot is not localized, /v2v/alive goes (or
stays) False and every consumer reverts to pre-V2V behavior. Stale data is
NEVER treated as "clear".

Freshness is judged by packet ARRIVAL time (monotonic clock), so the two
robots do not need synchronized clocks.

Run:
    ros2 run qcar_science_night_pkg v2v_receiver
    # optionally: --ros-args --params-file <pkg>/config/v2v_params.yaml
"""

import json
import math
import os
import socket
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException

from qcar_science_night_pkg.path_utils import PathUtils
from qcar_science_night_pkg.v2v_common import (
    PacketError,
    PathProjector,
    pack_command,
    parse_packet,
    se2_apply,
    se2_apply_array,
)
from qcar_science_night_pkg import v2v_conflict
from qcar_science_night_pkg.v2v_conflict import (
    classify_encounter,
    resample_horizon,
    resolve_encounter,
    sample_along_path,
)


class V2VReceiverNode(Node):

    def __init__(self):
        super().__init__("v2v_receiver_node")

        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("bind_port", 47100)
        self.declare_parameter("expected_vehicle_id", "rosbot3")
        self.declare_parameter("stale_sec", 0.6)

        # Rigid transform: ROSbot map frame -> QCar map frame. Identity when
        # both robots localize against the same map (intended deployment).
        self.declare_parameter("frame_tx", 0.0)
        self.declare_parameter("frame_ty", 0.0)
        self.declare_parameter("frame_tyaw", 0.0)

        # Must match the MPC's reference so gap/on_path mean the same thing.
        self.declare_parameter(
            "trajectory_file",
            "/home/nvidia/ros2_ws/recorded_path_amcl_final_long.npy",
        )
        self.declare_parameter("path_spacing", 0.03)
        self.declare_parameter("loop_path", True)
        self.declare_parameter("lane_half_width", 0.35)

        # Reverse command channel (QCar -> ROSbot). Sending is what makes the
        # sequencing possible: the ROSbot's own detour feasibility check
        # cannot see a QCar closing from behind, so QCar -- the only vehicle
        # running an optimizer -- tells it when to wait. Set command_ip empty
        # to disable transmission entirely (pure listen-only, pre-coordination
        # behavior).
        self.declare_parameter("command_ip", "")
        self.declare_parameter("command_port", 47101)
        self.declare_parameter("command_vehicle_id", "qcar2")
        self.declare_parameter("command_rate_hz", 10.0)
        # Hold lease length. Long enough to ride out packet loss at the
        # command rate, short enough that a dead link frees the ROSbot fast.
        self.declare_parameter("command_ttl_sec", 1.0)
        # Absolute ceiling on one continuous hold. A pass takes a few seconds;
        # anything longer means QCar's state machine is wedged, and a robot
        # parked in a live lane is worse than an uncoordinated one.
        self.declare_parameter("max_hold_sec", 12.0)

        # ---- Conflict prediction ----
        # A response is owed only when the two bodies are predicted to come
        # within conflict_radius inside conflict_horizon. Nearby-but-diverging
        # traffic produces nothing at all.
        self.declare_parameter("conflict_horizon_sec", 6.0)
        self.declare_parameter("conflict_step_sec", 0.2)
        self.declare_parameter("conflict_radius_m", 0.55)
        self.declare_parameter("qcar_free_speed", 0.15)
        # Yielding to a faster vehicle passing us: ease off, do NOT brake
        # hard. Braking lengthens the time it spends alongside.
        self.declare_parameter("yield_speed", 0.10)
        self.declare_parameter("give_way_speed", 0.05)
        self.declare_parameter("lead_pass_speed_max", 0.10)

        self.declare_parameter("log_file", "/home/nvidia/ros2_ws/v2v_rx_log.csv")

        gp = lambda name: self.get_parameter(name).value
        self.bind_ip = str(gp("bind_ip"))
        self.bind_port = int(gp("bind_port"))
        self.expected_id = str(gp("expected_vehicle_id"))
        self.stale_sec = float(gp("stale_sec"))
        self.frame_tx = float(gp("frame_tx"))
        self.frame_ty = float(gp("frame_ty"))
        self.frame_tyaw = float(gp("frame_tyaw"))
        self.lane_half_width = float(gp("lane_half_width"))

        # Reference path — used only for gap/on_path projection. If it fails
        # to load, pose/prediction topics still work; gap stays NaN (unknown)
        # and the MPC governor simply never engages.
        self.projector = None
        trajectory_file = str(gp("trajectory_file"))
        try:
            trajectory = PathUtils.load_trajectory(trajectory_file)
            self.projector = PathProjector(
                trajectory,
                spacing=float(gp("path_spacing")),
                loop=bool(gp("loop_path")),
            )
            self.get_logger().info(
                f"Projection path loaded: {trajectory_file} "
                f"({self.projector.n} pts, {self.projector.path_length():.1f} m)"
            )
        except Exception as e:
            self.get_logger().error(
                f"Could not load trajectory '{trajectory_file}': {e} — "
                "gap/on_path will stay unknown (governor inactive)."
            )

        # CSV log for the evaluation section.
        self.log_fh = None
        log_file = os.path.expanduser(str(gp("log_file")))
        try:
            self.log_fh = open(log_file, "w")
            self.log_fh.write(
                "t_rx,seq,age_gap_ms,x,y,yaw,v,localized,moving,"
                "qcar_idx,rosbot_idx,gap,lat_offset,on_path\n"
            )
            self.log_fh.flush()
        except OSError as e:
            self.get_logger().warn(f"V2V log disabled ({log_file}: {e})")

        # TF for the QCar's own pose (to compute the along-path gap).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Command transmitter state ----
        self.command_ip = str(gp("command_ip")).strip()
        self.command_port = int(gp("command_port"))
        self.command_vehicle_id = str(gp("command_vehicle_id"))
        self.command_ttl_sec = float(gp("command_ttl_sec"))
        self.max_hold_sec = float(gp("max_hold_sec"))
        self.command_seq = 0
        self.command_tx = 0
        self.command_errors = 0
        self.hold_requested = False
        self.hold_started = None
        self.hold_capped = False
        self.tx_sock = None
        if self.command_ip:
            self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tx_sock.setblocking(False)
            self.get_logger().info(
                f"V2V commands -> udp://{self.command_ip}:{self.command_port} "
                f"as '{self.command_vehicle_id}', ttl {self.command_ttl_sec}s"
            )
        else:
            self.get_logger().info(
                "V2V command transmission disabled (command_ip unset); "
                "the ROSbot will not be sequenced during passes"
            )

        self.alive_pub = self.create_publisher(Bool, "/v2v/alive", 10)
        self.blocked_pub = self.create_publisher(Bool, "/v2v/rosbot_blocked", 10)
        self.detour_pub = self.create_publisher(
            Bool, "/v2v/rosbot_detour_intent", 10
        )
        self.holding_pub = self.create_publisher(Bool, "/v2v/hold_active", 10)
        self.encounter_pub = self.create_publisher(String, "/v2v/encounter", 10)
        self.speed_cap_pub = self.create_publisher(
            Float32, "/v2v/conflict_speed_cap", 10
        )
        self.allow_pass_pub = self.create_publisher(
            Bool, "/v2v/conflict_allows_overtake", 10
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/v2v/rosbot_pose", 10)
        self.pred_pub = self.create_publisher(Path, "/v2v/predicted_path", 10)
        self.speed_pub = self.create_publisher(Float32, "/v2v/rosbot_speed", 10)
        self.gap_pub = self.create_publisher(Float32, "/v2v/gap", 10)
        self.on_path_pub = self.create_publisher(Bool, "/v2v/on_path", 10)
        self.stats_pub = self.create_publisher(String, "/v2v/stats", 10)

        # The MPC already tracks its own index along this very trajectory,
        # with continuity, and publishes it. Prefer that over re-deriving it
        # here: on this track two lanes run 0.31 m apart for ~45 waypoints,
        # which no independent position+heading projection can disambiguate
        # (observed in sim: the receiver sat at index 102 while the car was
        # really at 330 — an 11.4 m gap error). TF projection stays as the
        # fallback for when the MPC is not running.
        self.mpc_idx = None
        self.mpc_idx_time = None
        self.create_subscription(
            Int32, "/current_path_idx", self.mpc_idx_callback, 10
        )

        # The overtake state machine owns the hold decision; this node owns
        # the socket. Keeping them apart means the decision stays testable
        # without a network and the link stays the receiver's concern.
        self.create_subscription(
            Bool, "/v2v/hold_request", self.hold_request_callback, 10
        )

        # Our own commanded speed, for the closing-rate half of the
        # prediction. The MPC's output is the honest source: it is what the
        # car will actually do next, not what we hope it does.
        self.ego_speed = 0.0
        self.create_subscription(
            Twist, "/cmd_vel_nav", self.ego_cmd_callback, 10
        )

        # Shared state between the socket thread and the ROS timer.
        self.lock = threading.Lock()
        self.latest_raw = None          # (bytes, arrival monotonic time)
        self.rx_count = 0
        self.parse_errors = 0
        self.seq_drops = 0

        self.last_seq = -1
        self.last_parsed = None         # validated + transformed state
        self.last_arrival = None        # monotonic arrival of last_parsed
        self.prev_arrival = None
        self.last_gap = float("nan")
        self.last_on_path = False
        self.last_blocked = None        # None = sender cannot tell us
        self.last_detour_intent = False

        self.conflict_horizon = float(gp("conflict_horizon_sec"))
        self.conflict_step = float(gp("conflict_step_sec"))
        self.conflict_radius = float(gp("conflict_radius_m"))
        self.qcar_free_speed = float(gp("qcar_free_speed"))
        self.yield_speed = float(gp("yield_speed"))
        self.give_way_speed = float(gp("give_way_speed"))
        self.lead_pass_speed_max = float(gp("lead_pass_speed_max"))
        self.last_encounter = None
        self.last_response = None
        self.prev_encounter_kind = v2v_conflict.NONE
        # Previous projection indices — seed the windowed nearest search so a
        # self-adjacent loop cannot flip the projection across the track.
        self.prev_rosbot_idx = None
        self.prev_qcar_idx = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(0.2)

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        self.create_timer(0.05, self._process)      # 20 Hz
        self.create_timer(1.0, self._publish_stats)
        command_rate = float(gp("command_rate_hz"))
        if self.tx_sock is not None and command_rate > 0.0:
            self.create_timer(1.0 / command_rate, self._transmit_command)

        self.get_logger().info(
            f"V2V receiver up: udp://{self.bind_ip}:{self.bind_port}, "
            f"expecting '{self.expected_id}', stale after {self.stale_sec}s"
        )

    # ------------------------------------------------------------------
    def _rx_loop(self):
        """Socket thread: keep only the newest datagram."""
        while rclpy.ok():
            try:
                data, _addr = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                self.latest_raw = (data, time.monotonic())
                self.rx_count += 1

    # ------------------------------------------------------------------
    @staticmethod
    def _yaw_to_quat(yaw):
        import math
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def mpc_idx_callback(self, msg):
        self.mpc_idx = int(msg.data)
        self.mpc_idx_time = time.monotonic()

    def ego_cmd_callback(self, msg):
        self.ego_speed = abs(float(msg.linear.x))

    def _ego_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (
            t.transform.translation.x, t.transform.translation.y, yaw
        )

    def _evaluate_conflict(self, parsed):
        """Classify the encounter and publish the response.

        Runs on every accepted packet. When nothing is predicted this costs a
        single horizon comparison and publishes NONE — the common case must
        stay cheap and must not perturb the controller at all.
        """
        ego_pose = self._ego_pose()
        qcar_idx = self._qcar_idx()
        if ego_pose is None or self.projector is None or qcar_idx is None:
            return

        # Our own horizon follows the route we intend to drive; a
        # straight-line guess would leave the road on every bend.
        ego_horizon = sample_along_path(
            self.projector.xy,
            self.projector.yaw,
            self.projector.arc_length,
            self.projector.total_length,
            float(self.projector.arc_length[qcar_idx]),
            self.ego_speed,
            self.conflict_horizon,
            self.conflict_step,
            loop=self.projector.loop,
        )
        # The other vehicle told us its own horizon; believe that over any
        # model of it, and only fall back when it sent none.
        other_horizon = resample_horizon(
            parsed["predicted"],
            parsed["horizon_dt"],
            self.conflict_horizon,
            self.conflict_step,
        )
        if ego_horizon is None or other_horizon is None:
            return

        encounter = classify_encounter(
            ego_pose=ego_pose,
            ego_speed=self.ego_speed,
            other_pose=(parsed["x"], parsed["y"], parsed["yaw"]),
            other_speed=abs(parsed["v"]),
            signed_gap_m=self.last_gap,
            ego_horizon=ego_horizon,
            other_horizon=other_horizon,
            step_s=self.conflict_step,
            conflict_radius=self.conflict_radius,
            on_path=self.last_on_path,
        )
        response = resolve_encounter(
            encounter,
            ego_speed=self.ego_speed,
            other_speed=abs(parsed["v"]),
            free_speed=self.qcar_free_speed,
            yield_speed=self.yield_speed,
            give_way_speed=self.give_way_speed,
            lead_pass_speed_max=self.lead_pass_speed_max,
        )

        self.last_encounter = encounter
        self.last_response = response

        self.encounter_pub.publish(String(data=json.dumps({
            "kind": encounter.kind,
            "action": response.action,
            "t_conflict_s": (
                None if math.isinf(encounter.time_to_conflict)
                else round(encounter.time_to_conflict, 2)
            ),
            "min_separation_m": (
                None if math.isinf(encounter.min_separation)
                else round(encounter.min_separation, 3)
            ),
            "closing_mps": round(encounter.closing_speed, 3),
            "speed_cap": (
                None if math.isinf(response.speed_cap)
                else round(response.speed_cap, 3)
            ),
            "allow_overtake": response.allow_overtake,
            "hold_other": response.hold_other,
            "reason": response.reason,
        })))
        # inf is not representable in Float32 semantics the MPC can use, so
        # "no cap" is published as a negative sentinel the consumer ignores.
        self.speed_cap_pub.publish(Float32(data=(
            -1.0 if math.isinf(response.speed_cap)
            else float(response.speed_cap)
        )))
        self.allow_pass_pub.publish(Bool(data=bool(response.allow_overtake)))

        if encounter.kind != self.prev_encounter_kind:
            self.prev_encounter_kind = encounter.kind
            if encounter.is_conflict:
                self.get_logger().info(
                    f"V2V encounter {encounter.kind} -> {response.action}: "
                    f"{response.reason}"
                )
            else:
                self.get_logger().info("V2V encounter cleared")

    def hold_request_callback(self, msg):
        requested = bool(msg.data)
        if requested and not self.hold_requested:
            self.hold_started = time.monotonic()
            self.hold_capped = False
        elif not requested:
            self.hold_started = None
            self.hold_capped = False
        self.hold_requested = requested

    def _hold_active(self):
        """Requested hold, bounded by max_hold_sec.

        The cap exists because a hold stops a robot that is otherwise driving
        correctly. If QCar's state machine wedges mid-pass, the ROSbot must
        recover on its own rather than wait forever in a live lane.
        """
        if not self.hold_requested or self.hold_started is None:
            return False
        if (time.monotonic() - self.hold_started) >= self.max_hold_sec:
            if not self.hold_capped:
                self.hold_capped = True
                self.get_logger().error(
                    f"V2V hold exceeded {self.max_hold_sec:.0f}s and was "
                    "released; QCar did not complete its pass"
                )
            return False
        return True

    def _transmit_command(self):
        """Send the current hold/proceed instruction at the command rate.

        Sent unconditionally rather than on change: every datagram is a
        complete instruction carrying its own ttl, so loss costs at most one
        period and no acknowledgement or retransmission is needed.
        """
        if self.tx_sock is None:
            return
        hold = self._hold_active()
        self.holding_pub.publish(Bool(data=bool(hold)))
        try:
            packet = pack_command(
                self.command_vehicle_id,
                self.expected_id,
                self.command_seq,
                time.time(),
                hold,
                self.command_ttl_sec,
            )
        except ValueError as e:
            self.get_logger().error(
                f"Invalid V2V command config: {e}", throttle_duration_sec=5.0
            )
            return
        try:
            self.tx_sock.sendto(packet, (self.command_ip, self.command_port))
            self.command_tx += 1
        except OSError as e:
            self.command_errors += 1
            self.get_logger().warn(
                f"V2V command send failed: {e}", throttle_duration_sec=5.0
            )
        self.command_seq += 1

    def _qcar_idx(self):
        if self.projector is None:
            return None

        # Authoritative: the MPC's own index, if it is fresh.
        if (
            self.mpc_idx is not None
            and (time.monotonic() - self.mpc_idx_time) < 1.0
            and 0 <= self.mpc_idx < self.projector.n
        ):
            self.prev_qcar_idx = self.mpc_idx
            return self.mpc_idx

        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        idx = self.projector.closest_idx(
            t.transform.translation.x,
            t.transform.translation.y,
            hint=self.prev_qcar_idx,
            heading=yaw,
        )
        self.prev_qcar_idx = idx
        return idx

    def _process(self):
        with self.lock:
            raw = self.latest_raw
            self.latest_raw = None

        if raw is not None:
            data, arrival = raw
            try:
                parsed = parse_packet(data, expected_id=self.expected_id)
            except PacketError as e:
                with self.lock:
                    self.parse_errors += 1
                self.get_logger().warn(
                    f"Dropped V2V packet: {e}", throttle_duration_sec=2.0
                )
                parsed = None

            if parsed is not None:
                seq = parsed["seq"]
                # Accept increasing seq; a large backwards jump means the
                # broadcaster restarted — resync instead of deadlocking.
                if seq <= self.last_seq and self.last_seq - seq < 1000:
                    self.seq_drops += 1
                else:
                    self.last_seq = seq
                    # Transform into the QCar map frame.
                    x, y, yaw = se2_apply(
                        self.frame_tx, self.frame_ty, self.frame_tyaw,
                        parsed["x"], parsed["y"], parsed["yaw"],
                    )
                    parsed["x"], parsed["y"], parsed["yaw"] = x, y, yaw
                    parsed["predicted"] = se2_apply_array(
                        self.frame_tx, self.frame_ty, self.frame_tyaw,
                        parsed["predicted"],
                    )
                    self.prev_arrival = self.last_arrival
                    self.last_parsed = parsed
                    self.last_arrival = arrival
                    self._on_new_state(parsed, arrival)

        # Freshness + publications every tick.
        fresh = (
            self.last_parsed is not None
            and self.last_parsed["localized"]
            and (time.monotonic() - self.last_arrival) < self.stale_sec
        )

        self.alive_pub.publish(Bool(data=bool(fresh)))
        if fresh:
            self.gap_pub.publish(Float32(data=float(self.last_gap)))
            self.on_path_pub.publish(Bool(data=bool(self.last_on_path)))
            # A schema 1 sender reports blocked=None. Publishing False there
            # would assert a clear lane we were never told about, so the
            # topic stays silent and consumers keep their "unknown" default.
            if self.last_blocked is not None:
                self.blocked_pub.publish(Bool(data=bool(self.last_blocked)))
            self.detour_pub.publish(
                Bool(data=bool(self.last_detour_intent))
            )

    def _on_new_state(self, parsed, arrival):
        now = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = parsed["x"]
        pose_msg.pose.position.y = parsed["y"]
        qx, qy, qz, qw = self._yaw_to_quat(parsed["yaw"])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "map"
        for px, py, pyaw in parsed["predicted"]:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            qx, qy, qz, qw = self._yaw_to_quat(float(pyaw))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)
        self.pred_pub.publish(path_msg)

        self.speed_pub.publish(Float32(data=float(abs(parsed["v"]))))

        # Project onto the QCar reference path.
        gap = float("nan")
        lat = 0.0
        on_path = False
        qcar_idx = None
        rosbot_idx = None
        if self.projector is not None:
            rosbot_idx = self.projector.closest_idx(
                parsed["x"], parsed["y"],
                hint=self.prev_rosbot_idx,
                heading=parsed["yaw"],
            )
            self.prev_rosbot_idx = rosbot_idx
            # Cast numpy scalars to native types: they leak into JSON stats
            # and ROS messages otherwise.
            lat = float(self.projector.lateral_offset(
                parsed["x"], parsed["y"], rosbot_idx
            ))
            on_path = bool(abs(lat) < self.lane_half_width)
            qcar_idx = self._qcar_idx()
            if qcar_idx is not None:
                gap = float(
                    self.projector.signed_gap_along(qcar_idx, rosbot_idx)
                )

        self.last_gap = gap
        self.last_on_path = on_path
        self.last_blocked = parsed["blocked"]
        self.last_detour_intent = bool(parsed["detour_intent"])

        # Gap and on_path are set above; the conflict layer consumes both.
        self._evaluate_conflict(parsed)

        if self.log_fh is not None:
            age_gap_ms = (
                (arrival - self.prev_arrival) * 1000.0
                if self.prev_arrival is not None
                else -1.0
            )
            self.log_fh.write(
                f"{time.time():.3f},{parsed['seq']},{age_gap_ms:.0f},"
                f"{parsed['x']:.3f},{parsed['y']:.3f},{parsed['yaw']:.3f},"
                f"{parsed['v']:.3f},{int(parsed['localized'])},"
                f"{int(parsed['moving'])},"
                f"{-1 if qcar_idx is None else qcar_idx},"
                f"{-1 if rosbot_idx is None else rosbot_idx},"
                f"{gap:.3f},{lat:.3f},{int(on_path)}\n"
            )
            self.log_fh.flush()

    def _publish_stats(self):
        with self.lock:
            rx = self.rx_count
            errs = self.parse_errors
            drops = self.seq_drops
        age = (
            time.monotonic() - self.last_arrival
            if self.last_arrival is not None
            else -1.0
        )
        stats = {
            "rx": rx,
            "parse_errors": errs,
            "seq_drops": drops,
            "age_s": round(age, 3),
            "gap": (
                round(self.last_gap, 3)
                if math.isfinite(self.last_gap)
                else None
            ),
            "on_path": self.last_on_path,
            "blocked": self.last_blocked,
            "detour_intent": self.last_detour_intent,
            "hold_active": self._hold_active(),
            "command_tx": self.command_tx,
            "command_errors": self.command_errors,
        }
        self.stats_pub.publish(String(data=json.dumps(stats)))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
        if self.tx_sock is not None:
            try:
                self.tx_sock.close()
            except OSError:
                pass
        if self.log_fh is not None:
            self.log_fh.close()


def main():
    rclpy.init()
    node = V2VReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
