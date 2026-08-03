"""Shared V2V logic — no ROS imports, so every function here is unit-testable
offline (see utils/v2v_selftest.py) and importable by the fake broadcaster.

Wire format
-----------
One UDP datagram per update, JSON-encoded, keys kept short to stay well under
a single MTU (~1400 B budget):

    s    int    schema version (must equal SCHEMA_VERSION)
    id   str    vehicle id, e.g. "rosbot3"
    q    int    monotonically increasing sequence number
    t    float  sender wall-clock stamp (informational only — freshness is
                judged by ARRIVAL time on the receiver, so unsynced clocks
                between the robots cannot cause false "fresh" states)
    loc  int    1 = sender is localized (pose fields valid), 0 = heartbeat only
    x,y  float  position in the sender's map frame [m]
    th   float  yaw [rad]
    v    float  speed estimate [m/s]
    ms   str    "M" moving / "S" stopped
    hdt  float  prediction step [s]
    p    list   flat [x0,y0,th0, x1,y1,th1, ...] predicted poses, stage 0 =
                current pose; typically 26 stages at 0.08 s = 2.0 s horizon

Schema 2 adds the sender's own obstacle situation. These are what let the
QCar distinguish "slower traffic" (no reason to pass) from "blocked traffic
that is about to swerve into the passing lane" (must be sequenced):

    bl   int    1 = an obstacle blocks the sender's lane, 0 = clear.
                ABSENT means the sender predates schema 2 — unknown, never
                assume clear.
    bd   float  range to that obstacle [m]
    di   int    1 = the sender intends to, or already does, detour LEFT
                around it. This is the conflicting maneuver: the ROSbot's
                own feasibility check only looks 0.2 m behind itself, so it
                cannot see a QCar closing in the passing lane.

Command channel (schema 2, QCar -> ROSbot, `pack_command`/`parse_command`).
The coordination is still asymmetric: only the QCar optimizes. The ROSbot
runs no solver and holds no opinion — it is simply told to hold or proceed:

    s    int    schema version
    id   str    issuing vehicle, e.g. "qcar2"
    to   str    target vehicle, e.g. "rosbot3"
    q    int    sequence number
    t    float  sender stamp (informational)
    c    str    "H" hold / "P" proceed
    ttl  float  seconds this command stays valid after ARRIVAL

`ttl` is the fail-safe. A hold is a bounded lease, re-issued at the command
rate, never a latch: if the link dies mid-pass the ROSbot resumes on its own
after ttl instead of being stranded, and a lost PROCEED costs at most ttl
seconds of standing still. Holding forever on link loss and releasing
instantly on link loss are both wrong; a short lease is neither.

Why UDP and not DDS: this lab's network corrupts Fast DDS discovery when
mixed ROS distros share a domain (confirmed twice: PROGRESS_README.md issue 3,
and again on 2026-08-01 — every node died with std::bad_alloc). Both robots
therefore keep their ROS graphs local (ROS_LOCALHOST_ONLY=1 or a private
ROS_DOMAIN_ID) and V2V rides a raw socket that DDS discovery cannot touch.
Packet loss is harmless: every datagram is a complete state refresh.
"""

import json
import math

import numpy as np

SCHEMA_VERSION = 2
# Schema 1 senders are still accepted: they simply carry no obstacle report,
# which decodes to blocked=None ("unknown"). Every consumer treats unknown as
# "not proven clear", so an un-upgraded ROSbot degrades to the old behavior
# rather than to an unsafe one.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
MAX_PACKET_BYTES = 1400
MAX_PREDICTED_STAGES = 40
MAX_ABS_POSITION_M = 500.0
MAX_ABS_SPEED_MPS = 5.0
MAX_COMMAND_TTL_SEC = 5.0

COMMAND_HOLD = "H"
COMMAND_PROCEED = "P"


class PacketError(ValueError):
    """Raised when an incoming datagram fails validation."""


def pack_state(
    vehicle_id,
    seq,
    stamp,
    localized,
    x,
    y,
    yaw,
    v,
    moving,
    horizon_dt,
    predicted,
    blocked=None,
    blocked_distance=None,
    detour_intent=False,
):
    """Build one datagram. `predicted` is an iterable of (x, y, yaw) tuples.

    `blocked` is tri-state: True/False report the sender's lane, None omits
    the field entirely so the receiver decodes it as unknown.

    Returns bytes guaranteed <= MAX_PACKET_BYTES (the predicted tail is
    truncated first if the packet would be oversized — the near-term stages
    are the safety-relevant ones).
    """
    predicted = [
        (round(float(px), 3), round(float(py), 3), round(float(pth), 3))
        for px, py, pth in predicted
    ][:MAX_PREDICTED_STAGES]

    while True:
        flat = [c for pt in predicted for c in pt]
        msg = {
            "s": SCHEMA_VERSION,
            "id": str(vehicle_id),
            "q": int(seq),
            "t": round(float(stamp), 3),
            "loc": 1 if localized else 0,
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "th": round(float(yaw), 3),
            "v": round(float(v), 3),
            "ms": "M" if moving else "S",
            "hdt": round(float(horizon_dt), 3),
            "p": flat,
        }
        if blocked is not None:
            msg["bl"] = 1 if blocked else 0
            msg["di"] = 1 if detour_intent else 0
            if blocked_distance is not None and math.isfinite(
                float(blocked_distance)
            ):
                msg["bd"] = round(float(blocked_distance), 3)
        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(data) <= MAX_PACKET_BYTES or not predicted:
            return data
        predicted = predicted[: max(1, len(predicted) - 4)]


def pack_command(vehicle_id, target_id, seq, stamp, hold, ttl_sec):
    """Build one QCar -> ROSbot command datagram (see module docstring)."""
    ttl = float(ttl_sec)
    if not math.isfinite(ttl) or not 0.0 < ttl <= MAX_COMMAND_TTL_SEC:
        raise ValueError(
            f"ttl_sec must be in (0, {MAX_COMMAND_TTL_SEC}]"
        )
    msg = {
        "s": SCHEMA_VERSION,
        "id": str(vehicle_id),
        "to": str(target_id),
        "q": int(seq),
        "t": round(float(stamp), 3),
        "c": COMMAND_HOLD if hold else COMMAND_PROCEED,
        "ttl": round(ttl, 3),
    }
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def parse_command(data, expected_target=None, expected_issuer=None):
    """Validate and decode one command datagram.

    Returns a dict with keys: id, target, seq, stamp, hold, ttl_sec.
    Raises PacketError on anything malformed. An unrecognised command word
    is an error rather than a default, so a corrupted byte can never be read
    as "proceed".
    """
    if len(data) > MAX_PACKET_BYTES:
        raise PacketError("oversized command datagram")

    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PacketError(f"not valid JSON: {e}")

    if not isinstance(msg, dict):
        raise PacketError("not a JSON object")
    if msg.get("s") not in SUPPORTED_SCHEMA_VERSIONS:
        raise PacketError(f"unsupported schema {msg.get('s')!r}")

    issuer = msg.get("id")
    target = msg.get("to")
    if not isinstance(issuer, str) or not issuer:
        raise PacketError("missing issuing vehicle id")
    if not isinstance(target, str) or not target:
        raise PacketError("missing target vehicle id")
    if expected_issuer is not None and issuer != expected_issuer:
        raise PacketError(f"unexpected issuer {issuer!r}")
    if expected_target is not None and target != expected_target:
        raise PacketError(f"command not addressed to us ({target!r})")

    word = msg.get("c")
    if word not in (COMMAND_HOLD, COMMAND_PROCEED):
        raise PacketError(f"unknown command {word!r}")

    seq = msg.get("q")
    stamp = msg.get("t")
    ttl = msg.get("ttl")
    for name, value in (("q", seq), ("t", stamp), ("ttl", ttl)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PacketError(f"field {name!r} not numeric")
        if not math.isfinite(float(value)):
            raise PacketError(f"field {name!r} not finite")
    if not 0.0 < float(ttl) <= MAX_COMMAND_TTL_SEC:
        raise PacketError(f"ttl {ttl} out of range")

    return {
        "id": issuer,
        "target": target,
        "seq": int(seq),
        "stamp": float(stamp),
        "hold": word == COMMAND_HOLD,
        "ttl_sec": float(ttl),
    }


def parse_packet(data, expected_id=None):
    """Validate and decode one datagram.

    Returns a dict with keys: id, seq, stamp, localized, x, y, yaw, v,
    moving, horizon_dt, predicted (np.ndarray (M, 3), M >= 1 when localized),
    blocked (True/False/None-for-unknown), blocked_distance, detour_intent.
    Raises PacketError on anything malformed — the caller drops the packet.
    """
    if len(data) > MAX_PACKET_BYTES * 4:
        raise PacketError("oversized datagram")

    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PacketError(f"not valid JSON: {e}")

    if not isinstance(msg, dict):
        raise PacketError("not a JSON object")
    if msg.get("s") not in SUPPORTED_SCHEMA_VERSIONS:
        raise PacketError(
            f"schema {msg.get('s')!r} not in {SUPPORTED_SCHEMA_VERSIONS}"
        )

    vid = msg.get("id")
    if not isinstance(vid, str) or not vid:
        raise PacketError("missing vehicle id")
    if expected_id is not None and vid != expected_id:
        raise PacketError(f"unexpected vehicle id {vid!r}")

    def _num(key, lo, hi):
        val = msg.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise PacketError(f"field {key!r} not numeric")
        val = float(val)
        if not math.isfinite(val) or not (lo <= val <= hi):
            raise PacketError(f"field {key!r}={val} out of range")
        return val

    seq = int(_num("q", 0, 2**53))
    stamp = _num("t", 0, 4e9)
    localized = msg.get("loc") == 1
    x = _num("x", -MAX_ABS_POSITION_M, MAX_ABS_POSITION_M)
    y = _num("y", -MAX_ABS_POSITION_M, MAX_ABS_POSITION_M)
    yaw = _num("th", -10.0, 10.0)
    v = _num("v", -MAX_ABS_SPEED_MPS, MAX_ABS_SPEED_MPS)
    horizon_dt = _num("hdt", 0.01, 1.0)
    moving = msg.get("ms") == "M"

    flat = msg.get("p")
    if not isinstance(flat, list) or len(flat) % 3 != 0:
        raise PacketError("predicted list malformed")
    if len(flat) > MAX_PREDICTED_STAGES * 3:
        raise PacketError("predicted list too long")
    for index, val in enumerate(flat):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise PacketError("predicted entry not numeric")
        limit = 10.0 if index % 3 == 2 else MAX_ABS_POSITION_M
        if not math.isfinite(float(val)) or abs(float(val)) > limit:
            field = "yaw" if index % 3 == 2 else "position"
            raise PacketError(f"predicted {field} entry out of range")

    predicted = np.array(flat, dtype=float).reshape(-1, 3)
    if predicted.shape[0] == 0:
        # Heartbeat or degenerate packet: fall back to the single pose.
        predicted = np.array([[x, y, yaw]], dtype=float)

    # Obstacle report (schema 2). Absent -> None, meaning "unknown", which is
    # NOT the same as "clear": a schema 1 sender simply cannot tell us.
    raw_blocked = msg.get("bl")
    if raw_blocked is None:
        blocked = None
        blocked_distance = None
        detour_intent = False
    else:
        if raw_blocked not in (0, 1):
            raise PacketError(f"field 'bl'={raw_blocked!r} not 0 or 1")
        blocked = raw_blocked == 1
        blocked_distance = None
        if "bd" in msg:
            blocked_distance = _num("bd", 0.0, MAX_ABS_POSITION_M)
        detour_intent = msg.get("di") == 1

    return {
        "id": vid,
        "seq": seq,
        "stamp": stamp,
        "localized": localized,
        "x": x,
        "y": y,
        "yaw": yaw,
        "v": v,
        "moving": moving,
        "horizon_dt": horizon_dt,
        "predicted": predicted,
        "blocked": blocked,
        "blocked_distance": blocked_distance,
        "detour_intent": detour_intent,
    }


def se2_apply(tx, ty, tyaw, x, y, yaw):
    """Apply a rigid 2-D transform (sender map frame -> receiver map frame).

    With both robots localized against the same map (the intended deployment)
    the transform is identity; the parameters exist for the two-map fallback.
    """
    c = math.cos(tyaw)
    s = math.sin(tyaw)
    return (
        tx + c * x - s * y,
        ty + s * x + c * y,
        math.atan2(math.sin(yaw + tyaw), math.cos(yaw + tyaw)),
    )


def se2_apply_array(tx, ty, tyaw, poses):
    """Vectorized se2_apply for an (M, 3) pose array."""
    poses = np.asarray(poses, dtype=float)
    c = math.cos(tyaw)
    s = math.sin(tyaw)
    out = np.empty_like(poses)
    out[:, 0] = tx + c * poses[:, 0] - s * poses[:, 1]
    out[:, 1] = ty + s * poses[:, 0] + c * poses[:, 1]
    out[:, 2] = np.arctan2(
        np.sin(poses[:, 2] + tyaw), np.cos(poses[:, 2] + tyaw)
    )
    return out


class PathProjector:
    """Project a world position onto a recorded reference path.

    Wraps the [x, y, yaw, curvature] trajectory the MPC follows and answers:
    which waypoint is closest, how far off the path laterally, and what is
    the along-path gap from one index to another (loop-aware).
    """

    def __init__(self, trajectory, spacing, loop=True):
        trajectory = np.asarray(trajectory, dtype=float)
        if (
            trajectory.ndim != 2
            or len(trajectory) < 2
            or trajectory.shape[1] < 3
            or not np.all(np.isfinite(trajectory[:, :3]))
        ):
            raise ValueError("trajectory must be (N, >=3) of x, y, yaw")
        self.xy = trajectory[:, 0:2].copy()
        self.yaw = trajectory[:, 2].copy()
        self.spacing = float(spacing)
        if not math.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("spacing must be positive and finite")
        self.loop = bool(loop)
        self.n = len(trajectory)

        # Gap estimates are safety inputs, so derive their distance from the
        # actual path geometry instead of multiplying an index difference by
        # a configured nominal spacing.  This remains correct for resampled,
        # uneven, and fractional-closure trajectories.
        segment_lengths = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.open_length = float(self.arc_length[-1])
        self.seam_length = (
            float(np.linalg.norm(self.xy[0] - self.xy[-1]))
            if self.loop
            else 0.0
        )
        self.total_length = self.open_length + self.seam_length

    def closest_idx(
        self,
        x,
        y,
        hint=None,
        window=60,
        reacquire_dist=0.5,
        heading=None,
        heading_weight=0.5,
    ):
        """Index of the nearest waypoint, robust on self-adjacent tracks.

        Two failure modes were observed in simulation on the real recorded
        path, both fixed here:

        1. A plain GLOBAL nearest search flips across the loop. This path
           passes within 7 mm of itself (waypoints 534 and 220), so a few cm
           of noise reported a 2.8 m gap when the true gap was 13.6 m.
           Fixed by searching a window seeded from `hint`.

        2. A window alone is NOT enough. Waypoints around 102 and 330 form
           parallel adjacent lanes 0.75-1.06 m apart for 59 consecutive
           waypoints. With a loose `reacquire_dist` the window locks onto the
           WRONG lane and happily tracks along it (observed: the receiver sat
           at index 102 while the car was really at 330, an 11.4 m error).
           Fixed by (a) `reacquire_dist` tightened below the lane separation,
           and (b) an optional `heading` gate, which separates sections that
           run parallel in opposite directions.

        `heading` is the vehicle's yaw. Candidates whose path tangent
        disagrees are penalised by `heading_weight * (1 - cos(err))` added to
        the squared distance, in m^2.
        """
        target = np.array([x, y])

        def cost(idxs):
            d = self.xy[idxs] - target
            sq = np.einsum("ij,ij->i", d, d)
            if heading is None:
                return sq, sq
            err = self.yaw[idxs] - heading
            penalty = heading_weight * (1.0 - np.cos(err))
            return sq + penalty, sq

        if hint is None:
            all_idx = np.arange(self.n)
            c, _ = cost(all_idx)
            return int(np.argmin(c))

        offsets = np.arange(-window, window + 1)
        idxs = int(hint) + offsets
        if self.loop:
            idxs %= self.n
        else:
            idxs = idxs[(idxs >= 0) & (idxs < self.n)]

        c, sq = cost(idxs)
        pick = int(np.argmin(c))
        best_local = int(idxs[pick])
        best_dist = float(np.sqrt(sq[pick]))

        if best_dist <= reacquire_dist:
            return best_local

        all_idx = np.arange(self.n)
        c_all, _ = cost(all_idx)
        return int(np.argmin(c_all))

    def distance_to_path(self, x, y, idx):
        """Euclidean distance from (x, y) to waypoint idx."""
        return float(math.hypot(x - self.xy[idx, 0], y - self.xy[idx, 1]))

    def lateral_offset(self, x, y, idx=None):
        """Signed lateral offset from the path at the closest waypoint
        (positive = left of the path direction)."""
        if idx is None:
            idx = self.closest_idx(x, y)
        dx = x - self.xy[idx, 0]
        dy = y - self.xy[idx, 1]
        yaw = self.yaw[idx]
        return -math.sin(yaw) * dx + math.cos(yaw) * dy

    def gap_along(self, idx_from, idx_to):
        """Along-path distance travelling forward from idx_from to idx_to.

        On a loop this is always in [0, path_length). On a non-loop path a
        target behind the source returns -1.0 (meaning: not ahead).
        """
        idx_from = int(idx_from)
        idx_to = int(idx_to)
        if not (0 <= idx_from < self.n and 0 <= idx_to < self.n):
            raise IndexError("path indices are out of range")

        gap = float(self.arc_length[idx_to] - self.arc_length[idx_from])
        if self.loop and gap < 0.0:
            gap += self.total_length
        elif not self.loop and gap < 0.0:
            return -1.0
        return gap

    def signed_gap_along(self, idx_from, idx_to):
        """Return signed along-path separation to a nearby target.

        Positive means ``idx_to`` is ahead and negative means it is behind.
        On a closed route, use the shortest signed separation.  V2V only
        acts inside a short interaction range, so that sign is unambiguous
        and can prove that an overtake has actually completed.
        """
        idx_from = int(idx_from)
        idx_to = int(idx_to)
        if not (0 <= idx_from < self.n and 0 <= idx_to < self.n):
            raise IndexError("path indices are out of range")

        gap = float(self.arc_length[idx_to] - self.arc_length[idx_from])
        if self.loop and self.total_length > 0.0:
            gap %= self.total_length
            if gap > 0.5 * self.total_length:
                gap -= self.total_length
        return gap

    def path_length(self):
        return self.total_length
