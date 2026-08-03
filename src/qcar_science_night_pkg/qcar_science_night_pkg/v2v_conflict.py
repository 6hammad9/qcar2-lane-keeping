"""Predict whether the two vehicles will actually interact, and decide what
to do about it. Pure logic — no ROS imports, so it is testable offline.

The principle: a response is owed only when a conflict is *predicted*. Being
nearby, being slower, or being faster is not by itself a reason to act. Every
outcome below is therefore gated on a forward prediction that the two bodies
come within `conflict_radius` of each other inside the horizon.

Why two prediction models
-------------------------
Both robots follow the same recorded loop, which needs two different notions
of "close":

1. **Along-path (Frenet).** Same lane, one behind the other. The signed arc
   gap and the two speeds give the closing rate directly, it is loop-aware,
   and it does not care that the path bends. This covers following, being
   overtaken, and passing.

2. **Cartesian.** The recorded loop crosses *itself* — on this track two
   sections pass within 7 mm at waypoints 534 and 220. Two vehicles at
   completely different arc positions can therefore occupy the same physical
   point. The along-path model reports a huge gap there and sees nothing at
   all, so intersection conflicts need real x/y proximity over the horizon.

Neither model subsumes the other, which is why both run.

Asymmetry
---------
Only the QCar evaluates any of this. The outcomes it can choose are: cap its
own speed, decline to start a pass, yield its lane, or instruct the other
vehicle to hold. The other vehicle never optimizes — it drives its own route
and obeys a bounded hold lease.
"""

from dataclasses import dataclass
import math

import numpy as np

# Encounter kinds.
NONE = "NONE"
FOLLOWING = "FOLLOWING"           # we are closing on a slower vehicle ahead
OVERTAKEN = "OVERTAKEN"           # a faster vehicle is closing on us from behind
CROSSING = "CROSSING"             # paths intersect; neither is ahead of the other
HEAD_ON = "HEAD_ON"               # opposed headings on overlapping ground

# Actions.
PROCEED = "PROCEED"               # nothing owed
FOLLOW = "FOLLOW"                 # match/limit speed behind a lead
YIELD = "YIELD"                   # hold lane, do not obstruct, let them past
GIVE_WAY = "GIVE_WAY"             # slow to arrive after them at a crossing
HOLD_OTHER = "HOLD_OTHER"         # instruct the other vehicle to wait


@dataclass(frozen=True)
class Encounter:
    """What is predicted to happen between the two vehicles."""

    kind: str
    time_to_conflict: float       # seconds until closest approach, inf if none
    min_separation: float         # metres at closest approach
    closing_speed: float          # metres/second, positive = closing
    signed_gap: float             # along-path; +ve = other is ahead of us
    detail: str = ""

    @property
    def is_conflict(self):
        return self.kind != NONE


@dataclass(frozen=True)
class Response:
    """What the QCar does about it."""

    action: str
    speed_cap: float              # m/s; inf = no cap
    hold_other: bool              # issue a hold lease to the other vehicle
    allow_overtake: bool          # may the QCar start a lane change
    reason: str = ""


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def sample_constant_speed(pose, speed, horizon_s, step_s):
    """Straight-line extrapolation of a pose at constant speed.

    Used only as the fallback when a vehicle has broadcast no horizon of its
    own. It is deliberately crude: a vehicle that has told us its predicted
    path should always be believed over a model of it.
    """
    x, y, yaw = pose
    steps = max(1, int(round(horizon_s / step_s)) + 1)
    return np.array(
        [
            (x + speed * step_s * k * math.cos(yaw),
             y + speed * step_s * k * math.sin(yaw),
             yaw)
            for k in range(steps)
        ],
        dtype=float,
    )


def sample_along_path(
    path_xy, path_yaw, arc_length, total_length, start_arc, speed,
    horizon_s, step_s, loop=True,
):
    """Roll a vehicle forward along a reference path at constant speed.

    Better than a straight-line extrapolation for our own vehicle, because we
    know the route we intend to drive: on a bend the straight-line model
    leaves the road and would miss (or invent) a conflict.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    arc_length = np.asarray(arc_length, dtype=float)
    if len(path_xy) < 2 or len(arc_length) != len(path_xy):
        return None
    steps = max(1, int(round(horizon_s / step_s)) + 1)
    out = np.empty((steps, 3), dtype=float)
    for k in range(steps):
        arc = float(start_arc) + float(speed) * step_s * k
        if loop and total_length > 0.0:
            arc %= total_length
        else:
            arc = min(max(arc, 0.0), float(arc_length[-1]))
        idx = int(np.searchsorted(arc_length, arc))
        idx = min(max(idx, 0), len(path_xy) - 1)
        out[k, 0] = path_xy[idx, 0]
        out[k, 1] = path_xy[idx, 1]
        out[k, 2] = path_yaw[idx]
    return out


def resample_horizon(poses, source_step_s, horizon_s, step_s):
    """Re-time a broadcast horizon onto our own step, extending by holding.

    Holding the final pose is the conservative choice for the tail: it models
    the vehicle stopping where its prediction ran out, which can only make us
    more cautious, never less.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] < 3:
        return None
    steps = max(1, int(round(horizon_s / step_s)) + 1)
    out = np.empty((steps, 3), dtype=float)
    for k in range(steps):
        t = k * step_s
        src = t / source_step_s if source_step_s > 0 else 0.0
        lo = int(math.floor(src))
        if lo >= len(poses) - 1:
            out[k] = poses[-1]
            continue
        alpha = src - lo
        out[k, 0] = poses[lo, 0] + alpha * (poses[lo + 1, 0] - poses[lo, 0])
        out[k, 1] = poses[lo, 1] + alpha * (poses[lo + 1, 1] - poses[lo, 1])
        out[k, 2] = poses[lo, 2] + alpha * _wrap(
            poses[lo + 1, 2] - poses[lo, 2]
        )
    return out


def closest_approach(ego_poses, other_poses, step_s):
    """(min_separation, time_of_closest_approach, stage) over the horizon."""
    ego = np.asarray(ego_poses, dtype=float)
    other = np.asarray(other_poses, dtype=float)
    if ego.ndim != 2 or other.ndim != 2 or len(ego) == 0 or len(other) == 0:
        return float("inf"), float("inf"), -1
    count = min(len(ego), len(other))
    if count <= 0:
        return float("inf"), float("inf"), -1
    delta = ego[:count, :2] - other[:count, :2]
    if not np.all(np.isfinite(delta)):
        return float("inf"), float("inf"), -1
    distances = np.hypot(delta[:, 0], delta[:, 1])
    stage = int(np.argmin(distances))
    return float(distances[stage]), float(stage * step_s), stage


def classify_encounter(
    *,
    ego_pose,
    ego_speed,
    other_pose,
    other_speed,
    signed_gap_m,
    ego_horizon,
    other_horizon,
    step_s,
    conflict_radius,
    same_direction_tol_rad=math.radians(45.0),
    opposing_tol_rad=math.radians(135.0),
    on_path=True,
):
    """Decide which encounter, if any, the next few seconds hold.

    Returns an ``Encounter``. ``kind == NONE`` means no response is owed —
    that is the common case and it must stay cheap and quiet.
    """
    min_sep, t_min, _stage = closest_approach(
        ego_horizon, other_horizon, step_s
    )

    # The gate. Everything below only runs because the two bodies are
    # predicted to come within conflict_radius of each other.
    if not math.isfinite(min_sep) or min_sep > conflict_radius:
        return Encounter(
            kind=NONE,
            time_to_conflict=float("inf"),
            min_separation=min_sep,
            closing_speed=0.0,
            signed_gap=float(signed_gap_m)
            if signed_gap_m is not None and math.isfinite(signed_gap_m)
            else float("nan"),
            detail="no predicted approach inside the horizon",
        )

    heading_delta = abs(_wrap(float(other_pose[2]) - float(ego_pose[2])))
    gap = (
        float(signed_gap_m)
        if signed_gap_m is not None and math.isfinite(signed_gap_m)
        else float("nan")
    )

    if heading_delta >= opposing_tol_rad:
        return Encounter(
            kind=HEAD_ON,
            time_to_conflict=t_min,
            min_separation=min_sep,
            closing_speed=float(ego_speed) + float(other_speed),
            signed_gap=gap,
            detail=f"opposed headings ({math.degrees(heading_delta):.0f} deg)",
        )

    if heading_delta > same_direction_tol_rad:
        # Neither following nor opposing: the routes genuinely cross. On a
        # self-intersecting loop this is the case the along-path gap cannot
        # see, because the two arc positions are far apart.
        return Encounter(
            kind=CROSSING,
            time_to_conflict=t_min,
            min_separation=min_sep,
            closing_speed=float(ego_speed) + float(other_speed),
            signed_gap=gap,
            detail=f"paths cross ({math.degrees(heading_delta):.0f} deg)",
        )

    # Same direction. Which of us is in front decides who owes what, so an
    # unusable gap means we cannot tell and must not guess.
    if math.isnan(gap):
        return Encounter(
            kind=CROSSING,
            time_to_conflict=t_min,
            min_separation=min_sep,
            closing_speed=abs(float(ego_speed) - float(other_speed)),
            signed_gap=gap,
            detail="same heading but along-path order unknown",
        )

    if not on_path:
        # Same heading, close, but the other vehicle is not on our lane —
        # a parallel section. Nothing is owed beyond normal separation.
        return Encounter(
            kind=NONE,
            time_to_conflict=t_min,
            min_separation=min_sep,
            closing_speed=0.0,
            signed_gap=gap,
            detail="adjacent lane, not our path",
        )

    if gap >= 0.0:
        return Encounter(
            kind=FOLLOWING,
            time_to_conflict=t_min,
            min_separation=min_sep,
            closing_speed=float(ego_speed) - float(other_speed),
            signed_gap=gap,
            detail="slower vehicle ahead on our path",
        )

    return Encounter(
        kind=OVERTAKEN,
        time_to_conflict=t_min,
        min_separation=min_sep,
        closing_speed=float(other_speed) - float(ego_speed),
        signed_gap=gap,
        detail="faster vehicle closing from behind",
    )


def resolve_encounter(
    encounter,
    *,
    ego_speed,
    other_speed,
    free_speed,
    yield_speed,
    give_way_speed,
    lead_pass_speed_max,
    crossing_margin_s=2.0,
):
    """Choose the QCar's response to a classified encounter."""
    kind = encounter.kind

    if kind == NONE:
        return Response(
            action=PROCEED,
            speed_cap=float("inf"),
            hold_other=False,
            allow_overtake=True,
            reason="no predicted conflict",
        )

    if kind == OVERTAKEN:
        # A faster vehicle is completing a pass on us. The single worst thing
        # we can do is start a lane change into it, so overtaking is denied
        # outright. We also do not brake hard: that lengthens the time it
        # spends alongside. We hold our line at a steady, slightly reduced
        # speed and let it through. We do NOT hold it -- it is the one with
        # the right of way here, and stopping it would defeat the purpose.
        return Response(
            action=YIELD,
            speed_cap=min(float(yield_speed), float(free_speed)),
            hold_other=False,
            allow_overtake=False,
            reason=(
                f"yielding to faster vehicle closing at "
                f"{encounter.closing_speed:.2f} m/s"
            ),
        )

    if kind == FOLLOWING:
        # Only a lead we can actually out-run is worth passing. Otherwise we
        # simply follow it; the longitudinal governor owns the spacing.
        passable = float(other_speed) <= float(lead_pass_speed_max)
        return Response(
            action=FOLLOW,
            speed_cap=float("inf"),
            # Hold it only while we are committed to going around it, so it
            # cannot swerve into the lane we are occupying. The caller passes
            # that commitment in via the overtake state machine.
            hold_other=False,
            allow_overtake=passable,
            reason=(
                "lead slow enough to pass"
                if passable
                else f"lead at {other_speed:.2f} m/s, following"
            ),
        )

    if kind == CROSSING:
        # Whoever reaches the shared point first goes. If our predicted
        # arrival is not clearly earlier, we slow so that it becomes clearly
        # later -- an ambiguous crossing is resolved by giving way, never by
        # racing for it.
        we_are_clearly_first = (
            encounter.time_to_conflict > 0.0
            and float(ego_speed) > float(other_speed)
            and encounter.time_to_conflict < crossing_margin_s
        )
        if we_are_clearly_first:
            return Response(
                action=PROCEED,
                speed_cap=float("inf"),
                hold_other=False,
                allow_overtake=False,
                reason=(
                    f"clearing the crossing first "
                    f"({encounter.time_to_conflict:.1f} s)"
                ),
            )
        return Response(
            action=GIVE_WAY,
            speed_cap=float(give_way_speed),
            hold_other=False,
            allow_overtake=False,
            reason=(
                f"giving way at crossing in "
                f"{encounter.time_to_conflict:.1f} s"
            ),
        )

    if kind == HEAD_ON:
        # Neither vehicle can pass through the other and we cannot assume it
        # will move. Slow hard, refuse lane changes, and hold it so the two
        # do not continue closing while we sort the geometry out.
        return Response(
            action=GIVE_WAY,
            speed_cap=0.0,
            hold_other=True,
            allow_overtake=False,
            reason=(
                f"head-on at {encounter.min_separation:.2f} m in "
                f"{encounter.time_to_conflict:.1f} s"
            ),
        )

    return Response(
        action=PROCEED,
        speed_cap=float("inf"),
        hold_other=False,
        allow_overtake=True,
        reason=f"unhandled encounter {kind!r}",
    )
