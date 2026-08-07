import math
import numpy as np


class PathUtils:

    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def load_trajectory(path_file):
        data = np.load(path_file)

        if data.ndim != 2 or data.shape[1] < 2:
            raise RuntimeError("Trajectory must be Nx2, Nx3, or Nx4")
        if len(data) < 3 or not np.all(np.isfinite(data)):
            raise RuntimeError(
                "Trajectory must contain at least three finite waypoints"
            )

        if data.shape[1] == 2:
            data = PathUtils.add_yaw_and_curvature(data)

        elif data.shape[1] == 3:
            xy = data[:, 0:2]
            curvature = PathUtils.compute_curvature(xy)

            data = np.column_stack([
                data[:, 0],
                data[:, 1],
                data[:, 2],
                curvature
            ])

        else:
            # Recompute rather than trust the stored column. Route files
            # written by the recorder carry an unsigned, twice-finite-
            # differenced curvature, which is the wrong sign convention for
            # feasible_overtake_offset and signed_curvature_preview and is
            # dominated by recording noise besides. The geometry in columns
            # 0-1 is authoritative; column 3 is derived from it.
            data = np.column_stack([
                data[:, 0],
                data[:, 1],
                data[:, 2],
                PathUtils.compute_curvature(data[:, 0:2]),
            ])

        data[:, 2] = np.unwrap(data[:, 2])

        return data.astype(float)

    @staticmethod
    def add_yaw_and_curvature(xy):
        yaw = []

        for i in range(len(xy)):

            if i < len(xy) - 1:
                dx = xy[i + 1, 0] - xy[i, 0]
                dy = xy[i + 1, 1] - xy[i, 1]

            else:
                dx = xy[i, 0] - xy[i - 1, 0]
                dy = xy[i, 1] - xy[i - 1, 1]

            yaw.append(math.atan2(dy, dx))

        yaw = np.unwrap(np.array(yaw))

        curvature = PathUtils.compute_curvature(xy)

        return np.column_stack([
            xy[:, 0],
            xy[:, 1],
            yaw,
            curvature
        ])

    # Exactly one @staticmethod.  Doubling it wraps the descriptor in itself,
    # and the attribute access below then yields the inner staticmethod
    # object rather than the function.  That object is callable from Python
    # 3.10 onward, so the mistake passes on a modern dev machine and raises
    # TypeError on the car's Python 3.8 -- where it takes out route loading,
    # and with it the whole MPC.
    @staticmethod
    def _fit_derivative_kernel(window, polyorder, deriv):
        """Convolution kernel for a derivative of a local polynomial fit.

        Savitzky-Golay differentiation: least-squares fit a polynomial over
        ``window`` samples and differentiate that, instead of differencing
        the samples themselves.  Same cost (one convolution), far less noise
        amplification.
        """
        half = window // 2
        t = np.arange(-half, half + 1, dtype=float)
        design = np.vander(t, polyorder + 1, increasing=True)
        coefficients = np.linalg.pinv(design)[deriv]
        return coefficients * float(math.factorial(deriv))

    @staticmethod
    def compute_curvature(xy, window=15, polyorder=3):
        """Signed path curvature, positive for a left turn.

        **Signed** because a lateral offset does not treat the two bend
        directions alike: offsetting left shrinks ``1 - kappa*d`` on a left
        bend, tightening the passing line, and grows it on a right bend,
        loosening it.  ``feasible_overtake_offset`` and
        ``signed_curvature_preview`` both depend on that distinction and were
        previously handed magnitudes, so every right-hand bend -- 23% of this
        route -- was scored as though passing left made it tighter.
        Consumers that want magnitude already take ``abs()`` at the call site
        (the speed limiter, and ``overtake_curvature_preview``).

        **Fitted** rather than finite-differenced because ``np.gradient``
        applied twice to 5 cm waypoints returns noise rather than geometry.
        This route crosses itself, which makes that measurable: waypoints 190
        and 303 are 1.8 cm apart on the same piece of floor and the old
        estimator gave them kappa +1.46 and -0.31 -- opposite directions of
        turn.  Median disagreement across all 114 such pairs was 0.33 1/m
        against a passing-width decision threshold of 0.98, so the gate was
        mostly measuring recording noise.
        """
        points = np.asarray(xy, dtype=float)
        x, y = points[:, 0], points[:, 1]
        n = len(points)

        window = int(window)
        if window % 2 == 0:
            window += 1
        # Degrade gracefully on paths shorter than the fit window.
        window = max(5, min(window, n if n % 2 else n - 1))
        polyorder = int(min(polyorder, window - 2))

        half = window // 2
        # A closed route must wrap or the seam gets edge-clamped curvature;
        # an open one has no far side to borrow from.
        closed = bool(
            np.hypot(x[0] - x[-1], y[0] - y[-1])
            <= 2.0 * np.median(np.hypot(np.diff(x), np.diff(y)))
        )
        pad = "wrap" if closed else "edge"
        xp = np.pad(x, half, mode=pad)
        yp = np.pad(y, half, mode=pad)

        first = PathUtils._fit_derivative_kernel(window, polyorder, 1)
        second = PathUtils._fit_derivative_kernel(window, polyorder, 2)

        # np.convolve reverses the kernel; correlation is what we want.
        dx = np.convolve(xp, first[::-1], mode="valid")
        dy = np.convolve(yp, first[::-1], mode="valid")
        ddx = np.convolve(xp, second[::-1], mode="valid")
        ddy = np.convolve(yp, second[::-1], mode="valid")

        # Guard only genuinely coincident waypoints. The floor has to be
        # relative: this denominator scales as (waypoint spacing)^3, so a
        # fixed one silently rescales the answer on any path finer than it
        # assumed. At 0.05 m spacing it is 1.2e-4 and the old absolute 1e-6
        # never bit; at 0.008 m it is 4.9e-7 and clamping there halved every
        # curvature on the path.
        speed_squared = dx * dx + dy * dy
        typical = float(np.median(speed_squared))
        speed_squared = np.maximum(
            speed_squared,
            1e-6 * typical if typical > 0.0 else 1e-12,
        )

        curvature = (dx * ddy - dy * ddx) / np.power(speed_squared, 1.5)

        return np.nan_to_num(
            curvature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    @staticmethod
    def closest_point(pose, trajectory, previous_idx, global_search=False,
                      window_back=10, window_forward=120):
        """Nearest waypoint, searched in a window around the previous index.

        ``window_forward`` bounds how far ahead the reference may jump in one
        step. It must exceed the per-step advance (speed * dt / spacing --
        about 3 waypoints at 2 m/s on a 5 cm path) but stay well under the
        index separation between branches of a self-crossing route. Too wide
        and the reference can hop to a parallel branch that happens to be
        nearby in space: the car then reads as on-heading but laterally
        displaced, which is what trips the tracking safety stop with a large
        position error and a near-zero yaw error.
        """
        previous_idx = int(np.clip(previous_idx, 0, len(trajectory) - 1))
        window_back = max(1, int(window_back))
        window_forward = max(2, int(window_forward))

        def find_best(start, end):
            candidates = trajectory[start:end]
            distances = np.hypot(
                pose[0] - candidates[:, 0],
                pose[1] - candidates[:, 1],
            )
            yaw_errors = np.abs(
                np.arctan2(
                    np.sin(pose[2] - candidates[:, 2]),
                    np.cos(pose[2] - candidates[:, 2]),
                )
            )
            # Heading disambiguates self-intersections without allowing a
            # nearby, opposite-direction branch to steal the reference.
            scores = distances + 0.6 * yaw_errors
            relative = int(np.argmin(scores))
            return start + relative, float(distances[relative])

        if global_search:
            # Explicit one-shot acquisition allows startup at any valid
            # waypoint.  It must not be inferred from previous_idx == 0:
            # loop resets also set that index and need to stay at the seam.
            best, _ = find_best(0, len(trajectory))
            return best

        start = max(0, previous_idx - window_back)
        end = min(len(trajectory), previous_idx + window_forward)
        best, distance = find_best(start, end)

        if previous_idx > 0 and distance > 0.75 and end < len(trajectory):
            # Recover after a localization jump, but never regress into a
            # completed part of the path. Loop reset is handled explicitly.
            #
            # The recovery search is BOUNDED. Searching to the end of the
            # trajectory defeats the window this function just applied: on a
            # self-crossing route the nearest point by distance is often a
            # parallel branch hundreds of indices away, and the monotonic
            # clamp below then makes that error permanent. Measured on the
            # 20260802 track, branches pass within 0.02-0.26 m of each other,
            # so an unbounded search reliably hops. Observed as idx 286 ->
            # 685 with the reference stuck 1.49 m away and the yaw error
            # still small -- the failure this docstring warns about.
            #
            # A bounded search still crosses a genuine localization jump,
            # which is at most a few metres. If nothing closer exists inside
            # the bound the previous index is kept, the position error stays
            # large, and the tracking safety stop halts the car. That is the
            # honest outcome: the reference really is unknown.
            recovery_end = min(
                len(trajectory),
                previous_idx + 2 * window_forward,
            )
            candidate, candidate_distance = find_best(
                previous_idx,
                recovery_end,
            )
            if candidate_distance < distance:
                best = candidate

        return max(best, previous_idx)

    @staticmethod
    def step_indices_for_speed(speed, dt, spacing):
        """
        Convert a target speed [m/s] into an equivalent number of
        path-array indices to advance per MPC stage, given the
        waypoint spacing [m] the path was resampled at and the
        controller timestep dt [s].

        Negative speed is treated as a magnitude for reverse-path lookahead;
        zero speed returns zero advancement.
        """
        speed = abs(float(speed))
        spacing = max(spacing, 1e-6)

        # A zero requested speed must not advance the reference.  This helper
        # remains for curvature look-ahead code; build_reference() below uses
        # continuous arc length and therefore does not quantize to indices.
        if speed <= 1e-9:
            return 0

        return max(1, int(round((speed * dt) / spacing)))

    @staticmethod
    def build_reference(
            trajectory,
            idx,
            horizon,
            dt,
            spacing,
            target_v,
            loop_path=False):
        """
        Build an interpolated ``(x, y, yaw)`` reference at distances
        ``k * abs(target_v) * dt`` along the trajectory.

        This is deliberately continuous in arc length.  Advancing at least
        one whole waypoint per stage made a 5 cm path behave like a 0.625 m/s
        reference at an 80 ms controller period, even when target_v was zero.
        At zero speed every stage now stays on the current waypoint.

        For a canonical closed path, ``loop_path=True`` lets the horizon wrap
        through the final-to-first segment instead of piling every remaining
        stage onto the last waypoint.  If the file repeats its first point as
        its final row, the zero-length duplicate seam is handled naturally.
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if (
            trajectory.ndim != 2
            or len(trajectory) < 2
            or trajectory.shape[1] < 3
        ):
            raise ValueError(
                "trajectory must contain at least two (x, y, yaw) rows"
            )
        if horizon < 1:
            return np.empty(0, dtype=float)
        if dt < 0.0 or spacing <= 0.0:
            raise ValueError(
                "dt must be non-negative and spacing must be positive"
            )

        idx = int(np.clip(idx, 0, len(trajectory) - 1))
        segment_lengths = np.linalg.norm(
            np.diff(trajectory[:, :2], axis=0), axis=1
        )
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        open_length = float(cumulative[-1])

        seam_length = 0.0
        if loop_path:
            seam_length = float(np.linalg.norm(
                trajectory[0, :2] - trajectory[-1, :2]
            ))
        total_length = open_length + seam_length

        if total_length <= 1e-9:
            return np.tile(trajectory[idx, :3], horizon).astype(float)

        start_s = float(cumulative[idx])
        distance_per_stage = abs(float(target_v)) * float(dt)
        ref = np.empty((horizon, 3), dtype=float)

        def interpolate(p0, p1, fraction):
            result = np.empty(3, dtype=float)
            result[:2] = p0[:2] + fraction * (p1[:2] - p0[:2])
            yaw_delta = PathUtils.wrap_angle(float(p1[2] - p0[2]))
            result[2] = float(p0[2]) + fraction * yaw_delta
            return result

        for k in range(horizon):
            query_s = start_s + k * distance_per_stage
            if loop_path:
                query_s %= total_length
            else:
                query_s = min(query_s, open_length)

            if query_s >= open_length:
                # This is either the exact final point of an open path or the
                # explicit final-to-first seam of a closed path.
                if not loop_path or seam_length <= 1e-9:
                    ref[k] = trajectory[-1, :3]
                else:
                    fraction = (query_s - open_length) / seam_length
                    ref[k] = interpolate(
                        trajectory[-1], trajectory[0], fraction
                    )
                continue

            segment_idx = int(np.searchsorted(
                cumulative, query_s, side="right"
            ) - 1)
            segment_idx = int(np.clip(segment_idx, 0, len(trajectory) - 2))
            length = float(segment_lengths[segment_idx])
            if length <= 1e-9:
                ref[k] = trajectory[segment_idx, :3]
            else:
                fraction = (query_s - cumulative[segment_idx]) / length
                ref[k] = interpolate(
                    trajectory[segment_idx],
                    trajectory[segment_idx + 1],
                    fraction,
                )

        return ref.reshape(-1)

    @staticmethod
    def enforce_forward_speed_cap(
            command_v,
            target_v,
            active_cap=None,
            stop_epsilon=1e-6):
        """Return a forward command that cannot exceed its active limits.

        ``target_v`` is a real command ceiling, not only an MPC cost target.
        A non-negative ``active_cap`` (for example, the V2V governor output)
        is an additional ceiling.  Zero at either input is a hard zero; this
        also prevents a later minimum-speed rule from defeating a stop.
        """
        values = (command_v, target_v)
        if not all(np.isfinite(value) for value in values):
            return 0.0

        limit = max(0.0, float(target_v))
        if (
            active_cap is not None
            and np.isfinite(active_cap)
            and active_cap >= 0.0
        ):
            limit = min(limit, max(0.0, float(active_cap)))

        if limit <= stop_epsilon:
            return 0.0

        return float(np.clip(float(command_v), 0.0, limit))

    @staticmethod
    def v2v_following_cap(
            gap,
            lead_speed,
            stop_gap,
            follow_gap,
            follow_gain,
            soft_decel):
        """Compute the longitudinal cap for a vehicle ahead on our path.

        The stopping-distance and car-following limits are both enforced; the
        smaller one wins.  At or inside ``stop_gap`` this returns exactly zero
        regardless of the advertised lead-vehicle speed.
        """
        values = (
            gap,
            lead_speed,
            stop_gap,
            follow_gap,
            follow_gain,
            soft_decel,
        )
        if not all(np.isfinite(value) for value in values):
            return 0.0

        gap = float(gap)
        stop_gap = max(0.0, float(stop_gap))
        if gap <= stop_gap:
            return 0.0

        v_brake = math.sqrt(
            max(0.0, 2.0 * max(0.0, float(soft_decel)) * (gap - stop_gap))
        )
        v_follow = max(
            0.0,
            float(lead_speed)
            + max(0.0, float(follow_gain)) * (gap - float(follow_gap)),
        )
        return max(0.0, min(v_brake, v_follow))

    @staticmethod
    def predicted_ellipse_clear(
            vehicle_states,
            obstacle_states,
            longitudinal_radius,
            lateral_radius,
            stages=None,
            tolerance=1e-3):
        """Hard post-solve check for a moving elliptical keep-out zone."""
        vehicle_states = np.asarray(vehicle_states, dtype=float)
        obstacle_states = np.asarray(obstacle_states, dtype=float)
        if (
            vehicle_states.ndim != 2
            or obstacle_states.ndim != 2
            or vehicle_states.shape[1] < 2
            or obstacle_states.shape[1] < 3
            or longitudinal_radius <= 0.0
            or lateral_radius <= 0.0
        ):
            return False
        count = min(len(vehicle_states), len(obstacle_states))
        if stages is not None:
            count = min(count, max(0, int(stages)))
        if count <= 0:
            return False

        vehicle = vehicle_states[:count, :2]
        obstacle = obstacle_states[:count, :3]
        if not np.all(np.isfinite(vehicle)) or not np.all(np.isfinite(obstacle)):
            return False

        delta = vehicle - obstacle[:, :2]
        cosine = np.cos(obstacle[:, 2])
        sine = np.sin(obstacle[:, 2])
        longitudinal = cosine * delta[:, 0] + sine * delta[:, 1]
        lateral = -sine * delta[:, 0] + cosine * delta[:, 1]
        metric = (
            (longitudinal / float(longitudinal_radius)) ** 2
            + (lateral / float(lateral_radius)) ** 2
        )
        return bool(np.all(metric >= 1.0 - abs(float(tolerance))))

    @staticmethod
    def target_speed(
            trajectory,
            idx,
            horizon,
            vmax,
            vmin,
            dt=None,
            spacing=None,
            lookahead_speed=None,
            max_decel=None,
            brake_lookahead_m=None,
            curve_kappa_threshold=0.25):
        """
        Look ahead over the horizon for the tightest curvature and
        slow down accordingly. If dt/spacing/lookahead_speed are
        provided, the lookahead is spaced by predicted arc length
        (consistent with build_reference) instead of raw index;
        otherwise falls back to the original index-per-stage behavior.

        If max_decel and brake_lookahead_m (and spacing) are provided,
        this also scans further ahead than `horizon` for the next
        meaningfully curved section and caps the returned speed so the
        vehicle can actually decelerate down to that section's safe
        speed in time -- this is what prevents accelerating hard on a
        short straight that's too brief to brake back down from.
        """
        last_idx = len(trajectory) - 1

        if dt is not None and spacing is not None:
            ref_speed = (
                lookahead_speed if lookahead_speed is not None else vmax
            )
            step = PathUtils.step_indices_for_speed(ref_speed, dt, spacing)
        else:
            step = 1

        curvatures = []

        for k in range(horizon):
            j = min(idx + k * step, last_idx)
            curvatures.append(
                abs(trajectory[j][3])
            )

        kappa = max(curvatures)

        v_curve = vmax / (
            1.0 + 1.0 * kappa
        )

        v_curve = float(
            np.clip(
                v_curve,
                vmin,
                vmax
            )
        )

        if max_decel is None or brake_lookahead_m is None or spacing is None:
            return v_curve

        # Scan beyond the MPC horizon for the next point where curvature
        # exceeds curve_kappa_threshold, up to brake_lookahead_m meters
        # ahead (in real arc length, via spacing). If found, compute the
        # max speed that allows braking from here to there in time:
        #   v_now^2 <= v_target^2 + 2 * max_decel * distance
        max_lookahead_idx = int(brake_lookahead_m / max(spacing, 1e-6))

        next_curve_kappa = None
        dist_to_curve = None

        for n in range(max_lookahead_idx):
            j = min(idx + n, last_idx)
            k_here = abs(trajectory[j][3])

            if k_here >= curve_kappa_threshold:
                next_curve_kappa = k_here
                dist_to_curve = n * spacing
                break

        if next_curve_kappa is None:
            return v_curve

        v_at_curve = vmax / (1.0 + 1.0 * next_curve_kappa)
        v_at_curve = float(np.clip(v_at_curve, vmin, vmax))

        v_brake_limited = math.sqrt(
            max(
                v_at_curve ** 2 + 2.0 * max_decel * dist_to_curve,
                0.0,
            )
        )

        v_final = min(v_curve, v_brake_limited)

        return float(
            np.clip(
                v_final,
                vmin,
                vmax,
            )
        )

    @staticmethod
    def apply_offset(ref, horizon, offset):
        shifted = ref.copy()

        max_k = min(
            horizon,
            len(ref) // 3
        )

        for k in range(max_k):
            yaw = shifted[3 * k + 2]

            nx = -math.sin(yaw)
            ny = math.cos(yaw)

            shifted[3 * k] += offset * nx
            shifted[3 * k + 1] += offset * ny

        return shifted

    @staticmethod
    def apply_offset_profile(ref, offsets):
        """Shift each reference stage left by its own lateral offset.

        The yaw is rebuilt from the shifted points.  A lane change therefore
        contains the heading needed to reach the adjacent lane instead of a
        discontinuous parallel target whose yaw still points straight ahead.
        """
        points = np.asarray(ref, dtype=float).reshape(-1, 3).copy()
        offsets = np.asarray(offsets, dtype=float).reshape(-1)
        if len(points) != len(offsets):
            raise ValueError("one offset is required for every reference stage")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(offsets)):
            raise ValueError("reference and offsets must be finite")

        nominal_yaw = points[:, 2].copy()
        points[:, 0] -= np.sin(nominal_yaw) * offsets
        points[:, 1] += np.cos(nominal_yaw) * offsets

        if len(points) >= 2:
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            valid = np.hypot(dx, dy) > 1e-8
            points[valid, 2] = np.arctan2(dy[valid], dx[valid])

        return points.reshape(-1)


def feasible_overtake_offset(
        curvature,
        start_idx,
        horizon,
        closed_path,
        min_offset,
        max_offset,
        step,
        kappa_limit,
        allow_right=False,
        percentile=100.0):
    """Widest lane offset that stays steerable over the maneuver.

    Positive is left.  Returns 0.0 when no offset works.

    The road rule is two lanes, drive right, overtake left, so the search is
    left-only by default and ``allow_right`` stays off on the road.

    What this changes versus gating on one fixed offset is the WIDTH.  A pass
    does not need the full offset, it needs enough to clear the obstacle.
    Offsetting a path sideways scales its curvature by 1/(1 - kappa*d), and
    on this route -- 72% left-hand bends, minimum radius 0.53 m -- a full
    0.68 m offset drives that denominator towards zero on every tight bend,
    so the passing lane folds into a cusp and the pass was refused outright.
    Searching downward from ``max_offset`` instead takes the widest berth the
    geometry actually supports, which is what turns a single passable stretch
    into several (measured 4.0% of the lap at fixed width against 22.4%
    adaptive, and 1 zone against 6).

    The hairpins remain genuinely impassable: at radius 0.53 m an inside pass
    of even 0.35 m leaves 0.18 m of radius.  No parameter fixes that; only a
    route with gentler corners does.

    ``percentile`` selects the statistic taken over the maneuver window.  At
    the default 100.0 it is the maximum, which is what the fixed-width gate
    used and what every existing caller still gets.  The maximum makes the
    whole window only as passable as its single worst waypoint, and on a
    recorded route that waypoint is usually an artifact: the curvature fit
    spans 0.75 m, so one genuinely tight metre of road shows up in ~15
    consecutive samples, whereas a lone spike does not.  Requiring instead
    that 95% of the window be steerable keeps real corners out (they are
    broad in the window) while no longer letting one sample veto 2.75 m of
    otherwise open road -- measured on this route, 35.7% of the lap passable
    at max against 47.6% at p95, and 2 full-maneuver stretches against 3.
    """
    k = np.asarray(curvature, dtype=float).reshape(-1)
    if len(k) == 0 or horizon <= 0 or step <= 0.0:
        return 0.0
    if not (0.0 < min_offset <= max_offset) or kappa_limit <= 0.0:
        return 0.0
    percentile = float(percentile)
    if not 50.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [50, 100]")

    start_idx = int(np.clip(start_idx, 0, len(k) - 1))
    indices = start_idx + np.arange(int(horizon), dtype=int)
    if closed_path:
        indices %= len(k)
    else:
        indices = np.minimum(indices, len(k) - 1)

    window = k[indices]
    if not np.all(np.isfinite(window)):
        return 0.0

    width = float(max_offset)
    sides = (1.0, -1.0) if allow_right else (1.0,)
    while width >= float(min_offset) - 1e-9:
        for side in sides:
            offset = side * width
            denominator = 1.0 - window * offset
            # Near-zero denominator is the cusp: the offset path is
            # degenerate there, not merely tight.
            # A cusp is disqualifying wherever it appears, so this stays an
            # any() regardless of ``percentile``: it is a degeneracy of the
            # offset geometry, not a noisy sample to be averaged out.
            if np.any(np.abs(denominator) < 1e-3):
                continue
            steering = np.abs(window / denominator)
            demand = (
                np.max(steering)
                if percentile >= 100.0
                else float(np.percentile(steering, percentile))
            )
            if demand <= kappa_limit:
                return float(offset)
        width -= float(step)
    return 0.0
