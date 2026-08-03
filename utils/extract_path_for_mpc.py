#!/usr/bin/env python3

import argparse
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage
except ImportError as e:
    print(f"[ERROR] Missing ROS 2 dependency: {e}")
    print("Run first:")
    print("  source /opt/ros/humble/setup.bash")
    sys.exit(1)

try:
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    print(f"[ERROR] Missing scipy dependency: {e}")
    print("Install with:")
    print("  sudo apt install python3-scipy")
    sys.exit(1)


# ============================================================
# Basic helpers
# ============================================================

def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(x, y, z, w):
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def circular_mean(angles):
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return math.atan2(s, c)


def multiply_transforms(t1, t2):
    """
    Chain two 2D transforms.

    t1: parent -> middle
    t2: middle -> child

    returns parent -> child
    """
    x1, y1, yaw1 = t1
    x2, y2, yaw2 = t2

    x = x1 + math.cos(yaw1) * x2 - math.sin(yaw1) * y2
    y = y1 + math.sin(yaw1) * x2 + math.cos(yaw1) * y2
    yaw = wrap_angle(yaw1 + yaw2)

    return x, y, yaw


# ============================================================
# Bag extraction
# ============================================================

def extract_poses_from_bag(bag_path):
    """
    Extract map -> base_link poses by chaining:

        map -> odom
        odom -> base_link

    Returns list of:

        t, x, y, yaw
    """
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=["/tf", "/tf_static"]
        )
    )

    tf_store = {}
    poses = []

    while reader.has_next():
        topic, rawdata, timestamp = reader.read_next()

        msg = deserialize_message(rawdata, TFMessage)

        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id

            tr = transform.transform.translation
            q = transform.transform.rotation

            tf_store[(parent, child)] = (
                tr.x,
                tr.y,
                quat_to_yaw(q.x, q.y, q.z, q.w),
            )

        t_map_odom = tf_store.get(("map", "odom"))
        t_odom_base = tf_store.get(("odom", "base_link"))

        if t_map_odom is not None and t_odom_base is not None:
            x, y, yaw = multiply_transforms(
                t_map_odom,
                t_odom_base,
            )

            poses.append((
                timestamp * 1e-9,
                x,
                y,
                yaw,
            ))

    poses.sort(key=lambda p: p[0])

    if len(poses) < 10:
        print("[ERROR] Could not extract enough poses.")
        print("[INFO] Available TF pairs found:")
        for pair in sorted(tf_store.keys()):
            print(f"  {pair[0]} -> {pair[1]}")
        raise RuntimeError("Not enough map->odom + odom->base_link transforms")

    print(f"[INFO] Extracted {len(poses)} raw poses from bag")

    start = poses[0]
    end = poses[-1]

    raw_gap = math.hypot(
        end[1] - start[1],
        end[2] - start[2],
    )

    raw_yaw_gap = wrap_angle(
        end[3] - start[3]
    )

    print(f"[INFO] Raw bag start: x={start[1]:.4f}, y={start[2]:.4f}, yaw={math.degrees(start[3]):.2f} deg")
    print(f"[INFO] Raw bag end:   x={end[1]:.4f}, y={end[2]:.4f}, yaw={math.degrees(end[3]):.2f} deg")
    print(f"[INFO] Raw bag end-start gap: {raw_gap:.4f} m")
    print(f"[INFO] Raw bag yaw difference: {math.degrees(raw_yaw_gap):.2f} deg")

    return poses


# ============================================================
# Stationary-window endpoint anchoring
# ============================================================

def find_stationary_window(poses, from_start, vel_threshold, min_duration_s,
                             max_scan_s, jump_threshold, speed_window_s=0.3):
    """
    Scan from either the start or end of `poses` for a contiguous
    stationary window lasting at least min_duration_s seconds, within
    the first/last max_scan_s seconds of the recording.

    Speed is estimated as displacement over a short window
    (speed_window_s, default 0.3s) rather than raw single-step
    velocity, since single-step velocity from finite-differenced,
    sensor-noisy position at typical TF publish rates is itself noisy
    enough to spuriously exceed a tight vel_threshold even when the
    robot is genuinely stationary. This also means a real position
    jump (e.g. a Cartographer pose-graph correction) tends to get
    naturally excluded: the windowed-speed check sees it as "motion"
    and splits the stationary period into separate runs around it,
    rather than averaging across the jump. The mid-window step-jump
    check below is a secondary safety net for jumps too small to be
    caught by the windowed-speed split but still larger than expected
    sensor noise.

    Returns (x, y, yaw, n_samples, had_jump) or None if no qualifying
    window is found.
    """
    times = np.array([p[0] for p in poses])
    xs = np.array([p[1] for p in poses])
    ys = np.array([p[2] for p in poses])
    yaws = np.array([p[3] for p in poses])

    t0 = times[0]
    t_end = times[-1]

    if from_start:
        mask_scan = times <= (t0 + max_scan_s)
        idxs = np.where(mask_scan)[0]
    else:
        mask_scan = times >= (t_end - max_scan_s)
        idxs = np.where(mask_scan)[0]

    if len(idxs) < 3:
        return None

    sub_t = times[idxs]
    sub_x = xs[idxs]
    sub_y = ys[idxs]
    sub_yaw = yaws[idxs]

    median_dt = float(np.median(np.diff(sub_t))) if len(sub_t) > 1 else 0.05
    median_dt = max(median_dt, 1e-3)
    win = max(1, int(round(speed_window_s / median_dt)))

    n = len(sub_t)
    speed = np.zeros(n)

    for i in range(n):
        j0 = max(0, i - win)
        j1 = min(n - 1, i + win)

        if j1 == j0:
            speed[i] = 0.0
            continue

        d = math.hypot(sub_x[j1] - sub_x[j0], sub_y[j1] - sub_y[j0])
        dt_window = sub_t[j1] - sub_t[j0]
        speed[i] = d / max(dt_window, 1e-3)

    stationary = speed < vel_threshold

    # find contiguous stationary runs
    runs = []
    run_start = None

    for i, s in enumerate(stationary):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            runs.append((run_start, i))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(stationary)))

    # pick the run closest to the actual start/end of the scan window
    # (index 0 for from_start, last index for from_end), since that's
    # the part of the recording closest to "parked at the arm"
    if not runs:
        return None

    if from_start:
        run_start, run_end = runs[0]
    else:
        run_start, run_end = runs[-1]

    duration = sub_t[run_end] - sub_t[run_start] if run_end < len(sub_t) else \
        sub_t[run_end - 1] - sub_t[run_start]

    if duration < min_duration_s:
        return None

    window_idx = np.arange(run_start, run_end + 1)
    window_idx = window_idx[window_idx < len(sub_x)]

    wx = sub_x[window_idx]
    wy = sub_y[window_idx]
    wyaw = sub_yaw[window_idx]

    # check for a mid-window jump
    if len(wx) > 2:
        step_dist = np.hypot(np.diff(wx), np.diff(wy))
        jump_idxs = np.where(step_dist > jump_threshold)[0]
    else:
        jump_idxs = np.array([], dtype=int)

    had_jump = len(jump_idxs) > 0

    if had_jump:
        jump_at = jump_idxs[-1] if from_start else jump_idxs[0]

        if from_start:
            # keep the portion AFTER the last jump (most settled)
            keep = slice(jump_at + 1, len(wx))
        else:
            # keep the portion BEFORE the first jump (most settled,
            # closest to end of recording without crossing the jump)
            keep = slice(0, jump_at + 1)

        wx = wx[keep]
        wy = wy[keep]
        wyaw = wyaw[keep]

    if len(wx) < 2:
        return None

    mean_x = float(np.mean(wx))
    mean_y = float(np.mean(wy))
    mean_yaw = circular_mean(wyaw)

    return mean_x, mean_y, mean_yaw, len(wx), had_jump


def estimate_seam_pose(poses, vel_threshold=0.03, min_duration_s=1.0,
                         max_scan_s=8.0, jump_threshold=0.05,
                         speed_window_s=0.3,
                         agreement_tolerance_m=0.05,
                         agreement_tolerance_deg=5.0):
    """
    Estimate the precise physical pose of the loop seam (e.g. the
    arm/pickup location) by averaging stationary windows at the start
    and end of the recording, since a closed loop starts and ends at
    the same physical place.

    Returns (x, y, yaw) to use as the anchored seam target, or None if
    no usable stationary window was found at either end (caller should
    fall back to raw first/last sample in that case).
    """
    start_est = find_stationary_window(
        poses,
        from_start=True,
        vel_threshold=vel_threshold,
        min_duration_s=min_duration_s,
        max_scan_s=max_scan_s,
        jump_threshold=jump_threshold,
        speed_window_s=speed_window_s,
    )

    end_est = find_stationary_window(
        poses,
        from_start=False,
        vel_threshold=vel_threshold,
        min_duration_s=min_duration_s,
        max_scan_s=max_scan_s,
        jump_threshold=jump_threshold,
    )

    if start_est is None and end_est is None:
        print("[WARN] No stationary window found at start or end of bag.")
        print("[WARN] Falling back to raw first/last sample for seam pose.")
        return None

    if start_est is not None:
        sx, sy, syaw, sn, sjump = start_est
        print(f"[INFO] Start stationary window: x={sx:.4f}, y={sy:.4f}, "
              f"yaw={math.degrees(syaw):.2f} deg, n={sn} samples, "
              f"jump_detected={sjump}")

    if end_est is not None:
        ex, ey, eyaw, en, ejump = end_est
        print(f"[INFO] End stationary window:   x={ex:.4f}, y={ey:.4f}, "
              f"yaw={math.degrees(eyaw):.2f} deg, n={en} samples, "
              f"jump_detected={ejump}")

    if start_est is not None and end_est is not None:
        sx, sy, syaw, _, _ = start_est
        ex, ey, eyaw, _, _ = end_est

        gap = math.hypot(ex - sx, ey - sy)
        yaw_gap = abs(math.degrees(wrap_angle(eyaw - syaw)))

        print(f"[INFO] Start/end stationary-window agreement: "
              f"gap={gap:.4f} m, yaw_gap={yaw_gap:.2f} deg")

        if gap > agreement_tolerance_m or yaw_gap > agreement_tolerance_deg:
            print(
                f"[WARN] Start and end stationary estimates disagree by "
                f"more than tolerance (pos tol={agreement_tolerance_m} m, "
                f"yaw tol={agreement_tolerance_deg} deg). This can mean "
                f"Cartographer's pose graph shifted between the start and "
                f"end of this recording, or the robot wasn't parked in "
                f"exactly the same spot. Using the END window estimate "
                f"(most recently optimized), but you should sanity check "
                f"this against the saved plot before trusting it."
            )

        # prefer end estimate: most time for Cartographer to have
        # optimized, and physically closest in time to when you placed
        # the robot at the arm before stopping the recording
        return ex, ey, eyaw

    if end_est is not None:
        ex, ey, eyaw, _, _ = end_est
        return ex, ey, eyaw

    sx, sy, syaw, _, _ = start_est
    return sx, sy, syaw


# ============================================================
# Distance downsampling
# ============================================================

def downsample_by_distance(poses, min_dist):
    kept = [poses[0]]

    for p in poses[1:]:
        dx = p[1] - kept[-1][1]
        dy = p[2] - kept[-1][2]

        if math.hypot(dx, dy) >= min_dist:
            kept.append(p)

    xs = np.array([p[1] for p in kept])
    ys = np.array([p[2] for p in kept])

    gap = np.hypot(
        xs[-1] - xs[0],
        ys[-1] - ys[0],
    )

    print(f"[INFO] After distance downsample: {len(kept)} points")
    print(f"[INFO] Downsampled end-start gap: {gap:.4f} m")

    return xs, ys


# ============================================================
# Cleaning
# ============================================================

def remove_duplicate_points(xs, ys, eps=1e-4):
    clean_x = [xs[0]]
    clean_y = [ys[0]]

    for i in range(1, len(xs)):
        d = np.hypot(
            xs[i] - clean_x[-1],
            ys[i] - clean_y[-1],
        )

        if d > eps:
            clean_x.append(xs[i])
            clean_y.append(ys[i])

    clean_x = np.array(clean_x)
    clean_y = np.array(clean_y)

    print(f"[INFO] After duplicate removal: {len(clean_x)} points")

    if len(clean_x) < 4:
        raise RuntimeError("Need at least 4 valid points for cubic B-spline")

    return clean_x, clean_y


# ============================================================
# Closed-loop B-spline
# ============================================================

def fit_closed_bspline(xs, ys, smoothing):
    gap = np.hypot(
        xs[-1] - xs[0],
        ys[-1] - ys[0],
    )

    print(f"[INFO] Input gap before B-spline: {gap:.4f} m")

    s = smoothing * len(xs)

    tck, u = splprep(
        [xs, ys],
        s=s,
        per=True,
        k=3,
    )

    return tck


def compute_arc_length(tck, dense_points=10000):
    u_dense = np.linspace(0.0, 1.0, dense_points)
    xs_d, ys_d = splev(u_dense, tck)

    dx = np.diff(xs_d)
    dy = np.diff(ys_d)

    arc = np.concatenate([
        [0.0],
        np.cumsum(np.hypot(dx, dy))
    ])

    return u_dense, arc


def resample_closed_spline(tck, spacing):
    u_dense, arc = compute_arc_length(tck)

    total_len = arc[-1]
    print(f"[INFO] Total spline path length: {total_len:.2f} m")

    n_pts = max(10, int(total_len / spacing))

    arc_uniform = np.linspace(
        0.0,
        total_len,
        n_pts,
        endpoint=False,
    )

    u_uniform = np.interp(
        arc_uniform,
        arc,
        u_dense,
    )

    xs_out, ys_out = splev(
        u_uniform,
        tck,
    )

    xs_out = np.array(xs_out)
    ys_out = np.array(ys_out)

    gap = np.hypot(
        xs_out[-1] - xs_out[0],
        ys_out[-1] - ys_out[0],
    )

    print(f"[INFO] Resampled points: {len(xs_out)}")
    print(f"[INFO] Final spline end-start waypoint gap: {gap:.4f} m")

    return xs_out, ys_out, u_uniform


# ============================================================
# Seam anchoring
# ============================================================

def anchor_seam_to_target(xs, ys, yaw, target_pose, blend_count):
    """
    Snap the seam (index 0, which is also the loop closure point for a
    periodic path) to an explicit target pose, blending smoothly into
    the surrounding `blend_count` points on each side so we don't
    introduce a sharp discontinuity in position/yaw right at the seam.

    xs, ys, yaw are modified in place (arrays of equal length, index 0
    == seam). Returns the corrected arrays.
    """
    if target_pose is None:
        return xs, ys, yaw

    tx, ty, tyaw = target_pose
    n = len(xs)

    blend_count = min(blend_count, n // 4)

    if blend_count < 1:
        xs[0] = tx
        ys[0] = ty
        yaw[0] = tyaw
        return xs, ys, yaw

    dx0 = tx - xs[0]
    dy0 = ty - ys[0]
    dyaw0 = wrap_angle(tyaw - yaw[0])

    print(f"[INFO] Seam anchor correction at index 0: "
          f"dx={dx0:.4f} m, dy={dy0:.4f} m, dyaw={math.degrees(dyaw0):.3f} deg")

    # blend forward from the seam (indices 0 .. blend_count-1)
    for i in range(blend_count):
        w = 1.0 - (i / float(blend_count))
        xs[i] += dx0 * w
        ys[i] += dy0 * w
        yaw[i] = wrap_angle(yaw[i] + dyaw0 * w)

    # blend backward from the seam (indices n-1 .. n-blend_count),
    # since for a periodic path the last point sits right before the
    # seam wraps back to index 0
    for i in range(blend_count):
        idx = n - 1 - i
        w = 1.0 - (i / float(blend_count))
        xs[idx] += dx0 * w
        ys[idx] += dy0 * w
        yaw[idx] = wrap_angle(yaw[idx] + dyaw0 * w)

    xs[0] = tx
    ys[0] = ty
    yaw[0] = tyaw

    gap = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    yaw_gap = math.degrees(wrap_angle(yaw[-1] - yaw[0]))

    print(f"[INFO] Post-anchor end-start waypoint gap: {gap:.4f} m")
    print(f"[INFO] Post-anchor yaw difference: {yaw_gap:.3f} deg")

    return xs, ys, yaw


# ============================================================
# Yaw and curvature
# ============================================================

def compute_yaw_and_curvature(tck, u_uniform):
    dx, dy = splev(u_uniform, tck, der=1)
    ddx, ddy = splev(u_uniform, tck, der=2)

    dx = np.array(dx)
    dy = np.array(dy)
    ddx = np.array(ddx)
    ddy = np.array(ddy)

    yaw = np.arctan2(dy, dx)
    yaw = np.unwrap(yaw)

    denom = np.power(
        dx * dx + dy * dy,
        1.5,
    )

    denom[denom < 1e-9] = 1e-9

    curvature = np.abs(
        dx * ddy - dy * ddx
    ) / denom

    curvature = np.nan_to_num(
        curvature,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    yaw_gap = wrap_angle(
        yaw[-1] - yaw[0]
    )

    print(f"[INFO] Yaw end-start difference: {math.degrees(yaw_gap):.2f} deg")
    print(f"[INFO] Curvature min/max before smoothing: {curvature.min():.3f} / {curvature.max():.3f} 1/m")

    return yaw, curvature


def remove_yaw_closure_error(yaw, u_uniform):
    """
    Distribute any residual yaw closure error evenly across the loop's
    arc-length parametrization, so the seam doesn't carry a heading
    discontinuity even after position has been anchored.
    """
    error = wrap_angle(yaw[-1] - yaw[0])
    correction = -error * u_uniform
    yaw_corrected = yaw + correction

    print(f"[INFO] Distributed yaw closure error: {math.degrees(error):.3f} deg "
          f"across full loop")

    return yaw_corrected


def smooth_curvature_loop(curvature, sigma, max_curvature):
    curvature_smooth = gaussian_filter1d(
        curvature,
        sigma=sigma,
        mode="wrap",
    )

    curvature_smooth = np.clip(
        curvature_smooth,
        0.0,
        max_curvature,
    )

    print(f"[INFO] Curvature min/max after smoothing: {curvature_smooth.min():.3f} / {curvature_smooth.max():.3f} 1/m")

    return curvature_smooth


# ============================================================
# Visualization
# ============================================================

def visualize_all(
    xs_raw,
    ys_raw,
    xs_spl,
    ys_spl,
    yaw,
    curvature,
    curvature_smooth,
    spacing,
    plot_path,
    target_pose=None,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    fig.suptitle(
        "Closed-Loop MPC Path Extraction and Smoothing",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(
        xs_raw,
        ys_raw,
        "o-",
        markersize=2,
        linewidth=1,
        alpha=0.5,
        label=f"Downsampled raw ({len(xs_raw)} pts)",
    )

    ax.plot(
        xs_spl,
        ys_spl,
        "-",
        linewidth=2,
        label=f"Closed B-spline ({len(xs_spl)} pts)",
    )

    ax.plot(
        xs_spl[0],
        ys_spl[0],
        "g^",
        markersize=10,
        label="Start/Seam",
    )

    ax.plot(
        xs_spl[-1],
        ys_spl[-1],
        "rs",
        markersize=8,
        label="End",
    )

    if target_pose is not None:
        ax.plot(
            target_pose[0],
            target_pose[1],
            "k*",
            markersize=14,
            label="Anchored arm target",
        )

    ax.set_title("Raw vs closed-loop smoothed path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    s_axis = np.arange(len(yaw)) * spacing

    ax = axes[0, 1]
    ax.plot(
        s_axis,
        np.degrees(yaw),
        linewidth=1.5,
    )
    ax.set_title("Unwrapped yaw")
    ax.set_xlabel("Arc length [m]")
    ax.set_ylabel("Yaw [deg]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(
        curvature,
        label="Original curvature",
        alpha=0.5,
    )
    ax.plot(
        curvature_smooth,
        label="Loop-aware smoothed curvature",
        linewidth=2,
    )
    ax.set_title("Curvature smoothing")
    ax.set_xlabel("Waypoint index")
    ax.set_ylabel("Curvature κ [1/m]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sc = ax.scatter(
        xs_spl,
        ys_spl,
        c=curvature_smooth,
        s=8,
    )
    ax.plot(
        xs_spl,
        ys_spl,
        linewidth=1,
        alpha=0.5,
    )
    ax.plot(
        xs_spl[0],
        ys_spl[0],
        "g^",
        markersize=10,
        label="Start/Seam",
    )
    ax.plot(
        xs_spl[-1],
        ys_spl[-1],
        "rs",
        markersize=8,
        label="End",
    )

    if target_pose is not None:
        ax.plot(
            target_pose[0],
            target_pose[1],
            "k*",
            markersize=14,
            label="Anchored arm target",
        )

    plt.colorbar(
        sc,
        ax=ax,
        label="Smoothed curvature κ [1/m]",
    )

    ax.set_title("Path colored by smoothed curvature")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"[INFO] Plot saved -> {plot_path}")

    # extra close-up plot of the seam region, since that's the
    # safety-critical area (arm clearance)
    zoom_path = plot_path.replace(".png", "_seam_zoom.png")

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    n_zoom = max(5, len(xs_spl) // 40)

    zoom_idx = list(range(len(xs_spl) - n_zoom, len(xs_spl))) + list(range(0, n_zoom))

    ax2.plot(
        xs_raw,
        ys_raw,
        "o",
        markersize=3,
        alpha=0.4,
        label="Raw points",
    )

    ax2.plot(
        [xs_spl[i] for i in zoom_idx],
        [ys_spl[i] for i in zoom_idx],
        "-o",
        markersize=4,
        linewidth=2,
        label="Spline near seam",
    )

    ax2.plot(
        xs_spl[0],
        ys_spl[0],
        "g^",
        markersize=14,
        label="Seam (idx 0)",
    )

    if target_pose is not None:
        ax2.plot(
            target_pose[0],
            target_pose[1],
            "k*",
            markersize=18,
            label="Anchored arm target",
        )

    pad = 0.4
    ax2.set_xlim(xs_spl[0] - pad, xs_spl[0] + pad)
    ax2.set_ylim(ys_spl[0] - pad, ys_spl[0] + pad)
    ax2.set_title("Seam / arm region close-up")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(zoom_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Seam close-up plot saved -> {zoom_path}")

    plt.show()


# ============================================================
# Save
# ============================================================

def save_final_path(xs, ys, yaw, curvature, output_path):
    data = np.column_stack([
        xs,
        ys,
        yaw,
        curvature,
    ])

    np.save(output_path, data)

    gap = np.hypot(
        xs[-1] - xs[0],
        ys[-1] - ys[0],
    )

    yaw_gap = wrap_angle(
        yaw[-1] - yaw[0]
    )

    print(f"[INFO] Saved final MPC path -> {output_path}")
    print(f"[INFO] Shape: {data.shape}")
    print("[INFO] Columns: x, y, yaw_unwrapped, curvature")
    print(f"[INFO] Final saved end-start waypoint gap: {gap:.4f} m")
    print(f"[INFO] Final saved yaw difference: {math.degrees(yaw_gap):.2f} deg")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract ROS2 bag path and create closed-loop MPC path with B-spline, yaw, and curvature"
    )

    parser.add_argument(
        "--bag",
        required=True,
        help="ROS2 bag folder path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file, Nx4: x, y, yaw, curvature",
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.02,
        help="Distance downsampling before spline [m]",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=0.03,
        help="Final waypoint spacing [m]",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="B-spline smoothing factor",
    )

    parser.add_argument(
        "--curvature-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for curvature smoothing",
    )

    parser.add_argument(
        "--max-curvature",
        type=float,
        default=1.2,
        help="Maximum clipped curvature [1/m]",
    )

    parser.add_argument(
        "--plot",
        default="/home/nvidia/ros2_ws/closed_mpc_path_analysis.png",
        help="Plot output path",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plots",
    )

    parser.add_argument(
        "--no-seam-anchor",
        action="store_true",
        help="Disable stationary-window seam anchoring (use raw spline seam as-is)",
    )

    parser.add_argument(
        "--seam-vel-threshold",
        type=float,
        default=0.03,
        help="Speed [m/s] below which the robot is considered stationary",
    )

    parser.add_argument(
        "--seam-speed-window",
        type=float,
        default=0.3,
        help="Window [s] over which speed is estimated for stationary detection (averages out position-noise jitter)",
    )

    parser.add_argument(
        "--seam-min-duration",
        type=float,
        default=1.0,
        help="Minimum stationary duration [s] required to trust a window",
    )

    parser.add_argument(
        "--seam-scan-window",
        type=float,
        default=8.0,
        help="How many seconds from the start/end of the bag to scan for a stationary window",
    )

    parser.add_argument(
        "--seam-jump-threshold",
        type=float,
        default=0.05,
        help="Position jump [m] within a stationary window treated as a Cartographer pose-graph correction",
    )

    parser.add_argument(
        "--seam-blend-points",
        type=int,
        default=15,
        help="Number of waypoints on each side of the seam to smoothly blend into the anchored target",
    )

    args = parser.parse_args()

    print("[START] Extracting closed-loop MPC path")
    print(f"[INFO] Bag: {args.bag}")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Min distance before spline: {args.min_dist:.3f} m")
    print(f"[INFO] Final spacing: {args.spacing:.3f} m")
    print(f"[INFO] B-spline smoothing: {args.smoothing}")
    print(f"[INFO] Curvature sigma: {args.curvature_sigma}")
    print(f"[INFO] Max curvature: {args.max_curvature}")

    poses = extract_poses_from_bag(args.bag)

    target_pose = None

    if not args.no_seam_anchor:
        target_pose = estimate_seam_pose(
            poses,
            vel_threshold=args.seam_vel_threshold,
            min_duration_s=args.seam_min_duration,
            max_scan_s=args.seam_scan_window,
            jump_threshold=args.seam_jump_threshold,
        )

    xs_raw, ys_raw = downsample_by_distance(
        poses,
        args.min_dist,
    )

    xs_raw, ys_raw = remove_duplicate_points(
        xs_raw,
        ys_raw,
    )

    tck = fit_closed_bspline(
        xs_raw,
        ys_raw,
        args.smoothing,
    )

    xs_spl, ys_spl, u_uniform = resample_closed_spline(
        tck,
        args.spacing,
    )

    yaw, curvature = compute_yaw_and_curvature(
        tck,
        u_uniform,
    )

    if target_pose is not None:
        xs_spl, ys_spl, yaw = anchor_seam_to_target(
            xs_spl,
            ys_spl,
            yaw,
            target_pose,
            args.seam_blend_points,
        )

        # curvature near the seam shifted slightly because we just
        # moved points there; recompute it from the corrected points
        # rather than trusting the pre-anchor spline-derivative values
        curvature = PathUtilsLocal_compute_curvature(xs_spl, ys_spl)

    yaw = remove_yaw_closure_error(yaw, u_uniform)

    curvature_smooth = smooth_curvature_loop(
        curvature,
        args.curvature_sigma,
        args.max_curvature,
    )

    save_final_path(
        xs_spl,
        ys_spl,
        yaw,
        curvature_smooth,
        args.output,
    )

    if not args.no_plot:
        visualize_all(
            xs_raw,
            ys_raw,
            xs_spl,
            ys_spl,
            yaw,
            curvature,
            curvature_smooth,
            args.spacing,
            args.plot,
            target_pose=target_pose,
        )

    print("[DONE] Final closed-loop MPC path is ready")


def PathUtilsLocal_compute_curvature(x, y):
    """
    Local curvature recompute (finite-difference, same formula used in
    path_utils.PathUtils.compute_curvature) for the post-anchor point
    set, since anchoring shifts points slightly near the seam and the
    pre-anchor spline-derivative curvature is no longer exactly correct
    there.
    """
    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom < 1e-6] = 1e-6

    curvature = np.abs(dx * ddy - dy * ddx) / denom

    return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    main()