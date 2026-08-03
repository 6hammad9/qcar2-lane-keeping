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
        label="Start",
    )

    ax.plot(
        xs_spl[-1],
        ys_spl[-1],
        "rs",
        markersize=8,
        label="End",
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
        label="Start",
    )
    ax.plot(
        xs_spl[-1],
        ys_spl[-1],
        "rs",
        markersize=8,
        label="End",
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
        )

    print("[DONE] Final closed-loop MPC path is ready")


if __name__ == "__main__":
    main()