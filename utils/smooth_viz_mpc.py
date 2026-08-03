#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def fit_bspline(xs, ys, smoothing):
    s = smoothing * len(xs)
    tck, u = splprep([xs, ys], s=s, per=False, k=3)
    return tck, u


def compute_arc_length(tck):
    u_dense = np.linspace(0.0, 1.0, 10000)
    xs_d, ys_d = splev(u_dense, tck)

    dx = np.diff(xs_d)
    dy = np.diff(ys_d)

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    return u_dense, arc


def resample_uniform(tck, spacing):
    u_dense, arc = compute_arc_length(tck)

    total_len = arc[-1]
    print(f"[INFO] Total path length: {total_len:.2f} m")

    n_pts = max(10, int(total_len / spacing))
    arc_uniform = np.linspace(0.0, total_len, n_pts)

    u_uniform = np.interp(arc_uniform, arc, u_dense)

    xs_out, ys_out = splev(u_uniform, tck)

    return np.array(xs_out), np.array(ys_out), u_uniform


def compute_yaw_and_curvature(tck, u_uniform):
    dx, dy = splev(u_uniform, tck, der=1)
    ddx, ddy = splev(u_uniform, tck, der=2)

    dx = np.array(dx)
    dy = np.array(dy)
    ddx = np.array(ddx)
    ddy = np.array(ddy)

    yaw = np.arctan2(dy, dx)
    yaw = np.unwrap(yaw)

    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom < 1e-9] = 1e-9

    curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    return yaw, curvature


def visualize(xs_raw, ys_raw, xs_spl, ys_spl, yaw, curvature, spacing, plot_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Path Smoothing Analysis", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(xs_raw, ys_raw, "o-", markersize=3, linewidth=1, alpha=0.6, label=f"Raw ({len(xs_raw)} pts)")
    ax.plot(xs_spl, ys_spl, "-", linewidth=2, alpha=0.9, label=f"Spline ({len(xs_spl)} pts)")
    ax.plot(xs_spl[0], ys_spl[0], "g^", markersize=10, label="Start")
    ax.plot(xs_spl[-1], ys_spl[-1], "rs", markersize=10, label="End")
    ax.set_title("Raw vs Spline Path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    s = np.arange(len(yaw)) * spacing

    ax = axes[0, 1]
    ax.plot(s, np.degrees(yaw), linewidth=1.5)
    ax.set_title("Unwrapped Yaw Along Path")
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Yaw (deg)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(s, curvature, linewidth=1.5)
    ax.set_title("Curvature Along Path")
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Curvature κ (1/m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sc = ax.scatter(xs_spl, ys_spl, c=curvature, s=8)
    plt.colorbar(sc, ax=ax, label="Curvature κ (1/m)")
    ax.plot(xs_spl[0], ys_spl[0], "g^", markersize=10, label="Start")
    ax.plot(xs_spl[-1], ys_spl[-1], "rs", markersize=10, label="End")
    ax.set_title("Path Colored by Curvature")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved -> {plot_path}")
    plt.show()


def save_npy(xs, ys, yaw, curvature, output_path):
    yaw = np.unwrap(yaw)

    data = np.column_stack([
        xs,
        ys,
        yaw,
        curvature
    ])

    np.save(output_path, data)

    print(f"[INFO] Saved {len(xs)} waypoints -> {output_path}")
    print(f"[INFO] Array shape: {data.shape}")
    print("[INFO] Columns: x, y, yaw_unwrapped, curvature")


def main():
    parser = argparse.ArgumentParser(
        description="B-spline smoothing, uniform resampling, yaw, curvature, and visualization for MPC path"
    )

    parser.add_argument("--input", required=True, help="Input .npy path: Nx2, Nx3, or Nx4")
    parser.add_argument("--output", required=True, help="Output .npy path: Nx4")
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--spacing", type=float, default=0.03)
    parser.add_argument("--plot", default="/home/nvidia/ros2_ws/path_analysis.png")
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    data = np.load(args.input)

    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError("Input path must be Nx2, Nx3, or Nx4")

    xs_raw = data[:, 0]
    ys_raw = data[:, 1]

    print(f"[INFO] Loaded {len(xs_raw)} raw waypoints from {args.input}")
    print(f"[INFO] Smoothing factor: {args.smoothing}")
    print(f"[INFO] Point spacing: {args.spacing * 100:.1f} cm")

    # Remove duplicate or almost-duplicate points
    clean_x = [xs_raw[0]]
    clean_y = [ys_raw[0]]

    for i in range(1, len(xs_raw)):
        d = np.hypot(xs_raw[i] - clean_x[-1], ys_raw[i] - clean_y[-1])
        if d > 1e-4:
            clean_x.append(xs_raw[i])
            clean_y.append(ys_raw[i])

    xs_raw = np.array(clean_x)
    ys_raw = np.array(clean_y)

    if len(xs_raw) < 4:
        raise RuntimeError("Need at least 4 valid points for cubic B-spline")

    tck, _ = fit_bspline(xs_raw, ys_raw, args.smoothing)

    xs_spl, ys_spl, u_uniform = resample_uniform(tck, args.spacing)

    yaw, curvature = compute_yaw_and_curvature(tck, u_uniform)

    print(f"[INFO] Curvature min/max: {curvature.min():.3f} / {curvature.max():.3f} 1/m")
    print(f"[INFO] Yaw min/max: {np.degrees(yaw.min()):.1f} / {np.degrees(yaw.max()):.1f} deg")

    if not args.no_plot:
        visualize(
            xs_raw,
            ys_raw,
            xs_spl,
            ys_spl,
            yaw,
            curvature,
            args.spacing,
            args.plot
        )

    save_npy(xs_spl, ys_spl, yaw, curvature, args.output)

    print("[DONE]")


if __name__ == "__main__":
    main()