#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def remove_stationary_points(df, min_dist):
    pts = []
    last = None

    for _, r in df.iterrows():
        p = np.array([r["x"], r["y"], r["theta"]], dtype=float)

        if last is None or np.hypot(p[0] - last[0], p[1] - last[1]) >= min_dist:
            pts.append(p)
            last = p

    return np.array(pts)


def fit_bspline(xs, ys, smoothing, closed):
    s = smoothing * len(xs)
    tck, _ = splprep([xs, ys], s=s, per=closed, k=3)
    return tck


def compute_arc_length(tck):
    u_dense = np.linspace(0.0, 1.0, 10000)
    x_dense, y_dense = splev(u_dense, tck)

    arc = np.concatenate([
        [0.0],
        np.cumsum(np.hypot(np.diff(x_dense), np.diff(y_dense)))
    ])

    return u_dense, arc


def resample_path(tck, spacing, closed):
    u_dense, arc = compute_arc_length(tck)
    total_len = arc[-1]

    n_pts = max(10, int(total_len / spacing))

    arc_uniform = np.linspace(
        0.0,
        total_len,
        n_pts,
        endpoint=not closed
    )

    u_uniform = np.interp(arc_uniform, arc, u_dense)
    xs, ys = splev(u_uniform, tck)

    return np.array(xs), np.array(ys), u_uniform, total_len


def compute_yaw_curvature(tck, u):
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    dx = np.array(dx)
    dy = np.array(dy)
    ddx = np.array(ddx)
    ddy = np.array(ddy)

    yaw = np.unwrap(np.arctan2(dy, dx))

    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom < 1e-9] = 1e-9

    curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = np.nan_to_num(curvature)

    # Remove spline boundary curvature artifacts
    if len(curvature) > 12:
        curvature[:5] = curvature[5]
        curvature[-5:] = curvature[-6]
    elif len(curvature) > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

    return yaw, curvature


def save_plot(raw, xs, ys, yaw, curvature, plot_path, spacing):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(raw[:, 0], raw[:, 1], "o-", markersize=2, alpha=0.5, label="Clean raw")
    axes[0, 0].plot(xs, ys, "-", linewidth=2, label="Final smoothed")
    axes[0, 0].plot(xs[0], ys[0], "g^", markersize=10, label="Start")
    axes[0, 0].plot(xs[-1], ys[-1], "rs", markersize=8, label="End")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    axes[0, 0].set_title("Path")

    s = np.arange(len(xs)) * spacing

    axes[0, 1].plot(s, np.degrees(yaw))
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Yaw")
    axes[0, 1].set_ylabel("deg")

    axes[1, 0].plot(s, curvature)
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Curvature")
    axes[1, 0].set_ylabel("1/m")

    sc = axes[1, 1].scatter(xs, ys, c=curvature, s=8)
    axes[1, 1].plot(xs, ys, linewidth=1, alpha=0.5)
    axes[1, 1].axis("equal")
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Path colored by curvature")
    plt.colorbar(sc, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Saved plot: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with x,y,theta")
    parser.add_argument("--output", required=True, help="Output NPY path")
    parser.add_argument("--plot", default="/home/nvidia/ros2_ws/final_path_plot.png")
    parser.add_argument("--spacing", type=float, default=0.03)
    parser.add_argument("--smoothing", type=float, default=0.03)
    parser.add_argument("--min-clean-dist", type=float, default=0.01)
    parser.add_argument("--closed", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"x", "y", "theta"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"CSV must contain columns: {required}")

    raw = remove_stationary_points(df, args.min_clean_dist)

    print(f"[INFO] Original CSV points: {len(df)}")
    print(f"[INFO] After cleaning: {len(raw)}")

    xs_raw = raw[:, 0]
    ys_raw = raw[:, 1]

    tck = fit_bspline(xs_raw, ys_raw, args.smoothing, args.closed)

    xs, ys, u, total_len = resample_path(tck, args.spacing, args.closed)

    yaw, curvature = compute_yaw_curvature(tck, u)

    data = np.column_stack([xs, ys, yaw, curvature])
    np.save(args.output, data)

    print(f"[INFO] Path length: {total_len:.2f} m")
    print(f"[INFO] Final waypoints: {len(data)}")
    print(f"[INFO] Curvature min/max: {curvature.min():.3f} / {curvature.max():.3f}")
    print(f"[INFO] Saved NPY: {args.output}")
    print("[INFO] Columns: x, y, yaw, curvature")

    save_plot(raw, xs, ys, yaw, curvature, args.plot, args.spacing)


if __name__ == "__main__":
    main()