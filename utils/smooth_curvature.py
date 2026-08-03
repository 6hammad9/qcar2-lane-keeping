#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def smooth_curvature(curvature, sigma, max_curvature):
    curvature_smooth = gaussian_filter1d(
        curvature,
        sigma=sigma
    )

    curvature_smooth = np.clip(
        curvature_smooth,
        0.0,
        max_curvature
    )

    return curvature_smooth


def visualize_curvature(curvature, curvature_smooth, plot_path):
    plt.figure(figsize=(12, 6))

    plt.plot(
        curvature,
        label="Original curvature",
        alpha=0.6
    )

    plt.plot(
        curvature_smooth,
        label="Smoothed curvature",
        linewidth=2
    )

    plt.xlabel("Waypoint index")
    plt.ylabel("Curvature κ (1/m)")
    plt.title("Curvature Before vs After Smoothing")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"[INFO] Curvature comparison saved -> {plot_path}")
    plt.show()


def visualize_path(x, y, curvature_smooth, plot_path):
    plt.figure(figsize=(8, 8))

    sc = plt.scatter(
        x,
        y,
        c=curvature_smooth,
        s=8
    )

    plt.plot(
        x,
        y,
        linewidth=1,
        alpha=0.6
    )

    plt.colorbar(
        sc,
        label="Smoothed curvature κ (1/m)"
    )

    plt.axis("equal")
    plt.grid(True)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Path Colored by Smoothed Curvature")

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"[INFO] Path curvature plot saved -> {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Smooth only curvature column of Nx4 MPC path without changing x/y/yaw"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input .npy file with columns x, y, yaw, curvature"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file with same x/y/yaw and smoothed curvature"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian smoothing sigma for curvature"
    )

    parser.add_argument(
        "--max-curvature",
        type=float,
        default=1.2,
        help="Maximum allowed curvature after smoothing"
    )

    parser.add_argument(
        "--curvature-plot",
        default="/home/nvidia/ros2_ws/curvature_comparison.png"
    )

    parser.add_argument(
        "--path-plot",
        default="/home/nvidia/ros2_ws/path_smoothed_curvature.png"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    args = parser.parse_args()

    data = np.load(args.input)

    if data.ndim != 2 or data.shape[1] < 4:
        raise RuntimeError(
            "Input must be Nx4: x, y, yaw, curvature"
        )

    x = data[:, 0]
    y = data[:, 1]
    yaw = data[:, 2]
    curvature = data[:, 3]

    curvature_smooth = smooth_curvature(
        curvature,
        args.sigma,
        args.max_curvature
    )

    output = np.column_stack([
        x,
        y,
        yaw,
        curvature_smooth
    ])

    np.save(
        args.output,
        output
    )

    print(f"[INFO] Saved output -> {args.output}")
    print(f"[INFO] Shape: {output.shape}")
    print("[INFO] x, y, yaw unchanged")
    print(f"[INFO] Original curvature min/max: {curvature.min():.3f} / {curvature.max():.3f}")
    print(f"[INFO] Smoothed curvature min/max: {curvature_smooth.min():.3f} / {curvature_smooth.max():.3f}")

    max_xy_change = np.max(
        np.hypot(
            output[:, 0] - data[:, 0],
            output[:, 1] - data[:, 1]
        )
    )

    max_yaw_change = np.max(
        np.abs(
            output[:, 2] - data[:, 2]
        )
    )

    print(f"[CHECK] Max x/y change: {max_xy_change:.12f}")
    print(f"[CHECK] Max yaw change: {max_yaw_change:.12f}")

    if not args.no_plot:
        visualize_curvature(
            curvature,
            curvature_smooth,
            args.curvature_plot
        )

        visualize_path(
            x,
            y,
            curvature_smooth,
            args.path_plot
        )

    print("[DONE]")


if __name__ == "__main__":
    main()