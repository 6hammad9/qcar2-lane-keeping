#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def split_path(input_file, forward_output, reverse_output, plot_output):
    data = np.load(input_file)

    if data.ndim != 2 or data.shape[1] < 3:
        raise RuntimeError("Input must be Nx3 or Nx4")

    x = data[:, 0]
    y = data[:, 1]

    yaw = np.unwrap(data[:, 2])

    # Detect largest yaw jump.
    yaw_diff = np.abs(np.diff(yaw))

    split_idx = int(np.argmax(yaw_diff)) + 1

    forward = data[:split_idx].copy()
    reverse = data[split_idx:].copy()

    np.save(forward_output, forward)
    np.save(reverse_output, reverse)

    print(f"[INFO] Split index: {split_idx}")
    print(f"[INFO] Forward points: {len(forward)}")
    print(f"[INFO] Reverse points: {len(reverse)}")
    print(
        f"[INFO] Max yaw jump: "
        f"{np.degrees(yaw_diff[split_idx - 1]):.1f} deg"
    )

    visualize_split(
        data,
        forward,
        reverse,
        split_idx,
        yaw_diff,
        plot_output,
    )


def visualize_split(
    full_path,
    forward,
    reverse,
    split_idx,
    yaw_diff,
    plot_output,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        "Forward / Reverse Path Split Analysis",
        fontsize=14,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Full path with split marker
    # --------------------------------------------------

    ax = axes[0, 0]

    ax.plot(
        full_path[:, 0],
        full_path[:, 1],
        linewidth=2,
        label="Full path",
    )

    ax.scatter(
        full_path[0, 0],
        full_path[0, 1],
        marker="o",
        s=100,
        label="Start",
    )

    ax.scatter(
        full_path[-1, 0],
        full_path[-1, 1],
        marker="x",
        s=100,
        label="End",
    )

    ax.scatter(
        full_path[split_idx, 0],
        full_path[split_idx, 1],
        marker="s",
        s=120,
        label="Split point",
    )

    ax.set_title("Detected Split Point")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------
    # Forward vs reverse
    # --------------------------------------------------

    ax = axes[0, 1]

    ax.plot(
        forward[:, 0],
        forward[:, 1],
        linewidth=2,
        label="Forward segment",
    )

    ax.plot(
        reverse[:, 0],
        reverse[:, 1],
        linewidth=2,
        label="Reverse segment",
    )

    ax.scatter(
        forward[0, 0],
        forward[0, 1],
        marker="o",
        s=80,
        label="Forward start",
    )

    ax.scatter(
        forward[-1, 0],
        forward[-1, 1],
        marker="^",
        s=80,
        label="Forward end",
    )

    ax.scatter(
        reverse[0, 0],
        reverse[0, 1],
        marker="s",
        s=80,
        label="Reverse start",
    )

    ax.scatter(
        reverse[-1, 0],
        reverse[-1, 1],
        marker="x",
        s=80,
        label="Reverse end",
    )

    ax.set_title("Separated Forward and Reverse Paths")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

    # --------------------------------------------------
    # Yaw plot
    # --------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        np.degrees(np.unwrap(full_path[:, 2])),
        linewidth=1.5,
    )

    ax.axvline(
        split_idx,
        linestyle="--",
        linewidth=2,
        label="Split",
    )

    ax.set_title("Yaw Along Path")
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("Yaw (deg)")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------
    # Yaw difference
    # --------------------------------------------------

    ax = axes[1, 1]

    ax.plot(
        np.degrees(yaw_diff),
        linewidth=1.5,
    )

    ax.axvline(
        split_idx,
        linestyle="--",
        linewidth=2,
        label="Detected split",
    )

    ax.set_title("Yaw Difference Between Points")
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("Yaw Change (deg)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        plot_output,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"[INFO] Plot saved: {plot_output}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Split forward and reverse sections of a recorded trajectory."
    )

    parser.add_argument("--input", required=True)

    parser.add_argument(
        "--forward-output",
        required=True,
    )

    parser.add_argument(
        "--reverse-output",
        required=True,
    )

    parser.add_argument(
        "--plot",
        default="/home/nvidia/ros2_ws/split_preview.png",
    )

    args = parser.parse_args()

    split_path(
        input_file=args.input,
        forward_output=args.forward_output,
        reverse_output=args.reverse_output,
        plot_output=args.plot,
    )


if __name__ == "__main__":
    main()