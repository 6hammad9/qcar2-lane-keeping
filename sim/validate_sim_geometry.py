#!/usr/bin/env python3
"""Validate canonical path identity, map clearance, lanes, and steering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np


SIM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SIM_DIR)
PACKAGE_ROOT = os.path.join(REPO_ROOT, "src", "qcar_science_night_pkg")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from qcar_science_night_pkg.path_map_validator import (  # noqa: E402
    _classify_pixels,
    _world_to_grid,
    load_map,
    validate_trajectory_against_map,
)


NORMAL_LANE_OFFSET_LEFT_M = 0.0
DIVIDER_OFFSET_LEFT_M = 0.21
OVERTAKE_LANE_OFFSET_LEFT_M = 0.47
COMMAND_OVERTAKE_OFFSET_LEFT_M = 0.52
ROAD_WIDTH_M = 0.96
ROAD_CENTER_OFFSET_LEFT_M = 0.5 * OVERTAKE_LANE_OFFSET_LEFT_M
EDGE_LINE_INSET_M = 0.020
RIGHT_EDGE_OFFSET_LEFT_M = (
    ROAD_CENTER_OFFSET_LEFT_M - 0.5 * ROAD_WIDTH_M + EDGE_LINE_INSET_M
)
LEFT_EDGE_OFFSET_LEFT_M = (
    ROAD_CENTER_OFFSET_LEFT_M + 0.5 * ROAD_WIDTH_M - EDGE_LINE_INSET_M
)


def load_canonical_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"x", "y", "theta"}.issubset(rows[0]):
        raise RuntimeError("canonical CSV must contain x,y,theta")
    return np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["theta"])] for row in rows],
        dtype=float,
    )


def offset_xy(xy: np.ndarray, offset_left_m: float) -> np.ndarray:
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    normal_left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    return xy + offset_left_m * normal_left


def corridor_report(xy: np.ndarray, map_data: dict, clearance_m: float) -> dict:
    rows, columns = _world_to_grid(xy[:, 0], xy[:, 1], map_data)
    height, width = map_data["pixels"].shape
    inside = (
        (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    )
    classes = np.full(len(xy), -1, dtype=np.int8)
    classes[inside] = _classify_pixels(
        map_data["pixels"][rows[inside], columns[inside]], map_data
    )

    radius_pixels = int(math.ceil(clearance_m / map_data["resolution"]))
    offsets = [
        (dr, dc)
        for dr in range(-radius_pixels, radius_pixels + 1)
        for dc in range(-radius_pixels, radius_pixels + 1)
        if dr * dr + dc * dc <= radius_pixels * radius_pixels
    ]
    clearance_ok = np.zeros(len(xy), dtype=bool)
    for index in np.flatnonzero(inside):
        is_clear = True
        for dr, dc in offsets:
            row = rows[index] + dr
            column = columns[index] + dc
            if not (0 <= row < height and 0 <= column < width):
                is_clear = False
                break
            classification = _classify_pixels(
                np.asarray([map_data["pixels"][row, column]]), map_data
            )[0]
            if classification != 1:
                is_clear = False
                break
        clearance_ok[index] = is_clear

    return {
        "point_count": int(len(xy)),
        "free_fraction": float(np.mean(classes == 1)),
        "unknown_fraction": float(np.mean(classes == 0)),
        "occupied_fraction": float(np.mean(classes == 2)),
        "outside_fraction": float(np.mean(classes == -1)),
        "clearance_m": float(clearance_m),
        "clearance_ok_fraction": float(np.mean(clearance_ok)),
        "ok": bool(np.all(classes == 1) and np.all(clearance_ok)),
    }


def validate(
    csv_path: str,
    npy_path: str,
    map_yaml: str,
    *,
    clearance_m: float = 0.10,
    max_steer_rad: float = 0.44,
) -> dict:
    canonical_csv = load_canonical_csv(csv_path)
    qcar = np.load(npy_path)
    failures = []

    if qcar.ndim != 2 or qcar.shape[1] != 4:
        raise RuntimeError("QCar canonical trajectory must be Nx4")
    identical = (
        canonical_csv.shape == qcar[:, :3].shape
        and np.array_equal(canonical_csv, qcar[:, :3])
    )
    if not identical:
        failures.append("ROSbot CSV and QCar NPY x/y/yaw are not identical")

    nominal = validate_trajectory_against_map(
        npy_path,
        map_yaml,
        clearance_m=clearance_m,
        max_unknown_fraction=0.0,
        max_clearance_violation_fraction=0.0,
        wheelbase_m=0.256,
        max_steer_rad=max_steer_rad,
        max_steering_violation_fraction=0.0,
        max_waypoint_spacing_m=0.075,
        loop_path=True,
        max_loop_gap_m=0.075,
    )
    failures.extend(f"nominal path: {failure}" for failure in nominal["failures"])

    xy = qcar[:, :2]
    map_data = load_map(map_yaml)
    corridors = {
        "normal_lane_center": corridor_report(
            offset_xy(xy, NORMAL_LANE_OFFSET_LEFT_M), map_data, clearance_m
        ),
        "lane_correction_right_limit": corridor_report(
            offset_xy(xy, -0.04), map_data, clearance_m
        ),
        "lane_correction_left_limit": corridor_report(
            offset_xy(xy, 0.04), map_data, clearance_m
        ),
        "overtake_lane_center": corridor_report(
            offset_xy(xy, OVERTAKE_LANE_OFFSET_LEFT_M), map_data, clearance_m
        ),
        "commanded_overtake_reference_limit": corridor_report(
            offset_xy(xy, COMMAND_OVERTAKE_OFFSET_LEFT_M),
            map_data,
            clearance_m,
        ),
        # Painted lines only need to remain in known free cells; the vehicle
        # envelope is checked at the two actual driving centers above.
        "dashed_divider": corridor_report(
            offset_xy(xy, DIVIDER_OFFSET_LEFT_M), map_data, 0.0
        ),
        "right_edge": corridor_report(
            offset_xy(xy, RIGHT_EDGE_OFFSET_LEFT_M), map_data, 0.0
        ),
        "left_edge": corridor_report(
            offset_xy(xy, LEFT_EDGE_OFFSET_LEFT_M), map_data, 0.0
        ),
    }
    for name, report in corridors.items():
        if not report["ok"]:
            failures.append(f"{name} leaves known free map space")

    yaw, curvature = periodic_geometry(xy)
    yaw_error = np.abs(np.arctan2(
        np.sin(qcar[:, 2] - yaw), np.cos(qcar[:, 2] - yaw)
    ))
    curvature_error = np.abs(qcar[:, 3] - curvature)
    if float(np.max(yaw_error)) > 1e-10:
        failures.append("stored yaw is inconsistent with periodic path tangent")
    if float(np.max(curvature_error)) > 1e-10:
        failures.append("stored curvature is inconsistent with periodic path geometry")

    return {
        "ok": not failures,
        "failures": failures,
        "canonical_files_identical": bool(identical),
        "lane_geometry_m": {
            "right_lane_width": (
                DIVIDER_OFFSET_LEFT_M - RIGHT_EDGE_OFFSET_LEFT_M
            ),
            "right_lane_midpoint": 0.5 * (
                DIVIDER_OFFSET_LEFT_M + RIGHT_EDGE_OFFSET_LEFT_M
            ),
            "nominal_from_right_lane_midpoint": (
                NORMAL_LANE_OFFSET_LEFT_M
                - 0.5 * (
                    DIVIDER_OFFSET_LEFT_M + RIGHT_EDGE_OFFSET_LEFT_M
                )
            ),
            "left_lane_width": (
                LEFT_EDGE_OFFSET_LEFT_M - DIVIDER_OFFSET_LEFT_M
            ),
            "overtake_from_left_lane_midpoint": (
                OVERTAKE_LANE_OFFSET_LEFT_M
                - 0.5 * (
                    LEFT_EDGE_OFFSET_LEFT_M + DIVIDER_OFFSET_LEFT_M
                )
            ),
        },
        "lane_offsets_left_m": {
            "normal": NORMAL_LANE_OFFSET_LEFT_M,
            "dashed_divider": DIVIDER_OFFSET_LEFT_M,
            "overtake_center": OVERTAKE_LANE_OFFSET_LEFT_M,
            "right_edge": RIGHT_EDGE_OFFSET_LEFT_M,
            "left_edge": LEFT_EDGE_OFFSET_LEFT_M,
        },
        "nominal": nominal,
        "corridors": corridors,
        "geometry_consistency": {
            "yaw_max_error_rad": float(np.max(yaw_error)),
            "curvature_max_error_1pm": float(np.max(curvature_error)),
        },
    }


def periodic_geometry(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous = np.roll(xy, 1, axis=0)
    following = np.roll(xy, -1, axis=0)
    derivative = 0.5 * (following - previous)
    second_derivative = following - 2.0 * xy + previous
    yaw = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0]))
    denominator = np.maximum(
        np.power(np.sum(derivative * derivative, axis=1), 1.5), 1e-9
    )
    curvature = np.abs(
        (
            derivative[:, 0] * second_derivative[:, 1]
            - derivative[:, 1] * second_derivative[:, 0]
        ) / denominator
    )
    return yaw, np.nan_to_num(curvature)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=os.path.join(SIM_DIR, "assets", "smoothed_trajectory.csv")
    )
    parser.add_argument(
        "--npy",
        default=os.path.join(SIM_DIR, "assets", "qcar_half_map_centerline.npy"),
    )
    parser.add_argument(
        "--map", default=os.path.join(SIM_DIR, "assets", "track_map.yaml")
    )
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--max-steer", type=float, default=0.44)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = validate(
        args.csv,
        args.npy,
        args.map,
        clearance_m=args.clearance,
        max_steer_rad=args.max_steer,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        nominal = report["nominal"]
        print(
            f"canonical identity: {'PASS' if report['canonical_files_identical'] else 'FAIL'}"
        )
        print(
            f"closed path: {nominal['point_count']} points, "
            f"{nominal['path_length_m'] + nominal['end_start_gap_m']:.3f} m, "
            f"seam {nominal['end_start_gap_m']:.3f} m"
        )
        print(
            f"steering: p99={nominal['required_steering_p99_rad']:.3f} rad, "
            f"max={nominal['required_steering_max_rad']:.3f} rad, "
            f"seam={nominal['loop_seam_required_steering_rad']:.3f} rad"
        )
        lane_geometry = report["lane_geometry_m"]
        print(
            "right lane: "
            f"width={lane_geometry['right_lane_width']:.3f} m, "
            "canonical-minus-midpoint="
            f"{lane_geometry['nominal_from_right_lane_midpoint']:+.4f} m"
        )
        print(
            "left lane: "
            f"width={lane_geometry['left_lane_width']:.3f} m, "
            "overtake-minus-midpoint="
            f"{lane_geometry['overtake_from_left_lane_midpoint']:+.4f} m"
        )
        for name, corridor in report["corridors"].items():
            print(
                f"{name}: free={corridor['free_fraction']:.1%}, "
                f"clear={corridor['clearance_ok_fraction']:.1%} "
                f"({'PASS' if corridor['ok'] else 'FAIL'})"
            )
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        print("PASS" if report["ok"] else "REJECTED")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
