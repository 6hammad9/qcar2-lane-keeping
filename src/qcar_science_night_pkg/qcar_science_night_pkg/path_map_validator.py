#!/usr/bin/env python3

"""Validate that an MPC trajectory is geometrically compatible with a map."""

import argparse
import json
import math
import os

import numpy as np
import yaml


def _next_pgm_token(stream):
    token = bytearray()

    while True:
        char = stream.read(1)
        if not char:
            raise RuntimeError("Unexpected end of PGM header")
        if char == b"#":
            stream.readline()
            continue
        if not char.isspace():
            token.extend(char)
            break

    while True:
        char = stream.read(1)
        if not char or char.isspace():
            break
        token.extend(char)

    return token.decode("ascii")


def read_pgm(path):
    """Read an 8-bit P2/P5 PGM without requiring OpenCV or Pillow."""

    with open(path, "rb") as stream:
        magic = _next_pgm_token(stream)
        width = int(_next_pgm_token(stream))
        height = int(_next_pgm_token(stream))
        max_value = int(_next_pgm_token(stream))

        if max_value <= 0 or max_value > 255:
            raise RuntimeError(
                f"Only 8-bit PGM maps are supported; max value is {max_value}"
            )

        if magic == "P5":
            pixels = np.frombuffer(stream.read(width * height), dtype=np.uint8)
        elif magic == "P2":
            remaining = stream.read().decode("ascii")
            values = []
            for line in remaining.splitlines():
                values.extend(line.split("#", 1)[0].split())
            pixels = np.asarray(values, dtype=np.uint8)
        else:
            raise RuntimeError(f"Unsupported PGM format {magic!r}")

    if pixels.size != width * height:
        raise RuntimeError(
            f"PGM pixel count mismatch: expected {width * height}, got {pixels.size}"
        )

    return pixels.reshape(height, width), max_value


def load_map(map_yaml_file):
    with open(map_yaml_file, "r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    required = ["image", "resolution", "origin"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(f"Map YAML is missing: {', '.join(missing)}")

    image_file = metadata["image"]
    if not os.path.isabs(image_file):
        image_file = os.path.join(os.path.dirname(map_yaml_file), image_file)
    image_file = os.path.abspath(image_file)

    pixels, max_value = read_pgm(image_file)
    mode = str(metadata.get("mode", "trinary")).lower()
    if mode != "trinary":
        raise RuntimeError(
            f"Map mode {mode!r} is unsupported; validator requires 'trinary'"
        )
    resolution = float(metadata["resolution"])
    origin = metadata["origin"]

    if resolution <= 0.0 or len(origin) < 2:
        raise RuntimeError("Map resolution/origin is invalid")

    origin_yaw = float(origin[2]) if len(origin) >= 3 else 0.0
    if not np.all(np.isfinite([
        resolution, float(origin[0]), float(origin[1]), origin_yaw
    ])):
        raise RuntimeError("Map resolution/origin contains non-finite values")

    return {
        "pixels": pixels,
        "max_value": max_value,
        "resolution": resolution,
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "origin_yaw": origin_yaw,
        "negate": int(metadata.get("negate", 0)),
        "occupied_thresh": float(metadata.get("occupied_thresh", 0.65)),
        "free_thresh": float(metadata.get("free_thresh", 0.196)),
        "image_file": image_file,
    }


def _wrap_array(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


def _world_to_grid(x, y, map_data):
    # A map origin is a full SE(2) pose, not just a translation.  Transform
    # world points into the map image frame before converting to cells.
    dx = np.asarray(x) - map_data["origin_x"]
    dy = np.asarray(y) - map_data["origin_y"]
    cosine = math.cos(map_data["origin_yaw"])
    sine = math.sin(map_data["origin_yaw"])
    map_x = cosine * dx + sine * dy
    map_y = -sine * dx + cosine * dy
    columns = np.floor(
        map_x / map_data["resolution"]
    ).astype(int)
    rows_from_bottom = np.floor(
        map_y / map_data["resolution"]
    ).astype(int)
    rows = map_data["pixels"].shape[0] - 1 - rows_from_bottom
    return rows, columns


def _classify_pixels(values, map_data):
    normalized = values.astype(float) / float(map_data["max_value"])
    occupancy = normalized if map_data["negate"] else 1.0 - normalized

    result = np.full(values.shape, 0, dtype=np.int8)  # 0 unknown
    result[occupancy < map_data["free_thresh"]] = 1  # 1 free
    result[occupancy > map_data["occupied_thresh"]] = 2  # 2 occupied
    return result


def validate_trajectory_against_map(
    trajectory_file,
    map_yaml_file,
    *,
    clearance_m=0.10,
    max_outside_fraction=0.0,
    max_occupied_fraction=0.0,
    max_unknown_fraction=0.05,
    max_clearance_violation_fraction=0.01,
    wheelbase_m=0.256,
    max_steer_rad=0.58,
    max_steering_violation_fraction=0.0,
    max_waypoint_spacing_m=0.10,
    loop_path=False,
    max_loop_gap_m=0.15,
):
    """Return a serializable report; ``report['ok']`` gates physical motion."""

    trajectory_file = os.path.abspath(os.path.expanduser(trajectory_file))
    map_yaml_file = os.path.abspath(os.path.expanduser(map_yaml_file))
    fraction_limits = {
        "max_outside_fraction": max_outside_fraction,
        "max_occupied_fraction": max_occupied_fraction,
        "max_unknown_fraction": max_unknown_fraction,
        "max_clearance_violation_fraction": max_clearance_violation_fraction,
        "max_steering_violation_fraction": max_steering_violation_fraction,
    }
    invalid_fractions = [
        name for name, value in fraction_limits.items()
        if not 0.0 <= value <= 1.0
    ]
    if invalid_fractions:
        raise ValueError(
            "Fraction limits must be in [0, 1]: " + ", ".join(invalid_fractions)
        )
    if clearance_m < 0.0 or max_waypoint_spacing_m <= 0.0 or max_loop_gap_m <= 0.0:
        raise ValueError(
            "clearance must be nonnegative; spacing and loop-gap limits must be positive"
        )
    trajectory = np.load(trajectory_file)

    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise RuntimeError("Trajectory must be Nx2, Nx3, or Nx4")
    if len(trajectory) < 3 or not np.all(np.isfinite(trajectory)):
        raise RuntimeError("Trajectory is too short or contains non-finite values")

    xy = trajectory[:, :2].astype(float)
    segments = np.diff(xy, axis=0)
    spacing = np.linalg.norm(segments, axis=1)
    tangent_yaw = np.arctan2(segments[:, 1], segments[:, 0])

    report = {
        "trajectory_file": trajectory_file,
        "map_yaml_file": map_yaml_file,
        "point_count": int(len(trajectory)),
        "path_length_m": float(np.sum(spacing)),
        "spacing_median_m": float(np.median(spacing)),
        "spacing_min_m": float(np.min(spacing)),
        "spacing_max_m": float(np.max(spacing)),
        "spacing_cv": float(np.std(spacing) / max(np.mean(spacing), 1e-12)),
        "start": [float(value) for value in trajectory[0, : min(4, trajectory.shape[1])]],
        "end": [float(value) for value in trajectory[-1, : min(4, trajectory.shape[1])]],
        "end_start_gap_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "warnings": [],
        "failures": [],
    }

    if np.any(spacing < 1e-5):
        report["failures"].append("trajectory contains duplicate consecutive points")

    median_spacing = report["spacing_median_m"]
    if median_spacing > 0.0 and (
        report["spacing_min_m"] < 0.5 * median_spacing
        or report["spacing_max_m"] > 1.75 * median_spacing
    ):
        report["failures"].append(
            "trajectory spacing is not uniform enough for index-based MPC references"
        )
    if report["spacing_max_m"] > max_waypoint_spacing_m:
        report["failures"].append(
            f"maximum waypoint spacing is {report['spacing_max_m']:.3f} m; "
            f"limit is {max_waypoint_spacing_m:.3f} m"
        )

    if trajectory.shape[1] >= 3:
        yaw_error = np.abs(_wrap_array(trajectory[:-1, 2] - tangent_yaw))
        report["yaw_tangent_error_median_deg"] = float(
            np.degrees(np.median(yaw_error))
        )
        report["yaw_tangent_error_max_deg"] = float(np.degrees(np.max(yaw_error)))
        report["yaw_tangent_bad_fraction"] = float(
            np.mean(yaw_error > math.radians(30.0))
        )
        if report["yaw_tangent_bad_fraction"] > 0.05:
            report["failures"].append(
                "more than 5% of waypoint headings disagree with path tangents by >30 deg"
            )

    if wheelbase_m <= 0.0 or max_steer_rad <= 0.0:
        raise RuntimeError("wheelbase_m and max_steer_rad must be positive")

    heading_change = np.abs(_wrap_array(np.diff(tangent_yaw)))
    local_arc = 0.5 * (spacing[:-1] + spacing[1:])
    local_curvature = heading_change / np.maximum(local_arc, 1e-6)
    required_steering = np.arctan(wheelbase_m * local_curvature)
    steering_violation_fraction = float(
        np.mean(required_steering > max_steer_rad)
    ) if len(required_steering) else 0.0
    report["required_steering_max_rad"] = float(
        np.max(required_steering) if len(required_steering) else 0.0
    )
    report["required_steering_p99_rad"] = float(
        np.percentile(required_steering, 99.0) if len(required_steering) else 0.0
    )
    report["steering_violation_fraction"] = steering_violation_fraction
    if steering_violation_fraction > max_steering_violation_fraction:
        report["failures"].append(
            f"{steering_violation_fraction:.1%} of path turns require steering "
            f"above {max_steer_rad:.2f} rad; allowed "
            f"{max_steering_violation_fraction:.1%}"
        )

    report["loop_path"] = bool(loop_path)
    if loop_path:
        closure = xy[0] - xy[-1]
        closure_length = float(np.linalg.norm(closure))
        if closure_length > max_loop_gap_m:
            report["failures"].append(
                f"loop seam gap is {closure_length:.3f} m; limit is "
                f"{max_loop_gap_m:.3f} m"
            )
        elif closure_length < 1e-5:
            report["failures"].append(
                "loop contains a duplicate end/start point; remove one endpoint"
            )
        else:
            closure_heading = math.atan2(closure[1], closure[0])
            seam_turns = np.abs(_wrap_array(np.asarray([
                closure_heading - tangent_yaw[-1],
                tangent_yaw[0] - closure_heading,
            ])))
            seam_arcs = np.asarray([
                0.5 * (spacing[-1] + closure_length),
                0.5 * (closure_length + spacing[0]),
            ])
            seam_steering = np.arctan(
                wheelbase_m * seam_turns / np.maximum(seam_arcs, 1e-6)
            )
            report["loop_seam_required_steering_rad"] = float(
                np.max(seam_steering)
            )
            if report["loop_seam_required_steering_rad"] > max_steer_rad:
                report["failures"].append(
                    "loop seam requires "
                    f"{report['loop_seam_required_steering_rad']:.2f} rad steering; "
                    f"limit is {max_steer_rad:.2f} rad"
                )

    map_data = load_map(map_yaml_file)
    height, width = map_data["pixels"].shape
    rows, columns = _world_to_grid(xy[:, 0], xy[:, 1], map_data)
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)

    classes = np.full(len(trajectory), -1, dtype=np.int8)  # -1 outside
    classes[inside] = _classify_pixels(
        map_data["pixels"][rows[inside], columns[inside]], map_data
    )

    outside_fraction = float(np.mean(classes == -1))
    unknown_fraction = float(np.mean(classes == 0))
    free_fraction = float(np.mean(classes == 1))
    occupied_fraction = float(np.mean(classes == 2))

    report.update(
        {
            "map_image_file": map_data["image_file"],
            "map_width": int(width),
            "map_height": int(height),
            "map_resolution_m": float(map_data["resolution"]),
            "outside_fraction": outside_fraction,
            "unknown_fraction": unknown_fraction,
            "free_fraction": free_fraction,
            "occupied_fraction": occupied_fraction,
        }
    )

    radius_pixels = max(0, int(math.ceil(clearance_m / map_data["resolution"])))
    disk_offsets = [
        (dr, dc)
        for dr in range(-radius_pixels, radius_pixels + 1)
        for dc in range(-radius_pixels, radius_pixels + 1)
        if dr * dr + dc * dc <= radius_pixels * radius_pixels
    ]
    clearance_violations = np.zeros(len(trajectory), dtype=bool)

    for index in np.flatnonzero(inside):
        for dr, dc in disk_offsets:
            row = rows[index] + dr
            column = columns[index] + dc
            if row < 0 or row >= height or column < 0 or column >= width:
                clearance_violations[index] = True
                break
            pixel_class = _classify_pixels(
                np.asarray([map_data["pixels"][row, column]]), map_data
            )[0]
            # Unknown space is not proven drivable and therefore cannot
            # satisfy a physical clearance requirement.
            if pixel_class != 1:
                clearance_violations[index] = True
                break

    clearance_violations[~inside] = True
    clearance_fraction = float(np.mean(clearance_violations))
    report["clearance_m"] = float(clearance_m)
    report["clearance_violation_fraction"] = clearance_fraction

    limits = [
        (outside_fraction, max_outside_fraction, "outside map"),
        (occupied_fraction, max_occupied_fraction, "on occupied cells"),
        (unknown_fraction, max_unknown_fraction, "on unknown cells"),
        (
            clearance_fraction,
            max_clearance_violation_fraction,
            f"within {clearance_m:.2f} m of occupied/unknown/outside cells",
        ),
    ]
    for measured, maximum, label in limits:
        if measured > maximum:
            report["failures"].append(
                f"{measured:.1%} of path points are {label}; allowed {maximum:.1%}"
            )

    report["ok"] = not report["failures"]
    return report


def format_report(report):
    lines = [
        f"trajectory: {report['trajectory_file']}",
        f"map: {report['map_yaml_file']}",
        (
            f"points={report['point_count']} length={report['path_length_m']:.2f} m "
            f"spacing={report['spacing_median_m']:.3f} m "
            f"(cv={report['spacing_cv']:.2f})"
        ),
        (
            f"map cells: free={report['free_fraction']:.1%} "
            f"unknown={report['unknown_fraction']:.1%} "
            f"occupied={report['occupied_fraction']:.1%} "
            f"outside={report['outside_fraction']:.1%}"
        ),
        (
            f"clearance violations ({report['clearance_m']:.2f} m): "
            f"{report['clearance_violation_fraction']:.1%}"
        ),
    ]
    if report.get("yaw_tangent_error_median_deg") is not None:
        lines.append(
            "yaw/tangent error: "
            f"median={report['yaw_tangent_error_median_deg']:.1f} deg "
            f"max={report['yaw_tangent_error_max_deg']:.1f} deg"
        )
    lines.append(
        "required steering: "
        f"p99={report['required_steering_p99_rad']:.2f} rad "
        f"max={report['required_steering_max_rad']:.2f} rad"
    )
    if report.get("loop_seam_required_steering_rad") is not None:
        lines.append(
            "loop seam required steering: "
            f"{report['loop_seam_required_steering_rad']:.2f} rad"
        )
    lines.extend(f"FAIL: {message}" for message in report["failures"])
    lines.extend(f"WARN: {message}" for message in report["warnings"])
    lines.append("PASS" if report["ok"] else "REJECTED FOR PHYSICAL MOTION")
    return "\n".join(lines)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--map", required=True, dest="map_yaml")
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--max-outside-fraction", type=float, default=0.0)
    parser.add_argument("--max-occupied-fraction", type=float, default=0.0)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.05)
    parser.add_argument("--max-clearance-violation-fraction", type=float, default=0.01)
    parser.add_argument("--wheelbase", type=float, default=0.256)
    parser.add_argument("--max-steer", type=float, default=0.58)
    parser.add_argument("--max-steering-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-waypoint-spacing", type=float, default=0.10)
    parser.add_argument("--loop-path", action="store_true")
    parser.add_argument("--max-loop-gap", type=float, default=0.15)
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(args=args)

    report = validate_trajectory_against_map(
        parsed.trajectory,
        parsed.map_yaml,
        clearance_m=parsed.clearance,
        max_outside_fraction=parsed.max_outside_fraction,
        max_occupied_fraction=parsed.max_occupied_fraction,
        max_unknown_fraction=parsed.max_unknown_fraction,
        max_clearance_violation_fraction=(
            parsed.max_clearance_violation_fraction
        ),
        wheelbase_m=parsed.wheelbase,
        max_steer_rad=parsed.max_steer,
        max_steering_violation_fraction=(
            parsed.max_steering_violation_fraction
        ),
        max_waypoint_spacing_m=parsed.max_waypoint_spacing,
        loop_path=parsed.loop_path,
        max_loop_gap_m=parsed.max_loop_gap,
    )
    print(json.dumps(report, indent=2) if parsed.json else format_report(report))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
