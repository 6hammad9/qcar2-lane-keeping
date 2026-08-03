#!/usr/bin/env python3
"""Run the production lane geometry detector on a saved camera frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "qcar_science_night_pkg"
sys.path.insert(0, str(PACKAGE_ROOT))

from qcar_science_night_pkg.lane_centering_node import (  # noqa: E402
    LaneCenteringNode,
)


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{
            key: namespace(item) for key, item in value.items()
        })
    return value


def validate_frame(image_path, config_path, top=None, bottom=None):
    with open(config_path, "r", encoding="utf-8") as stream:
        config = namespace(yaml.safe_load(stream))
    if top is not None:
        config.image.strip_top_fraction = float(top)
    if bottom is not None:
        config.image.strip_bottom_fraction = float(bottom)

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"could not read image: {image_path}")
    frame = cv2.resize(
        frame,
        (
            int(config.performance.resize_width),
            int(config.performance.resize_height),
        ),
        interpolation=cv2.INTER_AREA,
    )
    height, width = frame.shape[:2]
    y_top = int(height * config.image.strip_top_fraction)
    y_bottom = int(height * config.image.strip_bottom_fraction)
    strip = frame[y_top:y_bottom, :]

    # Construct without Node.__init__: the geometry methods are deliberately
    # side-effect free and only need the parsed detector configuration.
    detector = LaneCenteringNode.__new__(LaneCenteringNode)
    detector.cfg = config
    binary = LaneCenteringNode.segment_white(detector, strip)
    segments = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=int(config.hough.threshold),
        minLineLength=int(config.hough.min_line_length),
        maxLineGap=int(config.hough.max_line_gap),
    )
    center_line = LaneCenteringNode.detect_cl(
        detector, binary, segments, width
    )
    road_line = LaneCenteringNode.detect_rl(
        detector, binary, segments, width
    )
    sample_y = max(0, len(strip) - 1)
    center_x = (
        LaneCenteringNode.eval_line_at_y(center_line, sample_y)
        if center_line is not None else None
    )
    road_x = (
        LaneCenteringNode.eval_poly_at_y(road_line, sample_y)
        if road_line is not None else None
    )
    valid = (
        center_x is not None
        and road_x is not None
        and 0.0 <= center_x < width
        and 0.0 <= road_x < width
        and config.detection.min_lane_width_px
        <= road_x - center_x
        <= config.detection.max_lane_width_px
    )
    return {
        "valid": bool(valid),
        "crop": [y_top, y_bottom],
        "white_fraction": float(np.mean(binary > 0)),
        "hough_segments": 0 if segments is None else int(len(segments)),
        "divider_x": None if center_x is None else float(center_x),
        "right_edge_x": None if road_x is None else float(road_x),
        "detected_width_px": (
            None if center_x is None or road_x is None
            else float(road_x - center_x)
        ),
        "image_center_x": 0.5 * width,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "lane_params.yaml"),
    )
    parser.add_argument("--top", type=float)
    parser.add_argument("--bottom", type=float)
    args = parser.parse_args()
    report = validate_frame(
        args.image, args.config, top=args.top, bottom=args.bottom
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
