#!/usr/bin/env python3

import sys
import math
import numpy as np


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


if len(sys.argv) < 2:
    print("Usage:")
    print("  python3 check_npy_gap.py /path/to/path.npy")
    sys.exit(1)

path_file = sys.argv[1]
data = np.load(path_file)

if data.ndim != 2 or data.shape[1] < 2:
    raise RuntimeError("Path must be Nx2, Nx3, or Nx4")

x0, y0 = data[0, 0], data[0, 1]
x1, y1 = data[-1, 0], data[-1, 1]

gap = math.hypot(x1 - x0, y1 - y0)

print(f"File: {path_file}")
print(f"Shape: {data.shape}")
print(f"Start: x={x0:.4f}, y={y0:.4f}")
print(f"End:   x={x1:.4f}, y={y1:.4f}")
print(f"End-start gap: {gap:.4f} m")

if data.shape[1] >= 3:
    yaw0 = data[0, 2]
    yaw1 = data[-1, 2]
    yaw_gap = wrap_angle(yaw1 - yaw0)

    print(f"Start yaw: {math.degrees(yaw0):.2f} deg")
    print(f"End yaw:   {math.degrees(yaw1):.2f} deg")
    print(f"Yaw difference: {math.degrees(yaw_gap):.2f} deg")

if data.shape[1] >= 4:
    curvature = data[:, 3]
    print(f"Curvature min/max: {curvature.min():.3f} / {curvature.max():.3f} 1/m")

if gap < 0.05:
    print("Result: very good loop gap")
elif gap < 0.10:
    print("Result: acceptable loop gap")
else:
    print("Result: large gap")