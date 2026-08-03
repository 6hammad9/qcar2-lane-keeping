#!/usr/bin/env python3

import json
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_JSON = '/home/nvidia/ros2_ws/recorded_path_clean_latest.json'
OUTPUT_JSON = '/home/nvidia/ros2_ws/recorded_path_mpc_smooth.json'

RESAMPLE_SPACING = 0.03      # meters between points
SMOOTH_SIGMA = 2.0           # larger = smoother

# --------------------------------------------------
# LOAD
# --------------------------------------------------

with open(INPUT_JSON, 'r') as f:
    poses = json.load(f)

x = np.array([p['x'] for p in poses], dtype=float)
y = np.array([p['y'] for p in poses], dtype=float)

print(f'Loaded {len(x)} points')

# --------------------------------------------------
# INITIAL SMOOTHING
# --------------------------------------------------

x_smooth = gaussian_filter1d(x, sigma=SMOOTH_SIGMA)
y_smooth = gaussian_filter1d(y, sigma=SMOOTH_SIGMA)

# --------------------------------------------------
# ARC LENGTH
# --------------------------------------------------

arc = [0.0]

for i in range(1, len(x_smooth)):
    ds = math.hypot(
        x_smooth[i] - x_smooth[i-1],
        y_smooth[i] - y_smooth[i-1]
    )
    arc.append(arc[-1] + ds)

arc = np.array(arc)

total_length = arc[-1]

print(f'Total path length: {total_length:.2f} m')

# --------------------------------------------------
# SPLINE FIT
# --------------------------------------------------

# parameterized spline

tck, u = splprep(
    [x_smooth, y_smooth],
    s=0.001
)

# --------------------------------------------------
# UNIFORM RESAMPLING
# --------------------------------------------------

num_points = int(total_length / RESAMPLE_SPACING)

u_new = np.linspace(0.0, 1.0, num_points)

x_new, y_new = splev(u_new, tck)

x_new = np.array(x_new)
y_new = np.array(y_new)

# --------------------------------------------------
# COMPUTE GEOMETRIC YAW
# --------------------------------------------------

yaw_new = []

for i in range(len(x_new) - 1):
    dx = x_new[i+1] - x_new[i]
    dy = y_new[i+1] - y_new[i]

    yaw = math.atan2(dy, dx)
    yaw_new.append(yaw)

# duplicate final yaw

yaw_new.append(yaw_new[-1])

yaw_new = np.unwrap(np.array(yaw_new))

# --------------------------------------------------
# CONVERT YAW TO QUATERNION
# --------------------------------------------------

output = []

for i in range(len(x_new)):
    yaw = yaw_new[i]

    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    output.append({
        'x': float(x_new[i]),
        'y': float(y_new[i]),
        'qz': float(qz),
        'qw': float(qw)
    })

# --------------------------------------------------
# SAVE
# --------------------------------------------------

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nSaved smoothed trajectory:')
print(OUTPUT_JSON)
print(f'Generated {len(output)} MPC-ready points')

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(10, 8))

plt.plot(x, y, 'r.', alpha=0.4, label='Raw Recorded Path')
plt.plot(x_new, y_new, 'b-', linewidth=2, label='Smoothed MPC Path')

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Trajectory Smoothing + Resampling')

plt.show()
