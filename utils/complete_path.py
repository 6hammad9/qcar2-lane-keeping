#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt


input_path = "/home/nvidia/ros2_ws/recorded_path_mpc_long_parking_curvature_smooth_1_5_10_06_2026.npy"
output_path = "/home/nvidia/ros2_ws/recorded_path_mpc_long_parking_curvature_smooth_1_5_10_06_2026_closed.npy"

close_threshold = 0.30
min_loop_index = 100
num_yaw_align = 30


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_yaw(xy):
    yaw = []

    for i in range(len(xy)):
        if i < len(xy) - 1:
            dx = xy[i + 1, 0] - xy[i, 0]
            dy = xy[i + 1, 1] - xy[i, 1]
        else:
            dx = xy[0, 0] - xy[i, 0]
            dy = xy[0, 1] - xy[i, 1]

        yaw.append(math.atan2(dy, dx))

    return np.unwrap(np.array(yaw))


path = np.load(input_path)

if path.ndim != 2 or path.shape[1] < 2:
    raise RuntimeError("Path must have at least x and y columns")

xy = path[:, :2]
start_xy = xy[0]

distances = np.linalg.norm(xy - start_xy, axis=1)

candidate_indices = np.where(distances < close_threshold)[0]
candidate_indices = candidate_indices[candidate_indices > min_loop_index]

if len(candidate_indices) == 0:
    search_distances = distances[min_loop_index:]
    nearest_relative = np.argmin(search_distances)
    cut_idx = nearest_relative + min_loop_index

    print(f"No loop closure found within {close_threshold:.2f} m")
    print(f"Using closest return point instead.")
else:
    cut_idx = candidate_indices[0]

closed_xy = xy[:cut_idx + 1]

closed_yaw = compute_yaw(closed_xy)

target_yaw = closed_yaw[0]
num_align = min(num_yaw_align, len(closed_yaw))

for i in range(num_align):
    idx = len(closed_yaw) - num_align + i
    alpha = (i + 1) / num_align

    yaw_error = wrap_angle(target_yaw - closed_yaw[idx])
    closed_yaw[idx] += alpha * yaw_error

closed_yaw[-1] = target_yaw

closed_path = np.column_stack([
    closed_xy[:, 0],
    closed_xy[:, 1],
    closed_yaw,
])

np.save(output_path, closed_path)

gap = np.linalg.norm(closed_xy[-1] - closed_xy[0])
yaw_diff = wrap_angle(closed_yaw[-1] - closed_yaw[0])

print(f"Original path: {len(path)} points")
print(f"Loop closure index: {cut_idx}")
print(f"Final path: {len(closed_path)} points")
print(f"End-start gap: {gap:.3f} m")
print(f"Yaw difference: {math.degrees(yaw_diff):.2f} deg")
print(f"Saved to: {output_path}")

plt.figure(figsize=(8, 8))
plt.plot(xy[:, 0], xy[:, 1], "b-", linewidth=1, label="Original path")
plt.plot(closed_xy[:, 0], closed_xy[:, 1], "r-", linewidth=2, label="Closed path")
plt.scatter(closed_xy[0, 0], closed_xy[0, 1], c="green", s=80, label="Start")
plt.scatter(closed_xy[-1, 0], closed_xy[-1, 1], c="red", s=80, label="New end")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Loop Closed Without Straight Connector")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()