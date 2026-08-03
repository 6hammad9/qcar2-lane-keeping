import pandas as pd
import numpy as np
import math
from scipy.interpolate import splprep, splev

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# 1. Load the raw trajectory
input_file = 'config/slam_trajectory.csv'
output_file = 'config/smoothed_trajectory.csv'
df = pd.read_csv(input_file)
waypoints = df.to_dict('records')

# 1b. Despike: drop single-point outliers caused by a one-frame SLAM
# scan-matching glitch — the recorder briefly records a bad pose, then
# immediately snaps back onto the real path. Detected as a point whose
# neighbors are much closer to *each other* directly than either is to the
# point itself (i.e. going through this point is a detour, not progress).
def _despike(points, detour_ratio=2.0):
    if len(points) < 3:
        return points
    cleaned = [points[0]]
    i = 1
    while i < len(points) - 1:
        prev, cur, nxt = cleaned[-1], points[i], points[i + 1]
        d_prev_cur = math.hypot(cur['x'] - prev['x'], cur['y'] - prev['y'])
        d_cur_next = math.hypot(nxt['x'] - cur['x'], nxt['y'] - cur['y'])
        d_prev_next = math.hypot(nxt['x'] - prev['x'], nxt['y'] - prev['y'])
        is_spike = (
            d_prev_next > 0
            and (d_prev_cur + d_cur_next) > detour_ratio * d_prev_next
        )
        if is_spike:
            i += 1  # drop this point, don't add it to cleaned
            continue
        cleaned.append(cur)
        i += 1
    cleaned.append(points[-1])
    return cleaned

_before = len(waypoints)
waypoints = _despike(waypoints)
if len(waypoints) < _before:
    print(f"Despiked {_before - len(waypoints)} outlier point(s) from raw recording "
          f"(likely one-frame SLAM pose glitches).")

# 2. Detect flip points (using your exact logic)
directions = []
for i in range(len(waypoints) - 1):
    dx = waypoints[i+1]['x'] - waypoints[i]['x']
    dy = waypoints[i+1]['y'] - waypoints[i]['y']
    dist = math.hypot(dx, dy)
    directions.append((dx/dist, dy/dist) if dist > 0.001 else (directions[-1] if directions else (1.0, 0.0)))

flip_indices = []
for i in range(len(directions) - 1):
    if (directions[i][0]*directions[i+1][0] + directions[i][1]*directions[i+1][1]) < -0.5:
        flip_indices.append(i + 1)

# 3. Create Segments
segments = []
start = 0
for f in flip_indices:
    segments.append((start, f))
    start = f
segments.append((start, len(waypoints) - 1))

# 3b. Merge spuriously short segments into their previous neighbor.
# A single noisy raw point (e.g. a one-frame SLAM scan-matching glitch) can
# register as a direction reversal followed immediately by a reversal back,
# producing a tiny segment that isn't a real intentional direction change in
# the drive. Segments below this length are treated as noise and folded into
# the preceding segment rather than kept as their own short, unstable
# pure-pursuit segment.
MIN_SEGMENT_LENGTH = 0.15  # meters

def _segment_length(start_idx, end_idx):
    length = 0.0
    for i in range(start_idx, end_idx):
        length += math.hypot(
            waypoints[i + 1]['x'] - waypoints[i]['x'],
            waypoints[i + 1]['y'] - waypoints[i]['y'],
        )
    return length

merged_segments = [segments[0]]
dropped = 0
for seg in segments[1:]:
    if _segment_length(*seg) < MIN_SEGMENT_LENGTH:
        prev_start, _ = merged_segments[-1]
        merged_segments[-1] = (prev_start, seg[1])
        dropped += 1
    else:
        merged_segments.append(seg)
segments = merged_segments
if dropped:
    print(f"Merged {dropped} short segment(s) (<{MIN_SEGMENT_LENGTH}m) into "
          f"their previous neighbor — likely recording/localization noise, "
          f"not real direction changes.")

# 4. Smooth and Resample each segment
smoothed_waypoints = []
resample_distance = 0.05  # 5 cm uniform spacing

for start_idx, end_idx in segments:
    # Extract segment points
    seg_x = [waypoints[i]['x'] for i in range(start_idx, end_idx + 1)]
    seg_y = [waypoints[i]['y'] for i in range(start_idx, end_idx + 1)]
    
    # Skip smoothing if the segment is extremely short (less than 4 points)
    if len(seg_x) < 4:
        for i in range(start_idx, end_idx + 1):
            smoothed_waypoints.append({'x': waypoints[i]['x'], 'y': waypoints[i]['y'], 'theta': waypoints[i]['theta']})
        continue

    # Fit B-spline (s parameter controls the amount of smoothing; increase if still jittery)
    tck, u = splprep([seg_x, seg_y], s=0.02, k=3)
    
    # Calculate total curve length to determine how many points we need for 5cm spacing
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)
    total_length = np.sum(np.hypot(np.diff(x_fine), np.diff(y_fine)))
    num_points = max(2, int(total_length / resample_distance))
    
    # Generate evenly spaced u values
    u_even = np.linspace(0, 1, num_points)
    smooth_x, smooth_y = splev(u_even, tck)
    
    # Calculate mathematically perfect headings (theta)
    for i in range(len(smooth_x)):
        if i < len(smooth_x) - 1:
            dx = smooth_x[i+1] - smooth_x[i]
            dy = smooth_y[i+1] - smooth_y[i]
        else:
            dx = smooth_x[i] - smooth_x[i-1]
            dy = smooth_y[i] - smooth_y[i-1]
        
        theta = math.atan2(dy, dx)
        
        # Avoid duplicating the exact flip point coordinates between segments
        if i == 0 and len(smoothed_waypoints) > 0:
            continue
            
        smoothed_waypoints.append({'x': smooth_x[i], 'y': smooth_y[i], 'theta': theta})

# 5. Save the clean trajectory
smooth_df = pd.DataFrame(smoothed_waypoints)
smooth_df.to_csv(output_file, index=False)
print(f"Smoothed trajectory saved to {output_file} with {len(smooth_df)} uniform waypoints.")