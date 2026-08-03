#!/usr/bin/env python3
"""Generate a Gazebo (gz-sim) world from a real ROS occupancy map.

Occupied pixels of the PGM are merged into rectangles and extruded into
static walls at their EXACT map coordinates (the map yaml's origin and
resolution are honored), so:

    Gazebo world frame == ROS `map` frame

Every trajectory, V2V gap, and obstacle coordinate used in simulation is
therefore valid on the physical robots unchanged.

Also places:
  - obstacles from an optional YAML file (edit it to mirror the real props),
  - a self-contained, ROSbot-like four-wheel proxy at the ROSbot trajectory
    start (driven later by sim_rosbot.py),
  - a flat green marker at the QCar trajectory start (spawn aid).

Usage:
    python3 map_to_sdf.py \
        --map ../../opta2-sami_ahmed/config/track_map.yaml \
        --obstacles obstacles.yaml \
        --rosbot-csv ../../opta2-sami_ahmed/config/smoothed_trajectory.csv \
        --out worlds/sami_track.sdf

Pure Python (only PyYAML optional -- falls back to a tiny parser), runs
anywhere.
"""

import argparse
import csv
import math
import os
import re
import struct
import sys
import zlib


# ----------------------------------------------------------------------
def load_map_yaml(path):
    """Minimal parser for map_server YAML (avoids a PyYAML dependency)."""
    data = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()
    origin = [float(v) for v in re.findall(r"-?\d+\.?\d*", data["origin"])]
    return {
        "image": data["image"],
        "resolution": float(data["resolution"]),
        "origin": origin,
        "occupied_thresh": float(data.get("occupied_thresh", 0.65)),
        "negate": int(data.get("negate", 0)),
    }


def load_pgm(path):
    """Read binary (P5) or ASCII (P2) PGM. Returns (width, height, pixels)."""
    with open(path, "rb") as fh:
        content = fh.read()

    # Tokenize header, skipping comments.
    tokens = []
    idx = 0
    while len(tokens) < 4:
        # Skip whitespace
        while idx < len(content) and content[idx : idx + 1].isspace():
            idx += 1
        if content[idx : idx + 1] == b"#":
            while idx < len(content) and content[idx : idx + 1] != b"\n":
                idx += 1
            continue
        start = idx
        while idx < len(content) and not content[idx : idx + 1].isspace():
            idx += 1
        tokens.append(content[start:idx])
    magic = tokens[0].decode()
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    idx += 1  # single whitespace after maxval

    if magic == "P5":
        pixels = list(content[idx : idx + width * height])
    elif magic == "P2":
        pixels = [int(t) for t in content[idx:].split()][: width * height]
    else:
        raise ValueError(f"Unsupported PGM magic {magic!r}")
    if len(pixels) != width * height:
        raise ValueError("PGM pixel count mismatch")
    if maxval != 255:
        pixels = [int(p * 255 / maxval) for p in pixels]
    return width, height, pixels


def occupied_grid(width, height, pixels, occupied_thresh, negate):
    """Boolean grid[row][col]: True where the map says 'wall'."""
    cut = 255.0 * (1.0 - occupied_thresh)  # v <= cut -> occupied (negate=0)
    grid = []
    for r in range(height):
        row = []
        for c in range(width):
            v = pixels[r * width + c]
            occ = (v >= 255 - cut) if negate else (v <= cut)
            row.append(occ)
        grid.append(row)
    return grid


def remove_small_blobs(grid, min_cells):
    """Drop occupied components smaller than min_cells (SLAM debris:
    furniture legs, people, scan noise). Keeps real walls, which are
    hundreds of cells."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if not grid[r][c] or seen[r][c]:
                continue
            stack = [(r, c)]
            comp = []
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and grid[nr][nc] and not seen[nr][nc]):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(comp) < min_cells:
                for cr, cc in comp:
                    grid[cr][cc] = False
    return grid


def merge_rectangles(grid):
    """Greedy merge of occupied cells into rectangles (row runs, then stack
    vertically-identical runs). Returns [(r0, c0, r1, c1)] inclusive."""
    height = len(grid)
    width = len(grid[0]) if height else 0

    # Horizontal runs per row.
    runs = []  # (row, c_start, c_end)
    for r in range(height):
        c = 0
        while c < width:
            if grid[r][c]:
                c0 = c
                while c < width and grid[r][c]:
                    c += 1
                runs.append((r, c0, c - 1))
            else:
                c += 1

    # Stack runs with identical extents across consecutive rows.
    open_rects = {}   # (c0, c1) -> [r_start, r_last]
    rects = []
    runs.sort()
    by_row = {}
    for r, c0, c1 in runs:
        by_row.setdefault(r, []).append((c0, c1))

    for r in range(height + 1):
        todays = set(by_row.get(r, []))
        for key in list(open_rects):
            if key not in todays:
                r0, r1 = open_rects.pop(key)
                rects.append((r0, key[0], r1, key[1]))
        for key in todays:
            if key in open_rects:
                open_rects[key][1] = r
            else:
                open_rects[key] = [r, r]
    return rects


# ----------------------------------------------------------------------
BOX_TMPL = """      <collision name="{name}_col">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
      </collision>
      <visual name="{name}_vis">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        <!-- Occupancy-map boxes remain physical LiDAR/collision geometry,
             but are hidden so they do not cover the painted track in the
             operator view or camera image. -->
        <transparency>1.0</transparency>
        <cast_shadows>false</cast_shadows>
        <material><ambient>0.25 0.25 0.28 1</ambient>
          <diffuse>0.35 0.35 0.40 1</diffuse></material>
      </visual>
"""

OBSTACLE_TMPL = """    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material><ambient>0.8 0.4 0.05 1</ambient>
            <diffuse>0.9 0.5 0.1 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

ROSBOT_TMPL = """    <model name="sim_rosbot">
      <static>false</static>
      <pose>{x:.4f} {y:.4f} 0.11 0 0 {yaw:.4f}</pose>
      <link name="body">
        <inertial><mass>2.0</mass>
          <inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="col">
          <pose>0 0 -0.005 0 0 0</pose>
          <geometry><box><size>0.36 0.28 0.17</size></box></geometry>
        </collision>
        <visual name="chassis_lower">
          <pose>0 0 -0.025 0 0 0</pose>
          <geometry><box><size>0.35 0.255 0.085</size></box></geometry>
          <material><ambient>0.035 0.045 0.055 1</ambient>
            <diffuse>0.055 0.070 0.085 1</diffuse></material>
        </visual>
        <visual name="top_plate">
          <pose>0 0 0.030 0 0 0</pose>
          <geometry><box><size>0.31 0.245 0.025</size></box></geometry>
          <material><ambient>0.035 0.20 0.34 1</ambient>
            <diffuse>0.045 0.31 0.54 1</diffuse></material>
        </visual>
        <visual name="electronics_enclosure">
          <pose>-0.025 0 0.079 0 0 0</pose>
          <geometry><box><size>0.205 0.18 0.075</size></box></geometry>
          <material><ambient>0.055 0.075 0.095 1</ambient>
            <diffuse>0.075 0.105 0.135 1</diffuse></material>
        </visual>
        <visual name="side_rail_left">
          <pose>0 0.139 0.005 0 0 0</pose>
          <geometry><box><size>0.345 0.024 0.065</size></box></geometry>
          <material><ambient>0.43 0.46 0.49 1</ambient>
            <diffuse>0.62 0.66 0.70 1</diffuse></material>
        </visual>
        <visual name="side_rail_right">
          <pose>0 -0.139 0.005 0 0 0</pose>
          <geometry><box><size>0.345 0.024 0.065</size></box></geometry>
          <material><ambient>0.43 0.46 0.49 1</ambient>
            <diffuse>0.62 0.66 0.70 1</diffuse></material>
        </visual>
        <visual name="front_bumper">
          <pose>0.187 0 -0.035 0 0 0</pose>
          <geometry><box><size>0.025 0.295 0.035</size></box></geometry>
          <material><ambient>0.34 0.37 0.40 1</ambient>
            <diffuse>0.54 0.58 0.62 1</diffuse></material>
        </visual>
        <visual name="rear_bumper">
          <pose>-0.187 0 -0.035 0 0 0</pose>
          <geometry><box><size>0.025 0.295 0.035</size></box></geometry>
          <material><ambient>0.34 0.37 0.40 1</ambient>
            <diffuse>0.54 0.58 0.62 1</diffuse></material>
        </visual>
        <visual name="sensor_mast">
          <pose>0.015 0 0.157 0 0 0</pose>
          <geometry><cylinder><radius>0.016</radius>
            <length>0.12</length></cylinder></geometry>
          <material><ambient>0.32 0.34 0.36 1</ambient>
            <diffuse>0.55 0.58 0.60 1</diffuse></material>
        </visual>
        <visual name="lidar_base">
          <pose>0.015 0 0.218 0 0 0</pose>
          <geometry><cylinder><radius>0.050</radius>
            <length>0.020</length></cylinder></geometry>
          <material><ambient>0.03 0.035 0.04 1</ambient>
            <diffuse>0.055 0.06 0.07 1</diffuse></material>
        </visual>
        <visual name="lidar_puck">
          <pose>0.015 0 0.247 0 0 0</pose>
          <geometry><cylinder><radius>0.044</radius>
            <length>0.040</length></cylinder></geometry>
          <material><ambient>0.04 0.045 0.05 1</ambient>
            <diffuse>0.075 0.085 0.095 1</diffuse></material>
        </visual>
        <visual name="front_camera">
          <pose>0.163 0 0.085 0 0 0</pose>
          <geometry><box><size>0.040 0.090 0.050</size></box></geometry>
          <material><ambient>0.36 0.38 0.40 1</ambient>
            <diffuse>0.62 0.65 0.68 1</diffuse></material>
        </visual>
        <visual name="front_camera_lens">
          <pose>0.190 0 0.085 0 1.5708 0</pose>
          <geometry><cylinder><radius>0.016</radius>
            <length>0.014</length></cylinder></geometry>
          <material><ambient>0.015 0.025 0.035 1</ambient>
            <diffuse>0.025 0.07 0.11 1</diffuse></material>
        </visual>
        <visual name="front_direction_marker">
          <pose>0.158 0 0.122 0 0 0</pose>
          <geometry><box><size>0.050 0.13 0.008</size></box></geometry>
          <material><ambient>0.85 0.34 0.02 1</ambient>
            <diffuse>1.0 0.48 0.03 1</diffuse></material>
        </visual>
        <visual name="wheel_fl">
          <pose>0.115 0.165 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.055</radius>
            <length>0.040</length></cylinder></geometry>
          <material><ambient>0.03 0.03 0.03 1</ambient>
            <diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
        <visual name="wheel_fr">
          <pose>0.115 -0.165 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.055</radius>
            <length>0.040</length></cylinder></geometry>
          <material><ambient>0.03 0.03 0.03 1</ambient>
            <diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
        <visual name="wheel_rl">
          <pose>-0.115 0.165 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.055</radius>
            <length>0.040</length></cylinder></geometry>
          <material><ambient>0.03 0.03 0.03 1</ambient>
            <diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
        <visual name="wheel_rr">
          <pose>-0.115 -0.165 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.055</radius>
            <length>0.040</length></cylinder></geometry>
          <material><ambient>0.03 0.03 0.03 1</ambient>
            <diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
        <visual name="hub_fl">
          <pose>0.115 0.186 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.023</radius>
            <length>0.006</length></cylinder></geometry>
          <material><ambient>0.26 0.29 0.31 1</ambient>
            <diffuse>0.48 0.52 0.56 1</diffuse></material>
        </visual>
        <visual name="hub_fr">
          <pose>0.115 -0.186 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.023</radius>
            <length>0.006</length></cylinder></geometry>
          <material><ambient>0.26 0.29 0.31 1</ambient>
            <diffuse>0.48 0.52 0.56 1</diffuse></material>
        </visual>
        <visual name="hub_rl">
          <pose>-0.115 0.186 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.023</radius>
            <length>0.006</length></cylinder></geometry>
          <material><ambient>0.26 0.29 0.31 1</ambient>
            <diffuse>0.48 0.52 0.56 1</diffuse></material>
        </visual>
        <visual name="hub_rr">
          <pose>-0.115 -0.186 -0.055 1.5708 0 0</pose>
          <geometry><cylinder><radius>0.023</radius>
            <length>0.006</length></cylinder></geometry>
          <material><ambient>0.26 0.29 0.31 1</ambient>
            <diffuse>0.48 0.52 0.56 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

ROAD_MESH_TMPL = """      <visual name="road_surface">
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
        <material>
          <ambient>0.20 0.21 0.23 1</ambient>
          <diffuse>0.64 0.66 0.70 1</diffuse>
          <specular>0 0 0 1</specular>
          <pbr><metal>
            <albedo_map>{texture_uri}</albedo_map>
            <metalness>0.0</metalness>
            <roughness>1.0</roughness>
          </metal></pbr>
        </material>
      </visual>
"""

DASH_TMPL = """      <visual name="divider_dash_{i}">
        <pose>{x:.4f} {y:.4f} 0.016 0 0 {yaw:.4f}</pose>
        <geometry><box><size>0.18 0.018 0.002</size></box></geometry>
        <material><ambient>0.9 0.9 0.85 1</ambient>
          <diffuse>1 1 0.95 1</diffuse></material>
      </visual>
"""


EDGE_TMPL = """      <visual name="{name}_{i}">
        <pose>{x:.4f} {y:.4f} 0.016 0 0 {yaw:.4f}</pose>
        <geometry><box><size>{seg_len:.3f} 0.022 0.002</size></box></geometry>
        <material><ambient>0.92 0.92 0.88 1</ambient>
          <diffuse>1 1 0.96 1</diffuse></material>
      </visual>
"""


def load_path_csv(csv_path):
    points = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = "theta" if "theta" in row else "yaw"
            points.append((float(row["x"]), float(row["y"]), float(row[key])))
    if len(points) < 3:
        raise RuntimeError("road path CSV must contain at least three points")
    return points


def offset_path(points, offset_left):
    """Offset a closed path along its local left normal."""
    result = []
    count = len(points)
    for i, (x, y, _yaw) in enumerate(points):
        x_prev, y_prev, _ = points[(i - 1) % count]
        x_next, y_next, _ = points[(i + 1) % count]
        yaw = math.atan2(y_next - y_prev, x_next - x_prev)
        result.append((
            x - math.sin(yaw) * offset_left,
            y + math.cos(yaw) * offset_left,
            yaw,
        ))
    return result


def closed_segment_geometry(points, start, step):
    end = (start + step) % len(points)
    x0, y0, _ = points[start]
    x1, y1, _ = points[end]
    yaw = math.atan2(y1 - y0, x1 - x0)
    return (
        0.5 * (x0 + x1),
        0.5 * (y0 + y1),
        yaw,
        math.hypot(x1 - x0, y1 - y0),
    )


def _png_chunk(chunk_type, data):
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def write_asphalt_texture(path, size=8):
    """Write a tiny medium-grey rough texture without dependencies."""
    # Keep every value far below the camera detector's V>=170 white cutoff.
    palette = (
        (104, 108, 114),
        (110, 114, 120),
        (99, 103, 109),
        (115, 119, 125),
    )
    rows = []
    for y in range(size):
        row = bytearray((0,))  # PNG filter: none
        for x in range(size):
            row.extend(palette[(x * 3 + y * 5) % len(palette)])
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as stream:
        stream.write(png)


def write_road_mesh(
    csv_path,
    width,
    overtake_offset_left,
    output_path,
    texture_path,
):
    """Write one closed OBJ strip, avoiding coplanar overlapping road boxes."""
    points = load_path_csv(csv_path)
    road_center_left = 0.5 * overtake_offset_left
    right = offset_path(points, road_center_left - 0.5 * width)
    left = offset_path(points, road_center_left + 0.5 * width)
    count = len(points)

    material_path = os.path.splitext(output_path)[0] + ".mtl"
    material_name = os.path.basename(material_path)
    texture_name = os.path.basename(texture_path)
    write_asphalt_texture(texture_path)
    with open(material_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "newmtl asphalt\n"
            "Ka 0.20 0.21 0.23\n"
            "Kd 0.64 0.66 0.70\n"
            "Ks 0.00 0.00 0.00\n"
            "Ns 1.0\n"
            "d 1.0\n"
            "illum 2\n"
            f"map_Kd {texture_name}\n"
        )

    with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("# generated closed two-lane road strip\n")
        stream.write(f"mtllib {material_name}\nusemtl asphalt\n")
        for side in (right, left):
            for x, y, _yaw in side:
                # Keep the visual ribbon visibly above the ground plane.
                # A 3 mm separation still z-fought in Ogre2 at the shallow
                # QCar camera angle, making the asphalt appear white.
                stream.write(f"v {x:.6f} {y:.6f} 0.012000\n")
        cumulative = [0.0]
        for index in range(1, count):
            x0, y0, _ = points[index - 1]
            x1, y1, _ = points[index]
            cumulative.append(
                cumulative[-1] + math.hypot(x1 - x0, y1 - y0)
            )
        # Repeat the small texture every 0.4 m. Both sides share u so the
        # triangulated ribbon does not skew the texture across the lane.
        for side_v in (0.0, 1.0):
            for distance in cumulative:
                stream.write(f"vt {distance / 0.4:.6f} {side_v:.1f}\n")
        stream.write("vn 0.0 0.0 1.0\n")
        for index in range(count):
            following = (index + 1) % count
            right_now = index + 1
            right_next = following + 1
            left_now = count + index + 1
            left_next = count + following + 1
            # Counter-clockwise winding when viewed from above.
            stream.write(
                f"f {right_now}/{right_now}/1 "
                f"{right_next}/{right_next}/1 "
                f"{left_next}/{left_next}/1\n"
            )
            stream.write(
                f"f {right_now}/{right_now}/1 "
                f"{left_next}/{left_next}/1 "
                f"{left_now}/{left_now}/1\n"
            )


def build_road(
    csv_path,
    width,
    divider_offset_left,
    overtake_offset_left,
    road_mesh_uri,
    asphalt_texture_uri,
):
    """Render two lanes around the canonical normal-lane center.

    The canonical path itself is the center of the right/normal lane. The
    dashed divider is offset left, and the left/overtake-lane center is kept
    separate from the divider. This avoids the former visual error where
    both robots appeared to drive directly on the dashed paint.
    """
    if not 0.0 < divider_offset_left < overtake_offset_left:
        raise ValueError(
            "divider must be left of normal lane and right of overtake lane"
        )
    if width <= overtake_offset_left:
        raise ValueError("road width is too small for both lane centers")

    pts = load_path_csv(csv_path)
    road_center_left = 0.5 * overtake_offset_left
    divider_pts = offset_path(pts, divider_offset_left)

    # Solid markings sit just inside the asphalt. With defaults the normal
    # lane center is 0.00 m and the overtake lane center is +0.47 m.
    line_inset = 0.020
    right_edge_left = road_center_left - 0.5 * width + line_inset
    left_edge_left = road_center_left + 0.5 * width - line_inset
    right_edge_pts = offset_path(pts, right_edge_left)
    left_edge_pts = offset_path(pts, left_edge_left)

    visuals = [ROAD_MESH_TMPL.format(
        uri=road_mesh_uri,
        texture_uri=asphalt_texture_uri,
    )]
    step = 3

    for edge_name, edge_points in (
        ("right_edge", right_edge_pts),
        ("left_edge", left_edge_pts),
    ):
        for i in range(0, len(edge_points), step):
            cx, cy, yaw, length = closed_segment_geometry(edge_points, i, step)
            visuals.append(EDGE_TMPL.format(
                name=edge_name,
                i=i,
                x=cx,
                y=cy,
                yaw=yaw,
                seg_len=length * 1.18,
            ))

    for i in range(0, len(divider_pts), 8):
        x, y, yaw = divider_pts[i]
        visuals.append(DASH_TMPL.format(i=i, x=x, y=y, yaw=yaw))

    return (
        '    <model name="road">\n      <static>true</static>\n'
        '      <link name="link">\n' + "".join(visuals)
        + "      </link>\n    </model>\n"
    )

MARKER_TMPL = """    <model name="qcar_start_marker">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} 0.001 0 0 {yaw:.4f}</pose>
      <link name="link">
        <visual name="vis">
          <geometry><box><size>0.40 0.25 0.002</size></box></geometry>
          <material><ambient>0.1 0.8 0.1 1</ambient>
            <diffuse>0.1 0.9 0.1 0.8</diffuse></material>
        </visual>
      </link>
    </model>
"""


def load_obstacles_yaml(path):
    """Tiny YAML-list parser for obstacles.yaml (no PyYAML needed)."""
    if not path or not os.path.exists(path):
        return []
    obstacles = []
    current = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].rstrip()
            if not line.strip():
                continue
            if line.strip().startswith("- "):
                if current:
                    obstacles.append(current)
                current = {}
                line = line.strip()[2:]
            if ":" in line and current is not None:
                key, val = line.strip().split(":", 1)
                val = val.strip()
                current[key.strip()] = (
                    val if key.strip() == "name" else float(val)
                )
    if current:
        obstacles.append(current)
    return obstacles


def first_row_of_csv(path):
    return load_path_csv(path)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="map_server YAML")
    ap.add_argument("--obstacles", default=None, help="obstacles YAML")
    ap.add_argument("--rosbot-csv", default=None,
                    help="ROSbot trajectory CSV (spawns sim_rosbot at start)")
    ap.add_argument("--qcar-npy", default=None,
                    help="QCar trajectory .npy (places start marker)")
    ap.add_argument("--wall-height", type=float, default=0.30)
    ap.add_argument("--world-name", default="sami_track")
    ap.add_argument("--min-blob-cells", type=int, default=12,
                    help="drop occupied blobs smaller than this (SLAM noise)")
    ap.add_argument("--road-width", type=float, default=0.96)
    ap.add_argument("--divider-offset-left", type=float, default=0.21)
    ap.add_argument("--overtake-offset-left", type=float, default=0.47)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = load_map_yaml(args.map)
    pgm_path = os.path.join(os.path.dirname(args.map), meta["image"])
    width, height, pixels = load_pgm(pgm_path)
    grid = occupied_grid(
        width, height, pixels, meta["occupied_thresh"], meta["negate"]
    )
    grid = remove_small_blobs(grid, args.min_blob_cells)
    rects = merge_rectangles(grid)

    res = meta["resolution"]
    ox, oy = meta["origin"][0], meta["origin"][1]
    wh = args.wall_height

    boxes = []
    for i, (r0, c0, r1, c1) in enumerate(rects):
        sx = (c1 - c0 + 1) * res
        sy = (r1 - r0 + 1) * res
        cx = ox + (c0 + (c1 - c0 + 1) / 2.0) * res
        cy = oy + (height - r1 - 1 + (r1 - r0 + 1) / 2.0) * res
        boxes.append(
            BOX_TMPL.format(
                name=f"w{i}", x=cx, y=cy, z=wh / 2.0, sx=sx, sy=sy, sz=wh
            )
        )

    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extra = []
    if args.rosbot_csv and os.path.exists(args.rosbot_csv):
        road_mesh_path = os.path.splitext(output_path)[0] + "_road.obj"
        asphalt_texture_path = (
            os.path.splitext(output_path)[0] + "_asphalt.png"
        )
        write_road_mesh(
            args.rosbot_csv,
            args.road_width,
            args.overtake_offset_left,
            road_mesh_path,
            asphalt_texture_path,
        )
        # Keep the generated world portable. The launcher adds the world's
        # directory to GZ_SIM_RESOURCE_PATH so this sibling OBJ resolves in
        # both the repo and an installed qcar2_worlds package.
        road_mesh_uri = os.path.basename(road_mesh_path)
        asphalt_texture_uri = os.path.basename(asphalt_texture_path)
        extra.append(build_road(
            args.rosbot_csv,
            args.road_width,
            args.divider_offset_left,
            args.overtake_offset_left,
            road_mesh_uri,
            asphalt_texture_uri,
        ))

    for i, ob in enumerate(load_obstacles_yaml(args.obstacles)):
        extra.append(
            OBSTACLE_TMPL.format(
                name=ob.get("name", f"obstacle_{i}"),
                x=ob["x"], y=ob["y"],
                z=ob.get("sz", 0.30) / 2.0,
                yaw=ob.get("yaw", 0.0),
                sx=ob.get("sx", 0.15), sy=ob.get("sy", 0.25),
                sz=ob.get("sz", 0.30),
            )
        )

    if args.rosbot_csv and os.path.exists(args.rosbot_csv):
        rx, ry, ryaw = first_row_of_csv(args.rosbot_csv)
        extra.append(ROSBOT_TMPL.format(x=rx, y=ry, yaw=ryaw))

    if args.qcar_npy and os.path.exists(args.qcar_npy):
        import numpy as np
        traj = np.load(args.qcar_npy)
        extra.append(
            MARKER_TMPL.format(
                x=float(traj[0, 0]), y=float(traj[0, 1]), yaw=float(traj[0, 2])
            )
        )

    world = f"""<?xml version="1.0" ?>
<!-- AUTO-GENERATED by map_to_sdf.py — do not hand-edit; regenerate.
     Source map: {os.path.basename(args.map)} (origin {ox}, {oy}, res {res})
     Gazebo world frame == ROS map frame of that map.
     Canonical path == normal/right lane center; divider=+{args.divider_offset_left} m left;
     overtake lane center=+{args.overtake_offset_left} m left. -->
<sdf version="1.9">
  <world name="{args.world_name}">
    <physics name="default" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.3 0.2 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="col">
          <geometry><plane><normal>0 0 1</normal>
            <size>60 60</size></plane></geometry>
        </collision>
        <visual name="vis">
          <geometry><plane><normal>0 0 1</normal>
            <size>60 60</size></plane></geometry>
          <material><ambient>0.24 0.34 0.22 1</ambient>
            <diffuse>0.32 0.46 0.29 1</diffuse></material>
        </visual>
      </link>
    </model>

    <model name="track_walls">
      <static>true</static>
      <link name="walls">
{''.join(boxes)}      </link>
    </model>

{''.join(extra)}  </world>
</sdf>
"""
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(world)
    print(
        f"[map_to_sdf] {len(rects)} wall boxes, {len(extra)} extra models "
        f"-> {args.out}"
    )
    print(f"[map_to_sdf] world frame == map frame (origin {ox}, {oy})")


if __name__ == "__main__":
    sys.exit(main())
