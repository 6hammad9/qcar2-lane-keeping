#!/usr/bin/env python3
"""Regression tests for the generated track and shared canonical path."""

import os
import struct
import sys
import unittest
import xml.etree.ElementTree as ET
import zlib

import numpy as np


SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from build_canonical_path import build_canonical_path, load_recorded_csv
from map_to_sdf import load_path_csv, offset_path
from validate_sim_geometry import validate


ASSETS = os.path.join(SIM_DIR, "assets")


class SimGeometryTest(unittest.TestCase):
    def test_source_tail_is_removed_and_loop_is_steerable(self):
        source = load_recorded_csv(
            os.path.join(ASSETS, "rosbot_recorded_trajectory.csv")
        )
        trajectory, report = build_canonical_path(source)
        self.assertEqual(report["seam_source_index"], 527)
        self.assertEqual(report["removed_tail_points"], 12)
        seam_gap = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])
        self.assertLess(seam_gap, 0.075)

        segment = np.roll(trajectory[:, :2], -1, axis=0) - trajectory[:, :2]
        length = np.linalg.norm(segment, axis=1)
        heading = np.arctan2(segment[:, 1], segment[:, 0])
        heading_delta = np.arctan2(
            np.sin(np.roll(heading, -1) - heading),
            np.cos(np.roll(heading, -1) - heading),
        )
        curvature = np.abs(heading_delta) / (
            0.5 * (length + np.roll(length, -1))
        )
        required_steer = np.arctan(0.256 * curvature)
        # Leave margin below the external Gazebo Ackermann plugin's observed
        # central-steer ceiling of approximately 0.464 rad.
        self.assertLessEqual(float(np.max(required_steer)), 0.44)

    def test_generated_paths_and_corridors_validate(self):
        report = validate(
            os.path.join(ASSETS, "smoothed_trajectory.csv"),
            os.path.join(ASSETS, "qcar_half_map_centerline.npy"),
            os.path.join(ASSETS, "track_map.yaml"),
        )
        self.assertTrue(report["ok"], report["failures"])
        self.assertTrue(report["canonical_files_identical"])
        self.assertTrue(report["corridors"]["normal_lane_center"]["ok"])
        self.assertTrue(report["corridors"]["overtake_lane_center"]["ok"])
        self.assertTrue(
            report["corridors"]["commanded_overtake_reference_limit"]["ok"]
        )
        # The canonical line is within 7.5 mm of the exact midpoint between
        # the right solid edge and dashed divider.
        self.assertAlmostEqual(
            report["lane_geometry_m"]["nominal_from_right_lane_midpoint"],
            0.0075,
            places=6,
        )

    def test_world_has_two_lane_paint_and_no_default_static_obstacles(self):
        world_path = os.path.join(SIM_DIR, "worlds", "sami_track.sdf")
        with open(world_path, "r", encoding="utf-8") as stream:
            root = ET.fromstring(stream.read())
        world = root.find("world")
        models = {model.attrib["name"]: model for model in world.findall("model")}
        self.assertNotIn("box_on_path_wp350", models)
        self.assertNotIn("clutter_off_path", models)
        self.assertEqual(
            len(models["track_walls"].findall("./link/collision")), 287
        )
        wall_visuals = models["track_walls"].findall("./link/visual")
        self.assertEqual(len(wall_visuals), 287)
        self.assertTrue(all(
            visual.findtext("transparency") == "1.0"
            and visual.findtext("cast_shadows") == "false"
            for visual in wall_visuals
        ))

        visuals = {
            visual.attrib["name"]: visual
            for visual in models["road"].findall("./link/visual")
        }
        self.assertTrue(any(name.startswith("right_edge_") for name in visuals))
        self.assertTrue(any(name.startswith("left_edge_") for name in visuals))
        self.assertIn("divider_dash_0", visuals)

        points = load_path_csv(
            os.path.join(ASSETS, "smoothed_trajectory.csv")
        )
        expected = offset_path(points, 0.21)[0]
        pose = [float(value) for value in visuals["divider_dash_0"].find("pose").text.split()]
        self.assertAlmostEqual(pose[0], expected[0], places=4)
        self.assertAlmostEqual(pose[1], expected[1], places=4)

        road_material = visuals["road_surface"].find("material")
        self.assertEqual(
            road_material.find("./pbr/metal/albedo_map").text,
            "sami_track_asphalt.png",
        )
        self.assertEqual(
            float(road_material.find("./pbr/metal/roughness").text), 1.0
        )

    def test_rosbot_proxy_is_self_contained_and_visibly_identifiable(self):
        world_path = os.path.join(SIM_DIR, "worlds", "sami_track.sdf")
        root = ET.parse(world_path).getroot()
        world = root.find("world")
        models = {model.attrib["name"]: model for model in world.findall("model")}
        rosbot = models["sim_rosbot"]
        body = rosbot.find("link[@name='body']")

        # This is deliberately one kinematically driven link, not a fake
        # differential-drive implementation. Its required visuals make the
        # proxy easy to distinguish from QCar and from static obstacles.
        self.assertEqual(len(rosbot.findall("link")), 1)
        self.assertEqual(len(rosbot.findall("joint")), 0)
        self.assertEqual(len(rosbot.findall("plugin")), 0)
        self.assertEqual(len(body.findall("collision")), 1)
        visuals = {visual.attrib["name"] for visual in body.findall("visual")}
        required = {
            "chassis_lower",
            "electronics_enclosure",
            "sensor_mast",
            "lidar_puck",
            "front_camera",
            "front_camera_lens",
            "front_direction_marker",
            "wheel_fl",
            "wheel_fr",
            "wheel_rl",
            "wheel_rr",
        }
        self.assertTrue(required.issubset(visuals), required - visuals)

    def test_asphalt_texture_is_grey_and_mesh_has_uvs_and_normals(self):
        texture_path = os.path.join(
            SIM_DIR, "worlds", "sami_track_asphalt.png"
        )
        with open(texture_path, "rb") as stream:
            png = stream.read()
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

        cursor = 8
        compressed = bytearray()
        width = height = None
        while cursor < len(png):
            length = struct.unpack(">I", png[cursor:cursor + 4])[0]
            kind = png[cursor + 4:cursor + 8]
            payload = png[cursor + 8:cursor + 8 + length]
            cursor += 12 + length
            if kind == b"IHDR":
                width, height = struct.unpack(">II", payload[:8])
            elif kind == b"IDAT":
                compressed.extend(payload)
        pixels = zlib.decompress(bytes(compressed))
        rows = np.frombuffer(pixels, dtype=np.uint8).reshape(height, 1 + 3 * width)
        self.assertTrue(np.all(rows[:, 0] == 0))
        self.assertGreaterEqual(int(np.min(rows[:, 1:])), 90)
        self.assertLess(int(np.max(rows[:, 1:])), 170)

        mesh_path = os.path.join(
            SIM_DIR, "worlds", "sami_track_road.obj"
        )
        with open(mesh_path, "r", encoding="utf-8") as stream:
            mesh = stream.read()
        self.assertIn("\nvt ", mesh)
        self.assertIn("\nvn 0.0 0.0 1.0\n", mesh)
        self.assertRegex(mesh, r"\nf \d+/\d+/1 ")


if __name__ == "__main__":
    unittest.main()
