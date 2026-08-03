#!/usr/bin/env python3

import math
import sys

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage
except ImportError as e:
    print(f"[ERROR] Missing ROS2 dependency: {e}")
    print("Run: source /opt/ros/humble/setup.bash")
    sys.exit(1)


def quat_to_yaw(x, y, z, w):
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def multiply_transforms(t1, t2):
    x1, y1, yaw1 = t1
    x2, y2, yaw2 = t2

    x = x1 + math.cos(yaw1) * x2 - math.sin(yaw1) * y2
    y = y1 + math.sin(yaw1) * x2 + math.cos(yaw1) * y2
    yaw = wrap_angle(yaw1 + yaw2)

    return x, y, yaw


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 check_bag_end.py /path/to/rosbag_folder")
        sys.exit(1)

    bag_path = sys.argv[1]

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=["/tf", "/tf_static"]
        )
    )

    tf_store = {}
    poses = []

    while reader.has_next():
        topic, rawdata, timestamp = reader.read_next()

        msg = deserialize_message(rawdata, TFMessage)

        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id

            tr = transform.transform.translation
            q = transform.transform.rotation

            tf_store[(parent, child)] = (
                tr.x,
                tr.y,
                quat_to_yaw(q.x, q.y, q.z, q.w),
            )

        # Same chain used by your path extractor
        t_map_odom = tf_store.get(("map", "odom"))
        t_odom_base = tf_store.get(("odom", "base_link"))

        if t_map_odom is not None and t_odom_base is not None:
            x, y, yaw = multiply_transforms(
                t_map_odom,
                t_odom_base,
            )

            poses.append((
                timestamp * 1e-9,
                x,
                y,
                yaw,
            ))

    if len(poses) < 2:
        print("Could not find enough chained transforms.")
        print("Available TF pairs found:")

        for pair in sorted(tf_store.keys()):
            print(f"  {pair[0]} -> {pair[1]}")

        raise RuntimeError(
            "No enough map->odom + odom->base_link transforms"
        )

    poses = np.array(poses)

    start = poses[0]
    end = poses[-1]

    gap = math.hypot(
        end[1] - start[1],
        end[2] - start[2],
    )

    yaw_diff = wrap_angle(
        end[3] - start[3]
    )

    print(f"Extracted poses: {len(poses)}")
    print(
        f"Start: x={start[1]:.4f}, "
        f"y={start[2]:.4f}, "
        f"yaw={math.degrees(start[3]):.2f} deg"
    )
    print(
        f"End:   x={end[1]:.4f}, "
        f"y={end[2]:.4f}, "
        f"yaw={math.degrees(end[3]):.2f} deg"
    )
    print(f"End-start gap: {gap:.4f} m")
    print(f"Yaw difference: {math.degrees(yaw_diff):.2f} deg")

    if gap < 0.05:
        print("Result: raw bag path is closed well (<5 cm).")
    elif gap < 0.15:
        print("Result: raw bag path is reasonably close.")
    else:
        print("Result: raw bag path has a significant gap.")


if __name__ == "__main__":
    main()