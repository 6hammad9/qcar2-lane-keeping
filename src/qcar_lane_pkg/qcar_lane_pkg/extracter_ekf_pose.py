#!/usr/bin/env python3
import rclpy
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
import rosbag2_py
import csv
from tf_transformations import euler_from_quaternion

def extract_waypoints(bag_path, downsample=10):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    waypoints = []
    initial_pose = None
    count = 0

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/ekf_pose_estimate':
            msg = deserialize_message(data, PoseStamped)
            _, _, theta = euler_from_quaternion([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            if initial_pose is None:
                initial_pose = [msg.pose.position.x, msg.pose.position.y, theta]
            if count % downsample == 0:
                waypoints.append([msg.pose.position.x, msg.pose.position.y, theta])
            count += 1

    # Save waypoints
    with open('waypoints.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'theta'])
        writer.writerows(waypoints)

    # Save initial pose
    with open('initial_pose.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'theta'])
        writer.writerow(initial_pose)

    print(f'Initial pose: {initial_pose}')
    print(f'Saved {len(waypoints)} waypoints')

if __name__ == '__main__':
    extract_waypoints('/home/nvidia/ros2_ws/rosbag2_2026_04_22-17_29_58')  # replace with your bag folder name