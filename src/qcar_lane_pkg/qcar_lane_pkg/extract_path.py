
import sqlite3
import json
import math
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

db = '/home/nvidia/ros2_ws/qcar_camera_bag/rosbag2_2026_04_24-15_30_19/rosbag2_2026_04_24-15_30_19_0.db3'
conn = sqlite3.connect(db)
cursor = conn.cursor()

# Get topic id for /tf
cursor.execute("SELECT id FROM topics WHERE name='/tf'")
topic_id = cursor.fetchone()[0]

cursor.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,))
rows = cursor.fetchall()
conn.close()

map_to_odom = None
odom_to_base = None
poses = []

def multiply_transforms(t1, r1, t2, r2):
    yaw1 = 2 * math.atan2(r1[2], r1[3])
    x = t1[0] + t2[0] * math.cos(yaw1) - t2[1] * math.sin(yaw1)
    y = t1[1] + t2[0] * math.sin(yaw1) + t2[1] * math.cos(yaw1)
    yaw2 = 2 * math.atan2(r2[2], r2[3])
    yaw = yaw1 + yaw2
    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    return x, y, qz, qw

for (data,) in rows:
    msg = deserialize_message(data, TFMessage)
    for t in msg.transforms:
        tr = t.transform.translation
        ro = t.transform.rotation

        if t.header.frame_id == 'map' and t.child_frame_id == 'odom':
            map_to_odom = (tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w)

        elif t.header.frame_id == 'odom' and t.child_frame_id == 'base_link':
            odom_to_base = (tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w)

    if map_to_odom and odom_to_base:
        x, y, qz, qw = multiply_transforms(
            map_to_odom[:3], map_to_odom[3:],
            odom_to_base[:3], odom_to_base[3:]
        )

        if poses:
            last = poses[-1]
            dist = math.sqrt((x - last['x'])**2 + (y - last['y'])**2)
            if dist < 0.02:
                continue

        poses.append({'x': x, 'y': y, 'qz': qz, 'qw': qw})

print(f'Extracted {len(poses)} poses')
print(f'First: {poses[0]}')
print(f'Last:  {poses[-1]}')

with open('/home/nvidia/ros2_ws/recorded_path_latest1.json', 'w') as f:
    json.dump(poses, f, indent=2)
print('Saved to /home/nvidia/ros2_ws/recorded_path_latest.json')
