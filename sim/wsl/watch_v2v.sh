#!/bin/bash
# One-shot V2V status: topic presence, then live values.
source /opt/ros/jazzy/setup.bash
source /home/hammad/rosbot_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1

echo "=== /v2v topics present ==="
timeout 8 ros2 topic list 2>/dev/null | grep v2v

for t in /v2v/stats /v2v/alive /v2v/gap /v2v/rosbot_speed /v2v/on_path; do
  echo "--- $t ---"
  timeout 8 ros2 topic echo "$t" --once 2>/dev/null | head -4
done

echo "=== receiver CSV log tail ==="
tail -3 /tmp/v2v_rx_log.csv 2>/dev/null || echo "(no log yet)"
