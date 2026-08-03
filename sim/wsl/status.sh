#!/bin/bash
source /opt/ros/jazzy/setup.bash
export ROS_LOCALHOST_ONLY=1

echo "=== gz GUI screenshot ==="
gz service -s /gui/screenshot \
  --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'data: "/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/gz_overview.png"' 2>&1 | tail -1
sleep 2
ls -la /mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/gz_overview.png 2>/dev/null \
  || echo "(no screenshot file — GUI service may differ in gz8)"

echo
echo "=== sim_rosbot log tail ==="
tail -4 /tmp/simrosbot.log

echo
echo "=== model poses ==="
gz topic -e -t /world/sami_track/pose/info -n 1 2>/dev/null \
  | grep -E -A4 'name: "(sim_rosbot|qcar2)"' | head -24
