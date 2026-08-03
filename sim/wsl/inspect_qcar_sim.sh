#!/bin/bash
WS=/home/hammad/rosbot_ws/src/qcar2_autonomous_lanes
echo "=== packages ==="
ls "$WS"
echo "=== launch files ==="
find "$WS" -name '*.launch.py'
echo "=== sim_bringup.launch.py ==="
F=$(find "$WS" -name 'sim_bringup.launch.py' | head -1)
echo "FILE: $F"
sed -n '1,120p' "$F"
