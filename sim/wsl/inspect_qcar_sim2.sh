#!/bin/bash
WS=/home/hammad/rosbot_ws
SRC=$WS/src/qcar2_autonomous_lanes
echo "=== sim_world.launch.py ==="
sed -n '1,60p' "$SRC/qcar2_worlds/launch/sim_world.launch.py"
echo
echo "=== bridge.yaml ==="
cat "$SRC/qcar2_bringup/config/bridge.yaml"
echo
echo "=== worlds dir ==="
ls "$SRC/qcar2_worlds/worlds/"
echo "=== workspace built? ==="
ls "$WS/install" 2>/dev/null | head
echo "=== casadi ==="
python3 -c "import casadi; print('casadi', casadi.__version__)" 2>&1 | tail -1
