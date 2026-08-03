#!/bin/bash
# Continuous human-readable QCar/ROSbot decision and command dashboard.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
exec python3 "$SIM_DIR/v2v_live_monitor.py"
