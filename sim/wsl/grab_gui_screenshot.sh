#!/bin/bash
# Save the current Gazebo GUI view into a directory (Gazebo picks the name).
set -e
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros
output_dir="${1:-$SIM_DIR}"
request="data: \"$output_dir\""
gz service -s /gui/screenshot \
  --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean \
  --timeout 5000 --req "$request"
