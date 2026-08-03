#!/bin/bash
# A Gazebo teleport leaves wheel odometry and map->odom inconsistent.  A safe
# reset therefore restarts the complete supervised simulation at its declared
# initial conditions.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
[ "$#" -eq 0 ] || {
  echo "positional teleport arguments are no longer supported" >&2
  exit 2
}
bash "$DIR/stop_all.sh"
exec bash "$DIR/start_all.sh"

