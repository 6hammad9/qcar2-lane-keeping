#!/bin/bash
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
printf '%-15s %-8s %s\n' COMPONENT STATE PID
for name in sim pose_bridge receiver amcl lidar rosbot lane cmd_bridge mpc; do
  if component_running "$name"; then
    printf '%-15s %-8s %s\n' "$name" UP "$(cat "$(pid_file "$name")")"
  else
    printf '%-15s %-8s %s\n' "$name" DOWN -
  fi
done
echo "logs: $SIM_LOG_DIR"

