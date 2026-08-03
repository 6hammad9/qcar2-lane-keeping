#!/bin/bash
# Shared paths, ROS environment and PID-file process supervision for the sim.

WSL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "$WSL_DIR/.." && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"
QCAR_SIM_WS="${QCAR_SIM_WS:-/home/hammad/rosbot_ws}"
SIM_STATE_DIR="${SIM_STATE_DIR:-/tmp/qcar_v2v_sim_${UID}}"
SIM_LOG_DIR="$SIM_STATE_DIR/logs"
mkdir -p "$SIM_STATE_DIR" "$SIM_LOG_DIR"

setup_ros() {
  source /opt/ros/jazzy/setup.bash
  if [ -f "$QCAR_SIM_WS/install/setup.bash" ]; then
    source "$QCAR_SIM_WS/install/setup.bash"
  fi
  # Keep the simulation on the explicit default domain even if a previous
  # real-robot shell exported another value. Override only through the
  # simulation-specific variable when deliberately running two stacks.
  export ROS_DOMAIN_ID="${SIM_ROS_DOMAIN_ID:-0}"
  export ROS_LOCALHOST_ONLY=1
  export PYTHONPATH="$REPO_DIR/src/qcar_science_night_pkg${PYTHONPATH:+:$PYTHONPATH}"
}

pid_file() {
  printf '%s/%s.pid\n' "$SIM_STATE_DIR" "$1"
}

component_running() {
  local file pid
  file="$(pid_file "$1")"
  [ -s "$file" ] || return 1
  pid="$(cat "$file" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_component() {
  local name="$1"
  shift
  if component_running "$name"; then
    echo "ERROR: component '$name' is already running (PID $(cat "$(pid_file "$name")"))." >&2
    return 1
  fi
  rm -f "$(pid_file "$name")"
  # Close the caller's orchestration-lock descriptor in the child. Otherwise
  # a background component would keep the start/stop flock held forever.
  setsid "$@" 9>&- >"$SIM_LOG_DIR/$name.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$(pid_file "$name")"
  sleep 0.15
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: component '$name' exited; see $SIM_LOG_DIR/$name.log" >&2
    rm -f "$(pid_file "$name")"
    return 1
  fi
  printf '  %-14s PID %-7s log %s/%s.log\n' "$name" "$pid" "$SIM_LOG_DIR" "$name"
}

stop_component() {
  local name="$1" file pid deadline
  file="$(pid_file "$name")"
  [ -s "$file" ] || return 0
  pid="$(cat "$file" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    # Every component is launched under setsid, so the negative PID targets
    # only that component's process group (including launch children).
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
    deadline=$((SECONDS + 5))
    while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
      sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    fi
  fi
  rm -f "$file"
}
