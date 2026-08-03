#!/bin/bash
# Put the Gazebo GUI close behind QCar so both the car and road are visible.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
setup_ros

for _ in $(seq 1 50); do
  if gz topic -l 2>/dev/null | grep -qx '/gui/track'; then
    gz topic -t /gui/track -m gz.msgs.CameraTrack -p '
      track_mode: FOLLOW_LOOK_AT,
      follow_target: {name: "qcar2"},
      track_target: {name: "sim_rosbot"},
      follow_offset: {x: -1.10, y: 0.0, z: 0.80},
      track_offset: {x: 0.0, y: 0.0, z: 0.08},
      follow_pgain: 0.9,
      track_pgain: 0.9' >/dev/null
    echo "Gazebo camera follows QCar2 and looks toward ROSbot."
    exit 0
  fi
  sleep 0.2
done

echo "Gazebo GUI tracking topic unavailable; camera was not changed." >&2
exit 1
