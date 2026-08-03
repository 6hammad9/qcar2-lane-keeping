#!/bin/bash
# Install the generated real-map world into the qcar2_worlds package so
# sim_bringup.launch.py can find it by name (world:=sami_track).
set -e
SRC=/home/hammad/rosbot_ws/src/qcar2_autonomous_lanes/qcar2_worlds/worlds
INST=/home/hammad/rosbot_ws/install/qcar2_worlds/share/qcar2_worlds/worlds
GEN=/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/worlds/sami_track.sdf
GEN_MESH=/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/worlds/sami_track_road.obj
GEN_MATERIAL=/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/worlds/sami_track_road.mtl
GEN_TEXTURE=/mnt/e/WSL/Ubuntu-24.04/qcar-izhan/sim/worlds/sami_track_asphalt.png

cp "$GEN" "$SRC/sami_track.sdf"
cp "$GEN_MESH" "$SRC/sami_track_road.obj"
cp "$GEN_MATERIAL" "$SRC/sami_track_road.mtl"
cp "$GEN_TEXTURE" "$SRC/sami_track_asphalt.png"
echo "installed -> $SRC/sami_track.sdf"
if [ -d "$INST" ]; then
  cp "$GEN" "$INST/sami_track.sdf"
  cp "$GEN_MESH" "$INST/sami_track_road.obj"
  cp "$GEN_MATERIAL" "$INST/sami_track_road.mtl"
  cp "$GEN_TEXTURE" "$INST/sami_track_asphalt.png"
  echo "installed -> $INST/sami_track.sdf"
fi
ls -la "$SRC" | grep sami
