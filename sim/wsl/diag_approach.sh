#!/bin/bash
# Did the V2V governor / overtake actually engage during the approach?
LOG=${1:-/tmp/mpc.log}
echo "=== smallest gaps seen ==="
grep -oE 'v2v_gap=[0-9.]+' "$LOG" | sed 's/v2v_gap=//' | sort -n | head -5 | tr '\n' ' '
echo
echo "=== governor engagements (v2v_cap >= 0) ==="
n=$(grep -cE 'v2v_cap=[0-9]' "$LOG")
echo "  $n samples"
grep -E 'v2v_cap=[0-9]' "$LOG" | tail -6 \
  | grep -oE 'idx=[0-9]+.*v2v_gap=[0-9.]+, v2v_cap=[0-9.]+' \
  | sed 's/lane=[^,]*, lane_valid=[^,]*, lane_active=[^,]*, //'

echo
echo "=== drive states seen ==="
grep -oE 'state=[A-Z_]+' "$LOG" | sort | uniq -c

echo
echo "=== LiDAR node view (obstacle / v2v injection) ==="
grep -oE 'state=[A-Z_]+ \| obs=[A-Za-z]+ .*v2v=[A-Z_a-z]+ \| v2v_gap=[0-9.-]+' \
  /tmp/lidar.log 2>/dev/null | tail -4 || echo "  (no lidar log lines)"
