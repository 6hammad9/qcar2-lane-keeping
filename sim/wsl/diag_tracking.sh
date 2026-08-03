#!/bin/bash
# Why is the QCar tracking badly? Speed vs curve target, steering saturation.
LOG=${1:-/tmp/mpc.log}
echo "=== last 5 control samples ==="
grep -oE 'idx=[0-9]+, track_err=[0-9.]+, yaw_err=[-0-9.]+, v=[0-9.]+, target_v=[0-9.]+, delta=[-0-9.]+' "$LOG" | tail -5

echo
echo "=== tracking error ==="
grep -oE 'track_err=[0-9.]+' "$LOG" | sed 's/track_err=//' | sort -rn \
  | awk 'NR==1{max=$1} {a[NR]=$1} END{printf "  worst %.3f m, median %.3f m, n=%d\n", max, a[int(NR/2)], NR}'

echo
echo "=== steering saturation (|delta| >= 0.45 rad) ==="
total=$(grep -c 'delta=' "$LOG")
sat=$(grep -cE 'delta=-?0\.(4[5-9]|5[0-9])' "$LOG")
echo "  $sat / $total samples"

echo
echo "=== speed vs curve target ==="
paste <(grep -oE 'v=[0-9.]+,' "$LOG" | sed 's/v=//;s/,//') \
      <(grep -oE 'target_v=[0-9.]+' "$LOG" | sed 's/target_v=//') \
  | awk '{n++; if($1>$2+0.02) over++} END{printf "  commanded speed exceeded target in %d/%d samples (%.0f%%)\n", over, n, 100*over/n}'
