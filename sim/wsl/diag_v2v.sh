#!/bin/bash
# Who is ahead of whom, how fast, and is the governor able to act?
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
export ROS_LOCALHOST_ONLY=1

echo "=== last receiver samples (qcar_idx, rosbot_idx, gap) ==="
tail -3 /tmp/v2v_rx_log.csv 2>/dev/null | awk -F, '{printf "  qcar_idx=%s rosbot_idx=%s gap=%s on_path=%s v_rosbot=%s\n",$10,$11,$12,$14,$7}'

echo
echo "=== speeds ==="
echo -n "  ROSbot commanded: "
pgrep -af sim_rosbot.py | grep -oE '\-\-speed [0-9.]+' | head -1
echo -n "  QCar max_speed:   "
pgrep -af path_mpc_node | grep -oE 'max_speed:=[0-9.]+' | head -1
echo -n "  QCar actual v:    "
grep -oE ' v=[0-9.]+' /tmp/mpc.log | tail -1

echo
echo "=== separation in INDEX terms (n=540, 0.05 m/wp) ==="
python3 - <<'EOF'
import csv
try:
    rows=list(csv.DictReader(open('/tmp/v2v_rx_log.csv')))[-1:]
except Exception as e:
    print("  no log:", e); raise SystemExit
for r in rows:
    q=int(r['qcar_idx']); b=int(r['rosbot_idx']); n=540
    ahead=(b-q)%n
    behind=(q-b)%n
    print("  ROSbot is %d wp AHEAD (%.2f m) or %d wp BEHIND (%.2f m) of the QCar"
          %(ahead,ahead*0.05,behind,behind*0.05))
    print("  -> reported gap %.2f m (loop-forward convention)"%(float(r['gap'])))
    if behind < ahead:
        print("  !! ROSbot is actually BEHIND and closing. The QCar's V2V")
        print("     governor only reacts to vehicles AHEAD, so it will not act.")
EOF
