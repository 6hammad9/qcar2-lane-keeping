#!/bin/bash
# Measure the actual V2V packet rate and freshness over a 10 s window.
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
export ROS_LOCALHOST_ONLY=1

s1=$(timeout 8 ros2 topic echo /v2v/stats --once 2>/dev/null | head -2 | tail -1)
sleep 10
s2=$(timeout 8 ros2 topic echo /v2v/stats --once 2>/dev/null | head -2 | tail -1)

echo "t0: $s1"
echo "t1: $s2"
python3 - "$s1" "$s2" <<'EOF'
import json, re, sys
def parse(s):
    m = re.search(r"\{.*\}", s)
    return json.loads(m.group(0)) if m else None
a, b = parse(sys.argv[1]), parse(sys.argv[2])
if not a or not b:
    print("could not parse stats"); raise SystemExit
d = b["rx"] - a["rx"]
print()
print("packets in ~10 s: %d  -> %.1f Hz (target 10)" % (d, d / 10.0))
print("parse_errors: %d   seq_drops: %d" % (b["parse_errors"], b["seq_drops"]))
print("age at t1: %.3f s (stale threshold 0.6)" % b["age_s"])
print("gap: %.2f m   on_path: %s" % (b["gap"], b["on_path"]))
EOF
echo "--- alive samples ---"
for i in 1 2 3; do
  timeout 5 ros2 topic echo /v2v/alive --once 2>/dev/null | head -1
done
