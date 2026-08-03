#!/usr/bin/env python3
"""Fake ROSbot — replays a trajectory over UDP in the real V2V wire format.

Pure Python (no ROS required), so it runs anywhere: WSL, a laptop, or the
QCar itself. Lets you bench-test the entire QCar-side V2V chain — receiver,
LiDAR fusion, MPC governor and DCBF — without powering the ROSbot.

Typical bench test (everything on the QCar, no robot motion needed):

    # terminal A on the QCar
    ros2 run qcar_science_night_pkg v2v_receiver

    # terminal B: virtual ROSbot creeping along the QCar's own path
    python3 v2v_fake_rosbot.py \
        --path ~/ros2_ws/recorded_path_amcl_final_long.npy \
        --target 127.0.0.1 --speed 0.25 --start-idx 200

    # terminal C: watch the receiver's outputs
    ros2 topic echo /v2v/gap

Supports .npy trajectories (x, y, yaw, curvature — the MPC format) and .csv
(x, y, theta — the ROSbot format). `--pause-every / --pause-for` emulate the
ROSbot's waypoint pauses so the stationary-obstacle behavior can be tested.
"""

import argparse
import csv
import json
import math
import socket
import time

SCHEMA_VERSION = 1
MAX_PACKET_BYTES = 1400


def load_path(path):
    if path.endswith(".npy"):
        import numpy as np
        data = np.load(path)
        return [(float(r[0]), float(r[1]), float(r[2])) for r in data]
    with open(path) as fh:
        reader = csv.DictReader(fh)
        key = None
        rows = []
        for row in reader:
            if key is None:
                key = "theta" if "theta" in row else "yaw"
            rows.append((float(row["x"]), float(row["y"]), float(row[key])))
        return rows


def cumulative(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(
            cum[-1]
            + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        )
    return cum


def pose_at_arc(pts, cum, arc, loop):
    total = cum[-1]
    if loop and total > 0:
        arc %= total
    else:
        arc = min(arc, total)
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < arc:
            lo = mid + 1
        else:
            hi = mid
    return pts[lo]


def pack(vehicle_id, seq, x, y, yaw, v, moving, hdt, predicted):
    predicted = [
        (round(px, 3), round(py, 3), round(pth, 3))
        for px, py, pth in predicted
    ]
    while True:
        msg = {
            "s": SCHEMA_VERSION,
            "id": vehicle_id,
            "q": seq,
            "t": round(time.time(), 3),
            "loc": 1,
            "x": round(x, 3),
            "y": round(y, 3),
            "th": round(yaw, 3),
            "v": round(v, 3),
            "ms": "M" if moving else "S",
            "hdt": round(hdt, 3),
            "p": [c for pt in predicted for c in pt],
        }
        data = json.dumps(msg, separators=(",", ":")).encode()
        if len(data) <= MAX_PACKET_BYTES or not predicted:
            return data
        predicted = predicted[: max(1, len(predicted) - 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", required=True, help=".npy or .csv trajectory")
    ap.add_argument("--target", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--speed", type=float, default=0.25, help="m/s")
    ap.add_argument("--rate", type=float, default=10.0, help="Hz")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--vehicle-id", default="rosbot3")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--hdt", type=float, default=0.08)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--pause-every", type=float, default=0.0,
                    help="pause every N seconds of driving (0 = never)")
    ap.add_argument("--pause-for", type=float, default=5.0,
                    help="pause duration [s]")
    args = ap.parse_args()

    pts = load_path(args.path)
    cum = cumulative(pts)
    loop = not args.no_loop
    print(f"[fake-rosbot] {len(pts)} pts, {cum[-1]:.1f} m, "
          f"-> udp://{args.target}:{args.port} @ {args.rate:.0f} Hz, "
          f"v={args.speed} m/s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    arc = cum[min(args.start_idx, len(cum) - 1)]
    seq = 0
    dt = 1.0 / args.rate
    drive_timer = 0.0
    paused_until = 0.0

    try:
        while True:
            now = time.monotonic()
            paused = now < paused_until
            v = 0.0 if paused else args.speed

            if not paused and args.pause_every > 0:
                drive_timer += dt
                if drive_timer >= args.pause_every:
                    drive_timer = 0.0
                    paused_until = now + args.pause_for
                    print(f"[fake-rosbot] pausing {args.pause_for}s")

            x, y, yaw = pose_at_arc(pts, cum, arc, loop)
            predicted = [
                pose_at_arc(pts, cum, arc + v * args.hdt * k, loop)
                for k in range(args.horizon)
            ]

            sock.sendto(
                pack(args.vehicle_id, seq, x, y, yaw, v, v > 0,
                     args.hdt, predicted),
                (args.target, args.port),
            )
            seq += 1
            arc += v * dt
            if seq % 50 == 0:
                print(f"[fake-rosbot] seq={seq} arc={arc:.1f} m "
                      f"pos=({x:.2f},{y:.2f}) v={v}")
            time.sleep(dt)
    except KeyboardInterrupt:
        print(f"\n[fake-rosbot] stopped after {seq} packets")


if __name__ == "__main__":
    main()
