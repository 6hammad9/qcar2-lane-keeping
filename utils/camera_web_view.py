#!/usr/bin/env python3
"""Serve a ROS image topic as MJPEG over HTTP, for machines with no GUI.

Written because rqt_image_view needs an X display and the QCar is normally
reached over plain SSH. Any browser can open an MJPEG stream, so this needs
nothing installed beyond what the ROS image pipeline already pulls in.

    python3 camera_web_view.py --topic /camera/color_image --port 8080

then browse to http://<car-ip>:8080/ from the laptop.

The stream is served over plain HTTP with no authentication, so it exposes
the camera to anything that can reach the port. Run it on a trusted lab
network only, and stop it when you are done.
"""

import argparse
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the robot's image
    cv2 = None


_LATEST = {"jpeg": None}
_LOCK = threading.Lock()


def to_bgr(msg):
    """Decode the common sensor_msgs/Image encodings into a BGR array.

    Deliberately avoids cv_bridge: it is an extra dependency that is easy to
    have mismatched against the local OpenCV, and only a few encodings ever
    appear on this robot.
    """
    height, width = msg.height, msg.width
    enc = msg.encoding.lower()
    if height == 0 or width == 0 or len(msg.data) == 0:
        raise ValueError(
            f"empty frame ({width}x{height}, {len(msg.data)} bytes) -- the "
            "camera is publishing but producing nothing"
        )

    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3)
        bgr = img[:, :, ::-1] if enc == "rgb8" else img
    elif enc in ("rgba8", "bgra8"):
        img = buf.reshape(height, width, 4)[:, :, :3]
        bgr = img[:, :, ::-1] if enc == "rgba8" else img
    elif enc in ("mono8", "8uc1"):
        bgr = np.dstack([buf.reshape(height, width)] * 3)
    elif enc in ("16uc1", "mono16"):
        # Depth. Scale to 8 bit for display only -- this is not a measurement.
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
        top = float(img.max()) or 1.0
        img8 = (img.astype(np.float32) * (255.0 / top)).astype(np.uint8)
        bgr = np.dstack([img8] * 3)
    else:
        raise ValueError(f"unhandled encoding {msg.encoding!r}")

    # The channel-reversing slice above yields a NEGATIVE-stride view, and
    # cv2.imencode cannot wrap one -- it reports the Mat as empty rather than
    # as badly strided, which is a confusing way to fail. Copy to a
    # contiguous buffer before handing it to OpenCV.
    return np.ascontiguousarray(bgr)


class Relay(Node):

    def __init__(self, topic, quality):
        super().__init__("camera_web_view")
        self.quality = int(quality)
        self.frames = 0
        self.errors = 0
        self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )
        self.create_timer(5.0, self._report)
        self.get_logger().info(f"relaying {topic} as MJPEG")

    def _on_image(self, msg):
        if self.frames == 0 and self.errors == 0:
            self.get_logger().info(
                f"first frame: {msg.width}x{msg.height} "
                f"encoding={msg.encoding} bytes={len(msg.data)}"
            )
        try:
            bgr = to_bgr(msg)
        except ValueError as e:
            self.errors += 1
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            self.errors += 1
            return
        with _LOCK:
            _LATEST["jpeg"] = buf.tobytes()
        self.frames += 1

    def _report(self):
        self.get_logger().info(
            f"frames={self.frames} errors={self.errors}"
        )


PAGE = b"""<!doctype html><meta name=viewport content="width=device-width">
<style>body{margin:0;background:#111;display:grid;place-items:center;
height:100vh}img{max-width:100%;max-height:100vh}</style>
<img src="/stream">"""


class Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass                      # keep the console free for ROS output

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return

        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while True:
                with _LOCK:
                    jpeg = _LATEST["jpeg"]
                if jpeg is None:
                    threading.Event().wait(0.1)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                threading.Event().wait(0.05)      # ~20 fps ceiling
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # browser tab closed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", default="/camera/color_image")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    if cv2 is None:
        raise SystemExit(
            "python3-opencv is required for JPEG encoding "
            "(try: python3 -c 'import cv2')"
        )

    rclpy.init()
    node = Relay(args.topic, args.quality)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    host = socket.gethostbyname(socket.gethostname())
    print(f"\n  open  http://{host}:{args.port}/   "
          f"(or http://<car-ip>:{args.port}/)\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
