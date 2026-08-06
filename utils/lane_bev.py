#!/usr/bin/env python3
"""Bird's-eye lane centering: warp, sliding window, quadratic fit, fill.

Replaces the straight-line Hough fitting in lane_centering_node, which fits
lines in the raw camera frame. That fails here for a structural reason: in
perspective a lane converges with distance and a bend looks like a kink, so
a straight-segment fit has no way to represent the road. Captured on this
track it fitted a wall/floor junction, a ROSbot, and a person's sandals, all
while reporting valid=True at ~85% of full offset authority.

In bird's-eye the lane has constant width and a bend is a true arc, so:

  - a quadratic x = A*y^2 + B*y + C fits each boundary properly;
  - the warp trapezoid geometrically excludes walls and radiators, which is
    most of what the old thresholding was picking up;
  - lane width and parallelism become checkable, so `valid` can mean
    something;
  - the offset comes out in METRES, because the destination rectangle fixes
    the pixel-to-metre scale.

The perspective points are taken from qcar_lane_pkg/lane_keeping_node.py,
where they were already calibrated for this camera and track. They are
normalised fractions of image size, so they survive a resolution change.

    python3 lane_bev.py                      # publishes /lane_bev_debug
    python3 lane_bev.py --save /tmp/bev      # also writes frames to disk
"""

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


# Calibrated in qcar_lane_pkg/lane_keeping_node.py. Fractions of image size.
SRC = np.float32([[0.3719, 0.6319], [0.6344, 0.6319],
                  [0.1953, 1.0000], [0.9688, 1.0000]])
DST = np.float32([[0.206, 0.000], [0.766, 0.000],
                  [0.206, 1.000], [0.766, 1.000]])

BEV = 800                       # bird's-eye output is BEV x BEV
LANE_WIDTH_M = 0.42             # from config/lane_params.yaml

# DST puts the two lane lines at these columns, so the scale follows.
LANE_WIDTH_PX = (DST[1][0] - DST[0][0]) * BEV
XM_PER_PIX = LANE_WIDTH_M / LANE_WIDTH_PX


def warp(img, reverse=False, out_shape=None):
    """Camera <-> bird's-eye.

    On the reverse pass ``img`` is the BEV overlay, so SRC must still be
    scaled by the CAMERA frame size and the output sized to it -- not to the
    overlay's own dimensions.
    """
    dst = DST * np.float32([BEV, BEV])
    if reverse:
        h, w = out_shape
        src = SRC * np.float32([w, h])
        m = cv2.getPerspectiveTransform(dst, src)
        return cv2.warpPerspective(img, m, (w, h))
    src = SRC * np.float32([img.shape[1], img.shape[0]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (BEV, BEV))


def threshold(bev):
    """Lane paint is brighter than the floor it is painted on.

    Otsu adapts to the ambient level, which matters because this lab's
    lighting varies a lot across the loop. The fixed HSV band the old node
    used ([0,0,170]-[180,45,255]) meant 'anything pale' and caught the walls;
    here the warp has already removed the walls, so a brightness split on the
    road surface alone is enough.
    """
    grey = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )


def sliding_window_fit(binary, nwindows=12, margin=70, minpix=40):
    """Locate both boundaries and fit a quadratic to each.

    Returns (left_fit, right_fit, left_px, right_px); a fit is None when its
    boundary was not found.
    """
    hist = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    mid = hist.shape[0] // 2
    if hist[:mid].max() < 255 * 3:
        left_base = None
    else:
        left_base = int(np.argmax(hist[:mid]))
    if hist[mid:].max() < 255 * 3:
        right_base = None
    else:
        right_base = int(np.argmax(hist[mid:]) + mid)

    nz = binary.nonzero()
    nzy, nzx = np.array(nz[0]), np.array(nz[1])
    win_h = binary.shape[0] // nwindows

    def track(base):
        if base is None:
            return None
        current = base
        keep = []
        for w in range(nwindows):
            ylo = binary.shape[0] - (w + 1) * win_h
            yhi = binary.shape[0] - w * win_h
            xlo, xhi = current - margin, current + margin
            hit = ((nzy >= ylo) & (nzy < yhi)
                   & (nzx >= xlo) & (nzx < xhi)).nonzero()[0]
            keep.append(hit)
            if len(hit) > minpix:
                current = int(np.mean(nzx[hit]))
        keep = np.concatenate(keep) if keep else np.array([], dtype=int)
        return keep if len(keep) > 150 else None

    li, ri = track(left_base), track(right_base)

    def fit(idx):
        if idx is None:
            return None, None
        y, x = nzy[idx], nzx[idx]
        if len(np.unique(y)) < 3:
            return None, None
        return np.polyfit(y, x, 2), (x, y)

    lf, lp = fit(li)
    rf, rp = fit(ri)
    return lf, rf, lp, rp


def lane_offset(left_fit, right_fit, y_eval):
    """Signed lateral offset in metres. Positive means the car is LEFT of
    the lane centre, matching the y-left-positive convention used elsewhere.

    With only one boundary, the other is inferred from the known lane width,
    which is exactly the constraint the perspective view could not provide.
    """
    half = LANE_WIDTH_PX / 2.0
    if left_fit is not None and right_fit is not None:
        lx = np.polyval(left_fit, y_eval)
        rx = np.polyval(right_fit, y_eval)
        centre = 0.5 * (lx + rx)
        width_px = abs(rx - lx)
        # Reject a fit pair that cannot be a lane.
        if not 0.55 * LANE_WIDTH_PX < width_px < 1.6 * LANE_WIDTH_PX:
            return None, "BAD_WIDTH", None, None
        return (BEV / 2.0 - centre) * XM_PER_PIX, "BOTH", lx, rx
    if left_fit is not None:
        lx = np.polyval(left_fit, y_eval)
        return (BEV / 2.0 - (lx + half)) * XM_PER_PIX, "LEFT", lx, lx + 2 * half
    if right_fit is not None:
        rx = np.polyval(right_fit, y_eval)
        return (BEV / 2.0 - (rx - half)) * XM_PER_PIX, "RIGHT", rx - 2 * half, rx
    return None, "NONE", None, None


def draw(frame, bev, left_fit, right_fit, offset, mode):
    """Filled lane region, inverse-warped onto the camera image."""
    overlay = np.zeros_like(bev)
    ys = np.linspace(0, BEV - 1, BEV).astype(int)

    if left_fit is not None or right_fit is not None:
        half = LANE_WIDTH_PX / 2.0
        if left_fit is not None and right_fit is not None:
            lx, rx = np.polyval(left_fit, ys), np.polyval(right_fit, ys)
        elif left_fit is not None:
            lx = np.polyval(left_fit, ys)
            rx = lx + 2 * half
        else:
            rx = np.polyval(right_fit, ys)
            lx = rx - 2 * half

        pts = np.hstack([
            np.array([np.transpose(np.vstack([lx, ys]))]),
            np.array([np.flipud(np.transpose(np.vstack([rx, ys])))]),
        ])
        cv2.fillPoly(overlay, np.int32([pts]), (255, 160, 40))
        cx = 0.5 * (lx + rx)
        for i in range(0, BEV - 12, 24):
            cv2.line(overlay, (int(cx[i]), ys[i]),
                     (int(cx[i + 10]), ys[i + 10]), (255, 255, 255), 3)

    back = warp(overlay, reverse=True, out_shape=frame.shape[:2])
    out = cv2.addWeighted(frame, 1.0, back, 0.45, 0)

    txt = "no lane" if offset is None else f"offset = {offset:+.4f} m"
    cv2.putText(out, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5)
    cv2.putText(out, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (60, 255, 255), 2)
    cv2.putText(out, f"mode = {mode}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 5)
    cv2.putText(out, f"mode = {mode}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (120, 255, 120), 2)
    return out


class LaneBEV(Node):

    def __init__(self, save=None):
        super().__init__("lane_bev")
        self.bridge = CvBridge()
        self.save = save
        self.n = 0
        self.offset_pub = self.create_publisher(Float32, "/lane_center_offset", 10)
        self.valid_pub = self.create_publisher(Bool, "/lane_center_valid", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_bev_debug", 10)
        self.bev_pub = self.create_publisher(Image, "/lane_bev_raw", 10)
        self.create_subscription(
            Image, "/camera/color_image", self.cb, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"lane_bev up: {XM_PER_PIX*1000:.3f} mm/px, "
            f"lane={LANE_WIDTH_PX:.0f} px = {LANE_WIDTH_M} m"
        )

    def cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        if frame is None or frame.size == 0:
            return

        bev = warp(frame)
        binary = threshold(bev)
        lf, rf, _lp, _rp = sliding_window_fit(binary)
        # Evaluate near the bottom of the BEV, i.e. just ahead of the car.
        offset, mode, _lx, _rx = lane_offset(lf, rf, BEV * 0.85)

        valid = offset is not None and abs(offset) < 0.5
        self.offset_pub.publish(Float32(data=float(offset or 0.0)))
        self.valid_pub.publish(Bool(data=bool(valid)))

        vis = draw(frame, bev, lf, rf, offset, mode)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
            bev_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.bev_pub.publish(self.bridge.cv2_to_imgmsg(bev_vis, "bgr8"))
        except Exception:
            pass

        self.n += 1
        if self.save and self.n % 12 == 0:
            i = self.n // 12
            cv2.imwrite(f"{self.save}_vis_{i}.png", vis)
            cv2.imwrite(f"{self.save}_bev_{i}.png", bev_vis)
        if self.n % 20 == 0:
            self.get_logger().info(
                f"mode={mode} offset="
                + ("none" if offset is None else f"{offset:+.4f} m")
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = LaneBEV(save=args.save)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
