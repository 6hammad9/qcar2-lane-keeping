"""Debug visualization for lane detection."""

import cv2
import numpy as np
from typing import Optional, Tuple

Line = Tuple[float, float]


def draw_debug_frame(
    frame: np.ndarray,
    strip: np.ndarray,
    binary: np.ndarray,
    cl_line: Optional[Line],
    rl_line: Optional[Line],
    cl_top_x: Optional[float],
    cl_mid_x: Optional[float],
    cl_bot_x: Optional[float],
    rl_top_x: Optional[float],
    rl_mid_x: Optional[float],
    rl_bot_x: Optional[float],
    lane_centre_x: Optional[float],
    error: float,
    strip_y: int,
    sh: int,
    img_width: int,
    split_fraction: float
):
    """Draw both debug windows."""
    strip_vis = strip.copy()

    # Draw lines and points
    if cl_line is not None:
        _draw_line(strip_vis, cl_line, sh, (255, 255, 0))
        _draw_point(strip_vis, cl_top_x, 5, 'CL_T', (255, 255, 0))
        _draw_point(strip_vis, cl_mid_x, sh // 2, 'CL_M', (255, 255, 0))
        _draw_point(strip_vis, cl_bot_x, sh - 5, 'CL_B', (255, 255, 0))

    if rl_line is not None:
        _draw_line(strip_vis, rl_line, sh, (0, 255, 255))
        _draw_point(strip_vis, rl_top_x, 5, 'RL_T', (0, 255, 255))
        _draw_point(strip_vis, rl_mid_x, sh // 2, 'RL_M', (0, 255, 255))
        _draw_point(strip_vis, rl_bot_x, sh - 5, 'RL_B', (0, 255, 255))

    # Lane centre (green) and image centre (red)
    if lane_centre_x is not None:
        cv2.line(strip_vis, (int(lane_centre_x), 0),
                 (int(lane_centre_x), sh), (0, 255, 0), 2)
    cv2.line(strip_vis, (img_width // 2, 0),
             (img_width // 2, sh), (0, 0, 255), 2)

    # Compose main display
    vis = frame.copy()
    vis[strip_y:, :] = strip_vis
    cv2.line(vis, (0, strip_y), (img_width, strip_y), (128, 128, 0), 1)

    # HUD
    _draw_hud(vis, cl_top_x, cl_mid_x, cl_bot_x,
              rl_top_x, rl_mid_x, rl_bot_x,
              error, sh, cl_line, rl_line)

    # Second window
    bottom_row = _create_strip_mask_view(strip_vis, binary, img_width, split_fraction)

    cv2.imshow('Lane Node', vis)
    cv2.imshow('Strip | Mask', bottom_row)
    cv2.waitKey(1)


def _draw_line(img: np.ndarray, line, sh: int, color: Tuple[int, int, int]):
    if line is None:
        return

    if isinstance(line, np.ndarray) and len(line) == 3:
        # Polynomial: x = Ay² + By + C
        pts = []
        for y in range(0, sh, 5):
            x = int(line[0] * y**2 + line[1] * y + line[2])
            if 0 <= x < img.shape[1]:
                pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], color, 2)
    else:
        # Hough: x = slope * y + intercept
        slope, intercept = line
        x_top = int(slope * 0 + intercept)
        x_bot = int(slope * sh + intercept)
        cv2.line(img, (x_top, 0), (x_bot, sh), color, 2)

def _draw_point(img: np.ndarray, x: Optional[float], y: int, label: str, color: Tuple[int, int, int]):
    if x is None:
        return
    ix, iy = int(x), int(y)
    cv2.circle(img, (ix, iy), 8, color, -1)
    cv2.putText(img, f'{label} ({ix},{iy})',
                (ix + 10, iy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def _draw_hud(vis: np.ndarray,
              cl_top_x, cl_mid_x, cl_bot_x,
              rl_top_x, rl_mid_x, rl_bot_x,
              error: float, sh: int,
              cl_line, rl_line):
    def fmt(x, y):
        return f'({int(x)},{int(y)})' if x is not None else '(--,--)'

    hud_lines = [
        (f'CL_T {fmt(cl_top_x, 0)}', (255, 255, 0)),
        (f'CL_M {fmt(cl_mid_x, sh//2)}', (255, 255, 0)),
        (f'CL_B {fmt(cl_bot_x, sh-5)}', (255, 255, 0)),
        (f'RL_T {fmt(rl_top_x, 0)}', (0, 255, 255)),
        (f'RL_M {fmt(rl_mid_x, sh//2)}', (0, 255, 255)),
        (f'RL_B {fmt(rl_bot_x, sh-5)}', (0, 255, 255)),
        (f'Error: {error:+.3f}', (255, 255, 255)),
    ]
    for i, (text, col) in enumerate(hud_lines):
        cv2.putText(vis, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    both = cl_line is not None and rl_line is not None
    one = (cl_line is not None) != (rl_line is not None)
    status = 'BOTH LINES' if both else ('ONE LINE' if one else 'NO LINES')
    s_col = (0, 255, 0) if both else ((0, 165, 255) if one else (0, 0, 255))
    cv2.putText(vis, status, (10, 25 + len(hud_lines) * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)


def _create_strip_mask_view(strip_vis: np.ndarray, binary: np.ndarray,
                             img_width: int, split_fraction: float) -> np.ndarray:
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    split_px = int(binary.shape[1] * split_fraction)
    cv2.line(binary_color, (split_px, 0),
             (split_px, binary.shape[0]), (0, 100, 255), 1)

    strip_resized = cv2.resize(strip_vis, (img_width // 2, 160))
    binary_resized = cv2.resize(binary_color, (img_width // 2, 160))
    return np.hstack([strip_resized, binary_resized])
