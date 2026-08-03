"""White line segmentation."""

import cv2
import numpy as np
from typing import Tuple


def segment_white(
    image_bgr: np.ndarray,
    hsv_low: Tuple[int, int, int],
    hsv_high: Tuple[int, int, int],
    morph_iterations: int = 2
) -> np.ndarray:
    """
    Segment white pixels from BGR image.
    
    Returns binary mask (255 = white line, 0 = background).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    low = np.array(hsv_low, dtype=np.uint8)
    high = np.array(hsv_high, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    
    # Morphological close with tall kernel (bridges dash gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    return mask
