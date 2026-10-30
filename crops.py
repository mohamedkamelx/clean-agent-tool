"""Evidence crop generation for findings."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from crawler import BoundingBox


def make_crop(
    image_path: str,
    bbox: BoundingBox,
    output_path: str,
    padding_ratio: float = 0.25,
    min_padding_px: int = 16,
    dim_outside: bool = True,
) -> Optional[str]:
    """Save a padded crop around bbox. Returns the output path or None."""
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    pad_x = max(min_padding_px, int(bbox.width * padding_ratio))
    pad_y = max(min_padding_px, int(bbox.height * padding_ratio))

    x0 = max(0, int(bbox.x) - pad_x)
    y0 = max(0, int(bbox.y) - pad_y)
    x1 = min(w, int(bbox.x + bbox.width) + pad_x)
    y1 = min(h, int(bbox.y + bbox.height) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0:y1, x0:x1].copy()

    if dim_outside:
        # Dim everything except the element itself (helps a vision model focus).
        dimmed = (crop * 0.45).astype(np.uint8)
        rx0 = max(0, int(bbox.x) - x0)
        ry0 = max(0, int(bbox.y) - y0)
        rx1 = min(crop.shape[1], int(bbox.x + bbox.width) - x0)
        ry1 = min(crop.shape[0], int(bbox.y + bbox.height) - y0)
        if rx1 > rx0 and ry1 > ry0:
            dimmed[ry0:ry1, rx0:rx1] = crop[ry0:ry1, rx0:rx1]
        crop = dimmed

    # Small border so the crop is visually anchored.
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (60, 60, 60), 1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, crop)
    return output_path
