"""Geometry shared by adapters; crop source is always the original RGB buffer."""
from dataclasses import dataclass
import math

import numpy as np

from mes_vision.inputs import Frame
from .contracts import Box, Crop, finite


@dataclass(frozen=True, slots=True)
class ResizeMap:
    """Inverse of original -> resized -> padded coordinates, using actual scales.

    RF-DETR's public predict already returns original coordinates and does NOT use
    this map. Other adapters must explicitly undo their own preprocessing once.
    """
    scale_x: float
    scale_y: float
    pad_left: float = 0
    pad_top: float = 0

    def __post_init__(self):
        for value in (self.scale_x, self.scale_y, self.pad_left, self.pad_top):
            finite(value, "resize map")
        if self.scale_x <= 0 or self.scale_y <= 0 or self.pad_left < 0 or self.pad_top < 0:
            raise ValueError("invalid scale or padding")

    def to_original(self, box: Box) -> Box:
        return Box((box.x1 - self.pad_left) / self.scale_x, (box.y1 - self.pad_top) / self.scale_y,
                   (box.x2 - self.pad_left) / self.scale_x, (box.y2 - self.pad_top) / self.scale_y)


def extract_crop(frame: Frame, object_id: str, box: Box, margin_px: int = 0) -> Crop:
    if type(margin_px) is not int or margin_px < 0:
        raise ValueError("margin_px must be a nonnegative integer")
    clipped = box.clip(frame.width, frame.height)
    if clipped is None:
        raise ValueError("object does not intersect the image")
    left = max(0, math.floor(clipped.x1) - margin_px)
    top = max(0, math.floor(clipped.y1) - margin_px)
    right = min(frame.width, math.ceil(clipped.x2) + margin_px)
    bottom = min(frame.height, math.ceil(clipped.y2) + margin_px)
    rgb = np.array(frame.rgb[top:bottom, left:right], copy=True, order="C")
    rgb.setflags(write=False)
    return Crop(frame.frame_id, object_id, Box(left, top, right, bottom), clipped, rgb)
