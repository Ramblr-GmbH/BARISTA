"""Bounding-box format conventions and conversion utilities.

Two coordinate orders are supported:

- **xyxy**: ``[xmin, ymin, xmax, ymax]`` — the standard order used by
  COCO/torchvision.
- **yxyx**: ``[ymin, xmin, ymax, xmax]`` — used by Gemini's ``box_2d``
  output.

Coordinates are integers on a normalised ``[0, scale]`` grid (default 1000).
Use :func:`pixel_to_normalized` / :func:`normalized_to_pixel` to move between
pixel space and the normalised grid, and :func:`convert` to switch between
``xyxy`` and ``yxyx`` at the same scale.
"""

from __future__ import annotations

from enum import Enum

# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class BboxFormat(str, Enum):
    """Supported bounding-box coordinate orders."""

    XYXY = 'xyxy'
    """``[xmin, ymin, xmax, ymax]``"""

    YXYX = 'yxyx'
    """``[ymin, xmin, ymax, xmax]`` (Gemini ``box_2d``)"""


def pixel_to_normalized(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    *,
    fmt: BboxFormat,
    scale: int = 1000,
) -> list[int]:
    """Convert pixel ``[x1, y1, x2, y2]`` to a normalised box.

    Returns integer coordinates clamped to ``[0, scale]`` in the requested
    *fmt* order.
    """
    xmin = _clamp(round(x1 / width * scale), 0, scale)
    ymin = _clamp(round(y1 / height * scale), 0, scale)
    xmax = _clamp(round(x2 / width * scale), 0, scale)
    ymax = _clamp(round(y2 / height * scale), 0, scale)
    if fmt is BboxFormat.XYXY:
        return [xmin, ymin, xmax, ymax]
    return [ymin, xmin, ymax, xmax]


def normalized_to_pixel(
    box: list[int | float],
    width: int,
    height: int,
    *,
    fmt: BboxFormat,
    scale: int = 1000,
) -> list[float]:
    """Convert a normalised box back to pixel ``[x1, y1, x2, y2]``."""
    if fmt is BboxFormat.XYXY:
        xmin, ymin, xmax, ymax = box
    else:
        ymin, xmin, ymax, xmax = box
    x1 = float(xmin) / scale * width
    y1 = float(ymin) / scale * height
    x2 = float(xmax) / scale * width
    y2 = float(ymax) / scale * height
    return [x1, y1, x2, y2]


def convert(
    box: list[int | float],
    *,
    src: BboxFormat,
    dst: BboxFormat,
) -> list[int | float]:
    """Swap between coordinate orders at the same scale.

    If *src* and *dst* are identical the box is returned as-is (copied).
    """
    if src is dst:
        return list(box)
    # xyxy <-> yxyx: swap the x/y pairs
    a, b, c, d = box
    return [b, a, d, c]


def format_description(fmt: BboxFormat) -> str:
    """Human-readable coordinate description for prompts."""
    if fmt is BboxFormat.XYXY:
        return '[xmin, ymin, xmax, ymax]'
    return '[ymin, xmin, ymax, xmax]'


def bbox_iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union for two xyxy boxes [x1, y1, x2, y2]."""
    inter_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_x * inter_y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    """Intersection-over-union for two xywh boxes [x, y, width, height]."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_y = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_x * inter_y
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_iou_matrix(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> list[list[float]]:
    """Pairwise IoU matrix between two lists of xyxy boxes.

    Returns a ``len(pred_boxes) x len(gt_boxes)`` matrix where
    ``result[i][j]`` is the IoU between ``pred_boxes[i]`` and ``gt_boxes[j]``.
    """
    return [[bbox_iou(pb, gb) for gb in gt_boxes] for pb in pred_boxes]
