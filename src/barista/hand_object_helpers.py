"""Shared interaction-level metrics and debug visualization for hand_object evaluation.

Used by both the VLM hand_object task and the CaRe-Ego run_evaluation script
so results can be compared fairly. GT extraction lives here so CaRe-Ego can use
it without importing vlm_benchmarks (which pulls in google-genai).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from barista.bbox import BboxFormat, bbox_iou, normalized_to_pixel, pixel_to_normalized
from barista.dataset import FrameAnnotation


def match_interactions(
    gt_dicts: list[dict],
    pred_dicts: list[dict],
    img_w: int,
    img_h: int,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
    iou_threshold: float = 0.0,
    *,
    pred_boxes_in_pixel: bool = False,
) -> list[tuple[dict, dict]]:
    """Match pred interactions to GT by object-box IoU (greedy, best match first).

    Requires IoU > threshold. Each GT is
    matched to the pred with highest object-box IoU; each pred is matched at most once.

    gt_dicts and pred_dicts: list of {hand_boxes, object_box, hand_type}.
    Boxes are in normalized format (0-scale) unless pred_boxes_in_pixel=True,
    in which case pred boxes are already in pixel coords.
    """
    if not gt_dicts or not pred_dicts:
        return []

    gt_obj_boxes = [
        normalized_to_pixel(d['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale) for d in gt_dicts
    ]
    if pred_boxes_in_pixel:
        pred_obj_boxes = [[float(v) for v in d.get('object_box', [])] for d in pred_dicts]
    else:
        pred_obj_boxes = [
            normalized_to_pixel(d['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale) for d in pred_dicts
        ]

    iou_matrix = [[bbox_iou(pred_obj, gt_obj) for pred_obj in pred_obj_boxes] for gt_obj in gt_obj_boxes]
    matched_pred: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    gt_order = sorted(
        range(len(gt_dicts)),
        key=lambda i: max(iou_matrix[i], default=0.0),
        reverse=True,
    )
    for gt_idx in gt_order:
        row = iou_matrix[gt_idx]
        best_iou, best_pred = -1.0, -1
        for pred_idx, iou_val in enumerate(row):
            if pred_idx not in matched_pred and iou_val > best_iou:
                best_iou, best_pred = iou_val, pred_idx
        if best_pred >= 0 and best_iou > iou_threshold:
            matched_pred.add(best_pred)
            pairs.append((gt_dicts[gt_idx], pred_dicts[best_pred]))
    return pairs


# ---------------------------------------------------------------------------
# Ground-truth extraction (used by CaRe-Ego and VLM hand_object task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionGroundTruth:
    """Ground truth for a single hand-object interaction within a frame."""

    object_id: str
    hand_boxes: list[list[int]]
    object_box: list[int]
    hand_type: str  # 'left' | 'right' | 'both'


def _hand_type_from_category_name(name: str) -> str | None:
    """Return 'left', 'right', or None if the category name is not a hand."""
    if not name:
        return None
    n = name.strip().lower()
    if n == 'left hand':
        return 'left'
    if n == 'right hand':
        return 'right'
    return None


def frame_interactions(
    *,
    frame: FrameAnnotation,
    interaction_relation_type: str,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
) -> list[InteractionGroundTruth]:
    """Extract all hand-object interactions from a single frame.

    Groups relations by target object so that two hands touching the same
    object become a single ``"both"`` interaction. Hand type is inferred
    from the source object's category name (``"left hand"`` / ``"right hand"``).
    """
    objects_by_id = frame.objects_by_id()

    grouped: dict[str, list] = {}
    for rel in frame.relations:
        if rel.relation_type != interaction_relation_type:
            continue
        src = objects_by_id.get(rel.source_object_id)
        tgt = objects_by_id.get(rel.target_object_id)
        if src is None or tgt is None:
            continue
        if _hand_type_from_category_name(src.category.name) is None:
            continue
        grouped.setdefault(str(tgt.object_id), []).append(rel)

    interactions: list[InteractionGroundTruth] = []
    for tgt_uuid, relations in grouped.items():
        tgt_obj = objects_by_id[relations[0].target_object_id]

        hand_boxes: list[list[int]] = []
        hand_sides: set[str] = set()
        seen_hand_uuids: set[str] = set()

        for rel in relations:
            src_obj = objects_by_id[rel.source_object_id]
            hand_uuid = str(src_obj.object_id)
            if hand_uuid in seen_hand_uuids:
                continue
            seen_hand_uuids.add(hand_uuid)

            hand_type_str = _hand_type_from_category_name(src_obj.category.name)
            if hand_type_str:
                hand_sides.add(hand_type_str)

            bbox_xyxy = src_obj.bbox_xyxy()
            x1, y1, x2, y2 = bbox_xyxy
            hand_boxes.append(
                pixel_to_normalized(x1, y1, x2, y2, frame.width, frame.height, fmt=bbox_fmt, scale=bbox_scale)
            )

        if not hand_boxes:
            continue

        tgt_bbox_xyxy = tgt_obj.bbox_xyxy()
        x1, y1, x2, y2 = tgt_bbox_xyxy
        object_box = pixel_to_normalized(x1, y1, x2, y2, frame.width, frame.height, fmt=bbox_fmt, scale=bbox_scale)

        if 'left' in hand_sides and 'right' in hand_sides:
            hand_type = 'both'
        elif 'left' in hand_sides:
            hand_type = 'left'
        else:
            hand_type = 'right'

        interactions.append(
            InteractionGroundTruth(
                object_id=tgt_uuid,
                hand_boxes=hand_boxes,
                object_box=object_box,
                hand_type=hand_type,
            )
        )

    return interactions


def interaction_to_dict(ia: InteractionGroundTruth) -> dict[str, Any]:
    return {
        'object_id': ia.object_id,
        'hand_boxes': ia.hand_boxes,
        'object_box': ia.object_box,
        'hand_type': ia.hand_type,
    }


def hand_iou_for_matched_pair(
    gt_dict: dict,
    pred_dict: dict,
    img_w: int,
    img_h: int,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
    *,
    pred_boxes_in_pixel: bool = False,
) -> float:
    """Per-hand IoU for a matched interaction pair. Greedy match pred hands to GT hands."""
    gt_hands = gt_dict.get('hand_boxes', [])
    pred_hands = pred_dict.get('hand_boxes', [])
    if not gt_hands:
        return 0.0

    gt_hands_px = [normalized_to_pixel(b, img_w, img_h, fmt=bbox_fmt, scale=bbox_scale) for b in gt_hands]
    if pred_boxes_in_pixel:
        pred_hands_px = [[float(v) for v in b] for b in pred_hands]
    else:
        pred_hands_px = [normalized_to_pixel(b, img_w, img_h, fmt=bbox_fmt, scale=bbox_scale) for b in pred_hands]

    iou_matrix = [[bbox_iou(ph, gh) for ph in pred_hands_px] for gh in gt_hands_px]
    matched_pred: set[int] = set()
    ious_for_gt: list[float] = []
    for gt_idx in range(len(gt_hands_px)):
        row = iou_matrix[gt_idx]
        best_iou, best_pred = -1.0, -1
        for pred_idx, iou_val in enumerate(row):
            if pred_idx not in matched_pred and iou_val > best_iou:
                best_iou, best_pred = iou_val, pred_idx
        ious_for_gt.append(best_iou if best_pred >= 0 else 0.0)
        if best_pred >= 0:
            matched_pred.add(best_pred)
    return sum(ious_for_gt) / len(ious_for_gt)


def compute_interaction_metrics_for_frame(
    gt_dicts: list[dict],
    pred_dicts: list[dict],
    img_w: int,
    img_h: int,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
    iou_threshold: float = 0.0,
    *,
    pred_boxes_in_pixel: bool = False,
) -> dict:
    """Compute interaction-level metrics for a single frame.

    Returns dict with: total_gt, total_pred, matched_gt, matched_pred, hand_ious,
    object_ious, hand_type_correct, has_gt, has_gt_but_zero_pred.
    """
    n_gt = len(gt_dicts)
    n_pred = len(pred_dicts)
    has_gt = n_gt > 0
    has_gt_but_zero_pred = n_gt > 0 and n_pred == 0

    matched_pairs = match_interactions(
        gt_dicts,
        pred_dicts,
        img_w,
        img_h,
        bbox_fmt,
        bbox_scale,
        iou_threshold=iou_threshold,
        pred_boxes_in_pixel=pred_boxes_in_pixel,
    )

    hand_ious: list[float] = []
    object_ious: list[float] = []
    hand_type_correct = 0

    for gt_d, pred_d in matched_pairs:
        hand_ious.append(
            hand_iou_for_matched_pair(
                gt_d,
                pred_d,
                img_w,
                img_h,
                bbox_fmt,
                bbox_scale,
                pred_boxes_in_pixel=pred_boxes_in_pixel,
            )
        )
        gt_obj = normalized_to_pixel(gt_d['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
        if pred_boxes_in_pixel:
            pred_obj = [float(v) for v in pred_d['object_box']]
        else:
            pred_obj = normalized_to_pixel(pred_d['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
        object_ious.append(bbox_iou(pred_obj, gt_obj))
        if pred_d.get('hand_type') == gt_d.get('hand_type'):
            hand_type_correct += 1

    return {
        'total_gt': n_gt,
        'total_pred': n_pred,
        'matched_gt': len(matched_pairs),
        'matched_pred': len(matched_pairs),
        'hand_ious': hand_ious,
        'object_ious': object_ious,
        'hand_type_correct': hand_type_correct,
        'has_gt': has_gt,
        'has_gt_but_zero_pred': has_gt_but_zero_pred,
    }


# ---------------------------------------------------------------------------
# Debug visualization (GT vs pred overlay)
# ---------------------------------------------------------------------------


def _debug_font(size: int = 20) -> ImageFont.ImageFont:
    """Load a readable font for debug labels; fall back to default if unavailable."""
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_debug_frame(
    image: Image.Image,
    out_path: Path,
    gt_dicts: list[dict],
    pred_dicts: list[dict],
    img_w: int,
    img_h: int,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
    *,
    pred_boxes_in_pixel: bool = False,
) -> None:
    """Draw GT (green) and pred (red) bboxes on a frame and save for debugging.

    gt_dicts and pred_dicts: list of {hand_boxes, object_box, hand_type}.
    GT boxes are always in normalized format (0-scale). Pred boxes are normalized
    unless pred_boxes_in_pixel=True (e.g. CaRe-Ego outputs pixel coords).
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = _debug_font()

    def _draw_box(box: list[float], color: str, label: str) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(0, y1 - 22)), label, fill=color, font=font)

    # GT boxes (green) — always normalized
    for d in gt_dicts:
        for box in d.get('hand_boxes', []):
            px = normalized_to_pixel(box, img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
            _draw_box(px, '#00ff00', 'GT hand')
        if d.get('object_box'):
            px = normalized_to_pixel(d['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
            _draw_box(px, '#00ff00', 'GT obj')

    # Pred boxes (red) — normalized or pixel depending on pred_boxes_in_pixel
    for ia in pred_dicts:
        for box in ia.get('hand_boxes', []):
            if pred_boxes_in_pixel:
                px = [float(v) for v in box]
            else:
                px = normalized_to_pixel(box, img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
            _draw_box(px, '#ff0000', 'pred hand')
        if ia.get('object_box'):
            if pred_boxes_in_pixel:
                px = [float(v) for v in ia['object_box']]
            else:
                px = normalized_to_pixel(ia['object_box'], img_w, img_h, fmt=bbox_fmt, scale=bbox_scale)
            _draw_box(px, '#ff0000', 'pred obj')

    img.save(out_path)
