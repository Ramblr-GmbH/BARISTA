from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from PIL import ImageDraw, ImageFont
from torchmetrics.detection import MeanAveragePrecision

from barista.bbox import BboxFormat, bbox_iou_matrix, format_description, normalized_to_pixel, pixel_to_normalized
from barista.dataset import FrameAnnotation, ObjectAnnotation, load_videos
from barista.vlm_benchmarks.tasks.base import BenchmarkTask
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    FrameInput,
    MessagePart,
    PredictionResult,
    TextPart,
)

# Tailored mainly towards Gemini format.
GROUNDING_SYSTEM_PROMPT_TEMPLATE = (
    'You are an expert at analyzing video frames to locate specific objects.\n'
    '\n'
    'For each object matching the description, output its bounding box.\n'
    '\n'
    'Bounding box format: {coord_desc}, integers 0-{scale}. Order: {coord_order}\n'
    '\n'
    'The final output should be a JSON object:\n'
    '{{"bboxes": [[a, b, c, d], [a, b, c, d]]}}\n'
    '\n'
    'Each inner list is exactly 4 integers. If no matching objects, return: {{"bboxes": []}}'
)

_MARKDOWN_FENCE_PATTERN = re.compile(r'```(?:\w*)\n(.*?)```', re.DOTALL)
# Fallback: extract any 4 comma-separated integers from the text.
_FOUR_INT_COORDS = re.compile(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return 'N/A'
    return f'{float(value):.4f}'


def _fmt_latency(value: float | None) -> str:
    if value is None:
        return 'N/A'
    ms = float(value)
    if ms >= 1000.0:
        return f'{ms / 1000.0:.2f}s'
    return f'{ms:.0f}ms'


def _has_valid_bbox(obj: ObjectAnnotation, min_bbox_area: float | None) -> bool:
    """Check whether an object has a valid bbox passing the area filter."""
    if min_bbox_area is not None and (obj.bbox_area() or 0) < float(min_bbox_area):
        return False
    return True


def _parse_grounding_response(raw_text: str) -> dict[str, Any] | None:
    """Parse model JSON response into a list of predicted bounding boxes."""
    text = raw_text.strip()
    fence_match = _MARKDOWN_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            key = 'bboxes' if 'bboxes' in parsed else 'boxes'
            if key in parsed and isinstance(parsed[key], list):
                return {'predicted_boxes': _coerce_boxes(parsed[key])}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: pull every group of 4 integers from the text.
    matches = _FOUR_INT_COORDS.findall(text)
    if matches:
        return {'predicted_boxes': [[int(a), int(b), int(c), int(d)] for a, b, c, d in matches]}

    return None


def _coerce_boxes(raw: list) -> list[list[int]]:
    """Convert whatever the model returned into a flat list of [a, b, c, d] boxes."""
    boxes: list[list[int]] = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) >= 4 and all(isinstance(v, (int, float)) for v in entry[:4]):
            boxes.append([int(round(float(v))) for v in entry[:4]])
    return boxes


def _to_torchmetrics_entry(
    box_list: list[list[int]],
    category_id: int,
    frame_width: int,
    frame_height: int,
    *,
    bbox_fmt: BboxFormat,
    bbox_scale: int,
    with_scores: bool,
) -> dict[str, list]:
    """Convert a list of boxes to a torchmetrics-compatible entry.

    All boxes share the same *category_id*.  When *with_scores* is True
    (predictions), a ``scores`` key with all-1.0 confidence values is included.
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    for box in box_list:
        boxes.append(normalized_to_pixel(box, frame_width, frame_height, fmt=bbox_fmt, scale=bbox_scale))
        labels.append(category_id)
    entry: dict[str, list] = {'boxes': boxes, 'labels': labels}
    if with_scores:
        entry['scores'] = [1.0] * len(boxes)
    return entry


def save_debug_frame(
    example: BenchmarkExample,
    prediction: PredictionResult,
    out_path: Path,
) -> None:
    """Draw GT (green) and pred (red) bounding boxes on a frame and save."""
    task_result = prediction.task_result or {}
    gt_boxes = cast(list[list[int]], example.task_data['ground_truth_boxes'])
    pred_boxes = cast(list[list[int]], task_result.get('predicted_boxes', []))
    frame_width = cast(int, example.task_data['frame_width'])
    frame_height = cast(int, example.task_data['frame_height'])
    gt_bbox_fmt = BboxFormat(str(example.task_data.get('bbox_format', 'xyxy')))
    pred_bbox_fmt = BboxFormat(str(task_result.get('_pred_bbox_format', gt_bbox_fmt.value)))
    bbox_scale = int(example.task_data.get('bbox_scale', 1000))
    phrase = str(example.task_data.get('phrase', ''))

    img = example.frames[0].load_pil_image()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()

    def _draw_box(box_2d: list[int], color: str, label: str, fmt: BboxFormat) -> None:
        px = normalized_to_pixel(box_2d, frame_width, frame_height, fmt=fmt, scale=bbox_scale)
        x1, y1, x2, y2 = [int(round(v)) for v in px]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 20)), label, fill=color, font=font)

    for box in gt_boxes:
        _draw_box(box, '#00ff00', f'GT: {phrase}', gt_bbox_fmt)

    for box in pred_boxes:
        _draw_box(box, '#ff0000', f'P: {phrase}', pred_bbox_fmt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Referring-description helpers
# ---------------------------------------------------------------------------

# Maps relation value -> (source_template, target_template) with {t} placeholder.
_RELATION_TEMPLATES: dict[str, tuple[str, str | None]] = {
    'under': ('under {t}', None),
    'on': ('on {t}', None),
    'inside': ('inside {t}', 'containing {t}'),
    'contains': ('containing {t}', 'inside {t}'),
    'attached to': ('attached to {t}', None),
    'part of': ('part of {t}', None),
    'pulls': ('pulling {t}', 'pulled by {t}'),
    'pushes': ('pushing {t}', 'pushed by {t}'),
    'presses': ('pressing {t}', 'pressed by {t}'),
    'holds': ('holding {t}', 'held by {t}'),
    'touches': ('touching {t}', 'touched by {t}'),
    'turns': ('turning {t}', 'turned by {t}'),
}


class _RelKey(NamedTuple):
    value: str
    target_category: str
    is_source: bool


class _PhraseComponents(NamedTuple):
    colors: frozenset[str]
    relation: _RelKey | None


def _phrase_score(comp: _PhraseComponents) -> tuple[int, int]:
    """Richness score for phrase selection: (has_relation, num_colors)."""
    return (1 if comp.relation is not None else 0, len(comp.colors))


def _parse_frame_objects(
    frame: FrameAnnotation,
    objects_by_id: dict[Any, ObjectAnnotation],
) -> tuple[dict[Any, frozenset[str]], dict[Any, list[_RelKey]]]:
    """Pre-parse colors and relations for every object in *frame*.

    Returns ``(colors_by_id, relations_by_id)`` where:

    - ``colors_by_id[obj_id]`` is a frozenset of colour attribute values
      (filtered: no ``'skin'``, no colours already in the category name).
    - ``relations_by_id[obj_id]`` is a list of ``(normalized_value,
      other_category, is_source)`` tuples for every recognised relation the
      object participates in.
    """
    colors_by_id: dict[Any, frozenset[str]] = {}
    relations_by_id: dict[Any, list[_RelKey]] = {}

    for obj in frame.objects:
        cat_lower = obj.category.name.lower()
        colors_by_id[obj.object_id] = frozenset(
            attr.value
            for attr in obj.attributes
            if attr.attribute_type == 'color' and attr.value != 'skin' and attr.value.lower() not in cat_lower
        )
        relations_by_id[obj.object_id] = []

    for rel in frame.relations:
        normalized = ' '.join(rel.value.lower().split())
        if normalized not in _RELATION_TEMPLATES:
            continue
        src = objects_by_id.get(rel.source_object_id)
        tgt = objects_by_id.get(rel.target_object_id)
        _, tgt_tmpl = _RELATION_TEMPLATES[normalized]
        if src is not None and tgt is not None and tgt.category.name != 'unknown':
            relations_by_id.setdefault(rel.source_object_id, []).append(_RelKey(normalized, tgt.category.name, True))
        if tgt_tmpl is not None and tgt is not None and src is not None and src.category.name != 'unknown':
            relations_by_id.setdefault(rel.target_object_id, []).append(_RelKey(normalized, src.category.name, False))

    return colors_by_id, relations_by_id


def _build_referring_description(
    obj: ObjectAnnotation,
    colors_by_id: dict[Any, frozenset[str]],
    relations_by_id: dict[Any, list[_RelKey]],
    rng: random.Random,
) -> tuple[str, _PhraseComponents]:
    """Build a short grounding description like 'red cup on table'.

    Returns ``(phrase, components)`` so callers can check whether another
    object matches the same description components.
    """
    category_name = obj.category.name
    colors = colors_by_id.get(obj.object_id, frozenset())
    base = f'{" ".join(sorted(colors))} {category_name}' if colors else category_name

    valid_rels = [r for r in relations_by_id.get(obj.object_id, []) if r.value in _RELATION_TEMPLATES]
    best = rng.choice(valid_rels) if valid_rels else None

    if best:
        src_tmpl, tgt_tmpl = _RELATION_TEMPLATES[best.value]
        phrase = (src_tmpl if best.is_source else tgt_tmpl).format(t=best.target_category)
        return f'{base} {phrase}', _PhraseComponents(colors, best)

    return base, _PhraseComponents(colors, None)


class GroundingTask(BenchmarkTask):
    """Referring-based object grounding.

    For each frame, objects with valid bboxes are grouped by **category**.
    One phrase is chosen per category – the richest description (most
    attributes + relation info) among the category's objects.  Ground
    truth boxes are only the objects that actually possess the attributes
    and relations described by the winning phrase.

    Evaluation uses torchmetrics ``MeanAveragePrecision`` (COCO-style mAP).

    Supported ``task_params`` keys:

    - ``bbox_format`` (str, default ``'yxyx'``): coordinate order — ``'xyxy'``
      for ``[xmin, ymin, xmax, ymax]`` or ``'yxyx'`` for ``[ymin, xmin, ymax, xmax]``
      (Gemini ``box_2d``).
    - ``bbox_scale`` (int, default 1000): normalisation scale for box
      coordinates (e.g. 1000 → integers in ``[0, 1000]``).
    - ``min_bbox_area`` (float | null, default null): minimum bbox pixel area;
      objects below this threshold are excluded from ground truth.
    - ``video_id`` (str | null): optional filter to a single video.
    - ``sample_fraction`` (float | null, default null): if set, evaluate only
      this fraction of frames per video (0.0–1.0], evenly spaced.
    - ``seed`` (int | null, default null): seed for the relation-selection RNG;
      set for reproducible phrase generation.
    - ``require_relations`` (bool, default false): if true, skip frames where no
      object participates in a recognized relation; focuses the task on frames
      with richer relational structure.
    """

    def __init__(self, dataset_root: Path, task_params: dict[str, Any]) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)
        self.bbox_format = BboxFormat(self.task_params.get('bbox_format', 'yxyx'))
        self.bbox_scale = int(self.task_params.get('bbox_scale', 1000))
        self._rng = random.Random(self.task_params.get('seed'))

    def iter_examples(self) -> list[BenchmarkExample]:
        min_bbox_area = self.task_params.get('min_bbox_area')
        sample_fraction = self.task_params.get('sample_fraction')
        activity_only = bool(self.task_params.get('activity_filter'))
        require_relations = bool(self.task_params.get('require_relations', False))
        video_id_filter = self.task_params.get('video_id')

        videos = load_videos(self.dataset_root)
        if video_id_filter is not None:
            videos = [v for v in videos if v.video_id == str(video_id_filter)]

        examples: list[BenchmarkExample] = []
        for video in videos:
            frames = video.iter_activity_frames() if activity_only else video.iter_frames()

            if require_relations:
                frames = [f for f in frames if len(f.relations) > 0]

            if sample_fraction is not None:
                n = max(1, round(len(frames) * float(sample_fraction)))
                if n < len(frames):
                    last = len(frames) - 1
                    positions = [i * last // (n - 1) for i in range(n)] if n > 1 else [last // 2]
                    frames = [frames[p] for p in positions]

            for frame in frames:
                objects_by_id = frame.objects_by_id()
                colors_by_id, relations_by_id = _parse_frame_objects(frame, objects_by_id)

                # Group valid objects by category.
                category_objects: dict[str, list[ObjectAnnotation]] = {}
                for obj in frame.objects:
                    if obj.category.name.lower() == 'unknown':
                        continue
                    if not _has_valid_bbox(obj, min_bbox_area):
                        continue
                    category_objects.setdefault(obj.category.name, []).append(obj)

                frame_input = FrameInput(
                    video_id=video.video_id,
                    frame_index=frame.frame_index,
                    mp4_path=video.mp4_path,
                )

                for category_name in sorted(category_objects):
                    objs = category_objects[category_name]

                    # Pick the richest description among objects of this category.
                    best_phrase = category_name
                    best_components = _PhraseComponents(frozenset(), None)
                    best_score: tuple[int, int] = (-1, -1)
                    for obj in objs:
                        desc, comp = _build_referring_description(obj, colors_by_id, relations_by_id, self._rng)
                        score = _phrase_score(comp)
                        if score > best_score:
                            best_score = score
                            best_phrase = desc
                            best_components = comp

                    # GT = bboxes of objects that match the winning phrase's components.
                    if best_score == (0, 0):
                        continue
                    gt_boxes: list[list[int]] = []
                    for obj in objs:
                        obj_colors = colors_by_id.get(obj.object_id, frozenset())
                        if obj_colors != best_components.colors:
                            continue
                        if best_components.relation is not None:
                            rel = best_components.relation
                            obj_rels = relations_by_id.get(obj.object_id, [])
                            if not any(
                                r.value == rel.value and r.target_category == rel.target_category for r in obj_rels
                            ):
                                continue
                        bbox_xyxy = obj.bbox_xyxy()
                        assert bbox_xyxy is not None
                        x1, y1, x2, y2 = bbox_xyxy
                        box_2d = pixel_to_normalized(
                            x1, y1, x2, y2, frame.width, frame.height, fmt=BboxFormat.XYXY, scale=self.bbox_scale
                        )
                        gt_boxes.append(box_2d)

                    if not gt_boxes:
                        continue

                    examples.append(
                        BenchmarkExample(
                            example_id=f'grounding:{video.video_id}:{frame.frame_index}:{category_name}',
                            task_name='grounding',
                            label=f'{video.video_id}_{frame.frame_index}_{category_name}',
                            frames=[frame_input],
                            metadata={
                                'video_id': video.video_id,
                                'frame_index': frame.frame_index,
                                'phrase': best_phrase,
                                'category': category_name,
                                'num_gt_boxes': len(gt_boxes),
                            },
                            task_data={
                                'phrase': best_phrase,
                                'category': category_name,
                                'ground_truth_boxes': gt_boxes,
                                'frame_width': frame.width,
                                'frame_height': frame.height,
                                'bbox_format': BboxFormat.XYXY.value,
                                'bbox_scale': self.bbox_scale,
                            },
                        )
                    )
        return examples

    def default_system_prompt(self) -> str:
        coord_desc = format_description(self.bbox_format)
        if self.bbox_format is BboxFormat.YXYX:
            coord_order = 'y-first: ymin (top), xmin (left), ymax (bottom), xmax (right)'
        else:
            coord_order = 'x-first: xmin (left), ymin (top), xmax (right), ymax (bottom)'
        return GROUNDING_SYSTEM_PROMPT_TEMPLATE.format(
            coord_desc=coord_desc,
            scale=self.bbox_scale,
            coord_order=coord_order,
        )

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        phrase = cast(str, example.task_data['phrase'])

        return [
            example.frames[0].to_image_part(),
            TextPart(text=f'Locate all instances of: "{phrase}"\n'),
        ]

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, Any] | None:
        return _parse_grounding_response(raw_text)

    def evaluate(self, example: BenchmarkExample, task_result: dict[str, Any]) -> dict[str, Any]:
        result = dict(task_result)
        gt_boxes = cast(list[list[int]], example.task_data['ground_truth_boxes'])
        predicted_boxes = cast(list[list[int]], result.get('predicted_boxes', []))
        frame_width = cast(int, example.task_data['frame_width'])
        frame_height = cast(int, example.task_data['frame_height'])
        gt_bbox_format = BboxFormat(str(example.task_data.get('bbox_format', 'xyxy')))

        gt_boxes_pixel = [
            normalized_to_pixel(box, frame_width, frame_height, fmt=gt_bbox_format, scale=self.bbox_scale)
            for box in gt_boxes
        ]
        pred_boxes_pixel = [
            normalized_to_pixel(box, frame_width, frame_height, fmt=self.bbox_format, scale=self.bbox_scale)
            for box in predicted_boxes
        ]

        result['gt_count'] = len(gt_boxes)
        result['pred_count'] = len(predicted_boxes)

        if pred_boxes_pixel and gt_boxes_pixel:
            ious = bbox_iou_matrix(pred_boxes_pixel, gt_boxes_pixel)
            max_ious_per_pred = [max(row) for row in ious]
            result['mean_best_iou'] = sum(max_ious_per_pred) / len(max_ious_per_pred)
        else:
            result['mean_best_iou'] = 0.0

        # Store for update_metrics() / save_debug_frame() without re-loading the example.
        result['_gt_boxes'] = gt_boxes
        result['_category'] = example.task_data['category']
        result['_frame_width'] = frame_width
        result['_frame_height'] = frame_height
        result['_pred_bbox_format'] = self.bbox_format.value

        return result

    def validate_example(self, example: BenchmarkExample) -> None:
        gt = cast(list[list[int]], example.task_data.get('ground_truth_boxes', []))
        if not gt:
            raise ValueError(f'Example {example.example_id} has empty ground_truth_boxes')
        phrase = example.task_data.get('phrase')
        if not phrase:
            raise ValueError(f'Example {example.example_id} has empty phrase')
        for box in gt:
            if len(box) != 4:
                raise ValueError(f'Example {example.example_id} has invalid box_2d: {box}')
            for coord in box:
                if not (0 <= coord <= self.bbox_scale):
                    raise ValueError(
                        f'Example {example.example_id} has box_2d coordinate out of [0, {self.bbox_scale}]: {coord}'
                    )

    def response_schema(self) -> type | None:
        return None

    def init_metrics(self, *, examples_total: int) -> dict[str, Any]:
        return {
            'examples_total': examples_total,
            'skipped_existing': 0,
            'provider_successes': 0,
            'call_failures': 0,
            'parse_failures': 0,
            'parsed_predictions': 0,
            'latency_samples': 0,
            'total_latency_ms': 0.0,
            'mean_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            '_all_preds': [],
            '_all_targets': [],
            '_category_to_id': {},
            'map': None,
            'map_50': None,
            'map_75': None,
            'map_small': None,
            'map_medium': None,
            'map_large': None,
            'mar_1': None,
            'mar_10': None,
            'mar_100': None,
            'mar_small': None,
            'mar_medium': None,
            'mar_large': None,
            'per_category': {},
        }

    def update_metrics(self, metrics: dict[str, Any], prediction: PredictionResult) -> None:
        if prediction.error is not None:
            if prediction.error.startswith('provider_error:'):
                metrics['call_failures'] = int(metrics['call_failures']) + 1
            else:
                metrics['provider_successes'] = int(metrics['provider_successes']) + 1
                metrics['parse_failures'] = int(metrics['parse_failures']) + 1
        else:
            metrics['provider_successes'] = int(metrics['provider_successes']) + 1
            metrics['parsed_predictions'] = int(metrics['parsed_predictions']) + 1

        if prediction.latency_ms is not None:
            latency = prediction.latency_ms
            metrics['latency_samples'] = int(metrics['latency_samples']) + 1
            metrics['total_latency_ms'] = float(metrics['total_latency_ms']) + latency
            current_min = metrics.get('min_latency_ms')
            current_max = metrics.get('max_latency_ms')
            metrics['min_latency_ms'] = latency if current_min is None else min(float(current_min), latency)
            metrics['max_latency_ms'] = latency if current_max is None else max(float(current_max), latency)

        if prediction.error is not None and prediction.error.startswith('provider_error:'):
            return

        task_result = prediction.task_result
        gt_boxes = cast(list[list[int]], task_result.get('_gt_boxes', []))
        if not gt_boxes:
            return

        predicted_boxes = cast(list[list[int]], task_result.get('predicted_boxes', []))
        category = cast(str, task_result.get('_category', 'unknown'))
        category_to_id = cast(dict[str, int], metrics['_category_to_id'])
        if category not in category_to_id:
            category_to_id[category] = len(category_to_id)
        cat_id = category_to_id[category]
        frame_width = cast(int, task_result.get('_frame_width', 1))
        frame_height = cast(int, task_result.get('_frame_height', 1))

        metrics['_all_preds'].append(
            _to_torchmetrics_entry(
                predicted_boxes,
                cat_id,
                frame_width,
                frame_height,
                bbox_fmt=self.bbox_format,
                bbox_scale=self.bbox_scale,
                with_scores=True,
            )
        )
        metrics['_all_targets'].append(
            _to_torchmetrics_entry(
                gt_boxes,
                cat_id,
                frame_width,
                frame_height,
                bbox_fmt=BboxFormat.XYXY,
                bbox_scale=self.bbox_scale,
                with_scores=False,
            )
        )

    def finalize_metrics(self, metrics: dict[str, Any]) -> None:
        latency_samples = int(metrics['latency_samples'])
        if latency_samples > 0:
            metrics['mean_latency_ms'] = float(metrics['total_latency_ms']) / latency_samples

        all_preds = metrics.pop('_all_preds', [])
        all_targets = metrics.pop('_all_targets', [])
        category_to_id = metrics.pop('_category_to_id', {})

        if not all_preds or not all_targets:
            return

        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
        for pred_entry, target_entry in zip(all_preds, all_targets):
            pred_boxes = pred_entry['boxes']
            pred_torch = {
                'boxes': torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4),
                'scores': torch.tensor(pred_entry['scores'], dtype=torch.float32),
                'labels': torch.tensor(pred_entry['labels'], dtype=torch.int64),
            }
            target_boxes = target_entry['boxes']
            target_torch = {
                'boxes': torch.tensor(target_boxes, dtype=torch.float32).reshape(-1, 4),
                'labels': torch.tensor(target_entry['labels'], dtype=torch.int64),
            }
            metric.update(preds=[pred_torch], target=[target_torch])  # type: ignore[arg-type]

        result = metric.compute()  # type: ignore[arg-type]
        for key in (
            'map',
            'map_50',
            'map_75',
            'map_small',
            'map_medium',
            'map_large',
            'mar_1',
            'mar_10',
            'mar_100',
            'mar_small',
            'mar_medium',
            'mar_large',
        ):
            val = result.get(key)
            metrics[key] = val.item() if val is not None else None

        id_to_category = {v: k for k, v in category_to_id.items()}
        map_per_class = result.get('map_per_class')
        mar_100_per_class = result.get('mar_100_per_class')
        classes = result.get('classes')
        per_category: dict[str, dict[str, float | None]] = {}
        if map_per_class is not None and map_per_class.numel() > 0:
            for i in range(map_per_class.shape[0]):
                cat_idx = classes[i].item() if classes is not None else i
                cat_name = id_to_category.get(cat_idx, f'class_{cat_idx}')
                cat_map = map_per_class[i].item()
                cat_mar = mar_100_per_class[i].item() if mar_100_per_class is not None else None
                per_category[cat_name] = {
                    'map': cat_map if cat_map >= 0 else None,
                    'mar_100': cat_mar if cat_mar is not None and cat_mar >= 0 else None,
                }
        metrics['per_category'] = per_category

    def format_summary(self, metrics: dict[str, Any], *, run_dir: Path) -> str:
        lines = [
            f'Run directory: {run_dir}',
            f'Examples total: {metrics["examples_total"]}',
            f'Skipped existing: {metrics["skipped_existing"]}',
            f'Provider successes: {metrics["provider_successes"]}',
            f'Call failures: {metrics["call_failures"]}',
            f'Parse failures: {metrics["parse_failures"]}',
            f'Parsed predictions: {metrics["parsed_predictions"]}',
            '',
            f'mAP @[.5:.95]: {_fmt_metric(metrics.get("map"))}',
            f'mAP @0.50:     {_fmt_metric(metrics.get("map_50"))}',
            f'mAP @0.75:     {_fmt_metric(metrics.get("map_75"))}',
            f'mAP @small:    {_fmt_metric(metrics.get("map_small"))}',
            f'mAP @medium:   {_fmt_metric(metrics.get("map_medium"))}',
            f'mAP @large:    {_fmt_metric(metrics.get("map_large"))}',
            f'mAR @1:        {_fmt_metric(metrics.get("mar_1"))}',
            f'mAR @10:       {_fmt_metric(metrics.get("mar_10"))}',
            f'mAR @100:      {_fmt_metric(metrics.get("mar_100"))}',
            f'mAR @small:    {_fmt_metric(metrics.get("mar_small"))}',
            f'mAR @medium:   {_fmt_metric(metrics.get("mar_medium"))}',
            f'mAR @large:    {_fmt_metric(metrics.get("mar_large"))}',
            '',
            f'Wall-clock time: {_fmt_latency(metrics.get("wall_clock_ms"))}',
            f'Concurrency: {metrics.get("concurrency", 1)}',
            f'Mean request latency: {_fmt_latency(metrics.get("mean_latency_ms"))}',
            f'Min request latency: {_fmt_latency(metrics.get("min_latency_ms"))}',
            f'Max request latency: {_fmt_latency(metrics.get("max_latency_ms"))}',
        ]
        per_category = dict(metrics.get('per_category') or {})
        if per_category:
            lines.append('')
            lines.append('Per-category mAP:')
            for cat_name in sorted(per_category):
                stats = per_category[cat_name]
                cat_map = _fmt_metric(stats.get('map'))
                cat_mar = _fmt_metric(stats.get('mar_100'))
                lines.append(f'  {cat_name}: mAP={cat_map}  mAR@100={cat_mar}')
        return '\n'.join(lines)
