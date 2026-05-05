"""Hand-object interaction benchmark task (egocentric video).

One example is generated **per frame** that contains at least one annotated
hand-object interaction.  The model is shown the frame and must output **all**
interactions it can find, each as a JSON object with:

- ``hand_boxes``: 1 or 2 bounding boxes for the interacting hand(s)
- ``object_box``: bounding box for the interacted object
- ``hand_type``: ``"left"``, ``"right"``, or ``"both"``

The output schema is a **JSON array** (one element per interaction).

Ground truth is derived from relations in the COCO annotation files:

- ``relation_type == "human_actions"`` (configurable via
  ``task_params.interaction_relation_type``)
- Source object category name must be ``"left hand"`` or ``"right hand"``
  (inferred from the relation's source object; no config needed).
- Relations pointing from hand(s) to the **same** target object are merged
  into a single interaction (enabling ``hand_type = "both"``).

Evaluation uses interaction-level metrics (same as CaRe-Ego for fair comparison):

- **Interaction recall/precision**: greedy match by object-box IoU (>0)
- **Hand IoU / Object IoU**: per-component quality on matched pairs
- **Hand-type accuracy**: on matched pairs
- **No-detection rate**: frames with GT but zero pred
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from barista.bbox import BboxFormat, convert, format_description
from barista.dataset import load_videos
from barista.hand_object_helpers import (
    compute_interaction_metrics_for_frame,
    frame_interactions,
    interaction_to_dict,
)
from barista.hand_object_helpers import (
    save_debug_frame as _save_debug_frame,
)
from barista.vlm_benchmarks.tasks.base import BenchmarkTask
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    FrameInput,
    MessagePart,
    PredictionResult,
    TextPart,
)

# Matches ``` optionally followed by a language tag, then newline, then content.
_MARKDOWN_FENCE_PATTERN = re.compile(r'```(?:\w*)\n(.*?)```', re.DOTALL)
_MATCH_IOU_THRESHOLD = 0.5

HAND_OBJECT_SYSTEM_PROMPT_TEMPLATE = (
    'You are analyzing egocentric (first-person) video frames. Identify every '
    "hand-object interaction: one or both of the camera wearer's hands actively "
    'manipulating, holding, or touching an object.\n'
    '\n'
    'For each interaction output:\n'
    '- "hand_boxes": a list of 1 or 2 bounding boxes — one per interacting hand.\n'
    '- "object_box": a single bounding box for the object being interacted with.\n'
    '- "hand_type": which hand(s) are involved in this interaction — "left", '
    '"right", or "both".\n'
    '\n'
    'Each interaction is a separate element. If the same hand touches multiple '
    'objects, report each interaction separately.\n'
    '\n'
    'Bounding box format: {coord_desc}, integers 0-{scale}. Order: {coord_order}\n'
    '\n'
    'Return ONLY a JSON array — one element per interaction. If none, return [].\n'
    '\n'
    'Example (right hand):\n'
    '[{{"hand_boxes": [[120, 80, 320, 280]], "object_box": [350, 200, 600, 450], '
    '"hand_type": "right"}}]\n'
    '\n'
    'Example (both hands on same object):\n'
    '[{{"hand_boxes": [[50, 60, 200, 250], [400, 70, 550, 260]], '
    '"object_box": [180, 100, 420, 350], "hand_type": "both"}}]\n'
    '\n'
    'Example (two separate interactions):\n'
    '[{{"hand_boxes": [[60, 100, 210, 290]], "object_box": [230, 150, 410, 320], '
    '"hand_type": "left"}}, '
    '{{"hand_boxes": [[520, 90, 680, 280]], "object_box": [300, 330, 700, 550], '
    '"hand_type": "right"}}]\n'
)


# ---------------------------------------------------------------------------
# Debug visualization (delegates to hand_object_helpers)
# ---------------------------------------------------------------------------


def save_debug_frame(
    example: BenchmarkExample,
    prediction: PredictionResult,
    out_path: Path,
) -> None:
    """Draw GT (green) and pred (red) bboxes on a frame and save for debugging."""
    gt_dicts = cast(list[dict[str, Any]], example.task_data['ground_truth'])
    pred_dicts = cast(list[dict[str, Any]], prediction.task_result.get('interactions', []))
    frame_width = cast(int, example.task_data['frame_width'])
    frame_height = cast(int, example.task_data['frame_height'])
    bbox_scale = int(example.task_data.get('bbox_scale', 1000))
    # GT is always XYXY; pred uses the run-config format. Normalise pred to XYXY.
    pred_bbox_fmt = BboxFormat(str(prediction.task_result.get('_pred_bbox_format', 'yxyx')))
    if pred_bbox_fmt is not BboxFormat.XYXY:
        pred_dicts = [
            {
                **d,
                'hand_boxes': [convert(b, src=pred_bbox_fmt, dst=BboxFormat.XYXY) for b in d.get('hand_boxes', [])],
                'object_box': convert(d['object_box'], src=pred_bbox_fmt, dst=BboxFormat.XYXY),
            }
            for d in pred_dicts
        ]

    _save_debug_frame(
        image=example.frames[0].load_pil_image(),
        out_path=out_path,
        gt_dicts=gt_dicts,
        pred_dicts=pred_dicts,
        img_w=frame_width,
        img_h=frame_height,
        bbox_fmt=BboxFormat.XYXY,
        bbox_scale=bbox_scale,
        pred_boxes_in_pixel=False,
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _try_parse_json_array(text: str) -> list | None:
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _iter_json_arrays(text: str) -> list[list]:
    """Return JSON arrays decoded from arbitrary surrounding text."""
    decoder = json.JSONDecoder()
    arrays: list[list] = []
    for match in re.finditer(r'\[', text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, list):
            arrays.append(value)
    return arrays


def _validate_interactions(parsed: list) -> list[dict[str, Any]] | None:
    """Validate and normalise a parsed JSON array of interactions."""
    validated: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            return None
        hand_boxes = entry.get('hand_boxes')
        object_box = entry.get('object_box')
        hand_type = entry.get('hand_type')

        if not isinstance(hand_boxes, list) or not hand_boxes:
            return None
        if not isinstance(object_box, list) or len(object_box) != 4:
            return None
        if not isinstance(hand_type, str) or hand_type not in {'left', 'right', 'both'}:
            return None

        norm_hand_boxes: list[list[int]] = []
        for box in hand_boxes:
            if not isinstance(box, list) or len(box) != 4:
                return None
            try:
                norm_hand_boxes.append([int(round(float(v))) for v in box])
            except (TypeError, ValueError):
                return None

        try:
            norm_object_box = [int(round(float(v))) for v in object_box]
        except (TypeError, ValueError):
            return None

        validated.append(
            {
                'hand_boxes': norm_hand_boxes,
                'object_box': norm_object_box,
                'hand_type': hand_type,
            }
        )
    return validated


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return 'N/A'
    return f'{float(value):.4f}'


def _fmt_latency(value: float | None) -> str:
    if value is None:
        return 'N/A'
    ms = float(value)
    return f'{ms / 1000.0:.2f}s' if ms >= 1000.0 else f'{ms:.0f}ms'


class HandObjectTask(BenchmarkTask):
    """Egocentric hand-object interaction detection.

    One ``BenchmarkExample`` is generated per frame that contains at least one
    annotated hand–object interaction.  ``task_data["ground_truth"]`` is a list
    of interaction dicts (one per GT interaction in the frame).

    Supported ``task_params`` keys:

    - ``bbox_format`` (str, default ``'yxyx'``): ``'xyxy'`` or ``'yxyx'``.
    - ``bbox_scale`` (int, default ``1000``): normalisation scale.
    - ``interaction_relation_type`` (str, default ``'human_actions'``): the
      relation type used to encode hand–object interaction in the dataset.
    - ``video_id`` (str | null): restrict to a single video.
    - ``sample_fraction`` (float | null, default null): if set, evaluate only
      this fraction of frames per video (0.0–1.0], evenly spaced.

    Hand categories are inferred from the source object's category name
    (``"left hand"`` / ``"right hand"``, case-insensitive).
    """

    def __init__(self, dataset_root: Path, task_params: dict[str, Any]) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)
        self.bbox_format = BboxFormat(self.task_params.get('bbox_format', 'yxyx'))
        self.bbox_scale = int(self.task_params.get('bbox_scale', 1000))
        self.interaction_relation_type = str(self.task_params.get('interaction_relation_type', 'human_actions'))

    # ------------------------------------------------------------------
    # Example generation
    # ------------------------------------------------------------------

    def iter_examples(self) -> list[BenchmarkExample]:
        videos = load_videos(self.dataset_root)
        video_id_filter = self.task_params.get('video_id')
        if video_id_filter is not None:
            videos = [v for v in videos if v.video_id == str(video_id_filter)]

        sample_fraction = self.task_params.get('sample_fraction')

        examples: list[BenchmarkExample] = []
        for video in videos:
            frame_indices = video.frame_indices()
            if sample_fraction is not None:
                n = max(1, round(len(frame_indices) * float(sample_fraction)))
                if n < len(frame_indices):
                    last = len(frame_indices) - 1
                    positions = [i * last // (n - 1) for i in range(n)] if n > 1 else [last // 2]
                    frame_indices = [frame_indices[p] for p in positions]
            for frame_index in frame_indices:
                frame = video.frame(frame_index)
                interactions = frame_interactions(
                    frame=frame,
                    interaction_relation_type=self.interaction_relation_type,
                    bbox_fmt=BboxFormat.XYXY,
                    bbox_scale=self.bbox_scale,
                )
                if not interactions:
                    continue

                examples.append(
                    BenchmarkExample(
                        example_id=f'hand_object:{video.video_id}:{frame_index}',
                        task_name='hand_object',
                        label=f'{video.video_id}_{frame_index}',
                        frames=[
                            FrameInput(
                                video_id=video.video_id,
                                frame_index=frame_index,
                                mp4_path=video.mp4_path,
                            )
                        ],
                        metadata={
                            'video_id': video.video_id,
                            'frame_index': frame_index,
                            'num_interactions': len(interactions),
                            'hand_types': [ia.hand_type for ia in interactions],
                        },
                        task_data={
                            'ground_truth': [interaction_to_dict(ia) for ia in interactions],
                            'frame_width': frame.width,
                            'frame_height': frame.height,
                            'bbox_format': BboxFormat.XYXY.value,
                            'bbox_scale': self.bbox_scale,
                        },
                    )
                )
        return examples

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def default_system_prompt(self) -> str:
        coord_desc = format_description(self.bbox_format)
        if self.bbox_format is BboxFormat.YXYX:
            coord_order = 'y-first: ymin (top), xmin (left), ymax (bottom), xmax (right)'
        else:
            coord_order = 'x-first: xmin (left), ymin (top), xmax (right), ymax (bottom)'
        return HAND_OBJECT_SYSTEM_PROMPT_TEMPLATE.format(
            coord_desc=coord_desc,
            scale=self.bbox_scale,
            coord_order=coord_order,
        )

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        return [
            example.frames[0].to_image_part(),
            TextPart(text='Analyze this frame.'),
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()

        # Strip markdown code fences if present.
        fence_match = _MARKDOWN_FENCE_PATTERN.search(text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Try direct JSON array parse.
        parsed = _try_parse_json_array(text)
        candidates = [parsed] if parsed is not None else _iter_json_arrays(text)

        for candidate in candidates:
            interactions = _validate_interactions(candidate)
            if interactions is not None:
                return {'interactions': interactions}

        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, example: BenchmarkExample, task_result: dict[str, Any]) -> dict[str, Any]:
        result = dict(task_result)
        result['_pred_bbox_format'] = self.bbox_format.value
        gt_dicts = cast(list[dict[str, Any]], example.task_data['ground_truth'])
        frame_width = cast(int, example.task_data['frame_width'])
        frame_height = cast(int, example.task_data['frame_height'])

        # GT is always stored as xyxy (canonical). Convert to the run-config
        # bbox_format so that compute_interaction_metrics_for_frame sees a
        # consistent format for both GT and predicted boxes.
        if self.bbox_format is not BboxFormat.XYXY:
            gt_dicts = [
                {
                    **d,
                    'hand_boxes': [convert(b, src=BboxFormat.XYXY, dst=self.bbox_format) for b in d['hand_boxes']],
                    'object_box': convert(d['object_box'], src=BboxFormat.XYXY, dst=self.bbox_format),
                }
                for d in gt_dicts
            ]

        pred_interactions = cast(list[dict[str, Any]], result.get('interactions', []))

        frame_metrics = compute_interaction_metrics_for_frame(
            gt_dicts,
            pred_interactions,
            frame_width,
            frame_height,
            self.bbox_format,
            self.bbox_scale,
            iou_threshold=_MATCH_IOU_THRESHOLD,
        )
        result['num_gt_interactions'] = frame_metrics['total_gt']
        result['num_pred_interactions'] = frame_metrics['total_pred']
        result['_frame_metrics'] = frame_metrics

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_example(self, example: BenchmarkExample) -> None:
        gt = cast(list[dict[str, Any]], example.task_data.get('ground_truth', []))
        if not gt:
            raise ValueError(f'Example {example.example_id} has empty ground_truth')
        for ia in gt:
            hand_boxes = ia.get('hand_boxes', [])
            object_box = ia.get('object_box', [])
            if not isinstance(hand_boxes, list) or not hand_boxes:
                raise ValueError(f'Example {example.example_id}: interaction has invalid hand_boxes')
            if not isinstance(object_box, list) or len(object_box) != 4:
                raise ValueError(f'Example {example.example_id}: interaction has invalid object_box')

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

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
            # interaction-level accumulators
            'total_gt': 0,
            'total_pred': 0,
            'matched_gt': 0,
            'matched_pred': 0,
            'hand_ious': [],
            'object_ious': [],
            'hand_type_correct': 0,
            'frames_with_gt': 0,
            'frames_with_gt_but_zero_pred': 0,
            # computed
            'interaction_recall': None,
            'interaction_precision': None,
            'hand_iou': None,
            'object_iou': None,
            'hand_type_accuracy': None,
            'no_detection_rate': None,
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
            lat = prediction.latency_ms
            metrics['latency_samples'] = int(metrics['latency_samples']) + 1
            metrics['total_latency_ms'] = float(metrics['total_latency_ms']) + lat
            cur_min = metrics.get('min_latency_ms')
            cur_max = metrics.get('max_latency_ms')
            metrics['min_latency_ms'] = lat if cur_min is None else min(float(cur_min), lat)
            metrics['max_latency_ms'] = lat if cur_max is None else max(float(cur_max), lat)

        if prediction.error is not None and prediction.error.startswith('provider_error:'):
            return

        frame_metrics = cast(dict, prediction.task_result.get('_frame_metrics', {}))
        if not frame_metrics:
            return

        metrics['total_gt'] = int(metrics['total_gt']) + int(frame_metrics['total_gt'])
        metrics['total_pred'] = int(metrics['total_pred']) + int(frame_metrics['total_pred'])
        metrics['matched_gt'] = int(metrics['matched_gt']) + int(frame_metrics['matched_gt'])
        metrics['matched_pred'] = int(metrics['matched_pred']) + int(frame_metrics['matched_pred'])
        metrics['hand_ious'].extend(frame_metrics.get('hand_ious', []))
        metrics['object_ious'].extend(frame_metrics.get('object_ious', []))
        metrics['hand_type_correct'] = int(metrics['hand_type_correct']) + int(
            frame_metrics.get('hand_type_correct', 0)
        )
        if frame_metrics.get('has_gt'):
            metrics['frames_with_gt'] = int(metrics['frames_with_gt']) + 1
        if frame_metrics.get('has_gt_but_zero_pred'):
            metrics['frames_with_gt_but_zero_pred'] = int(metrics['frames_with_gt_but_zero_pred']) + 1

    def finalize_metrics(self, metrics: dict[str, Any]) -> None:
        if int(metrics['latency_samples']) > 0:
            metrics['mean_latency_ms'] = float(metrics['total_latency_ms']) / int(metrics['latency_samples'])

        total_gt = int(metrics['total_gt'])
        total_pred = int(metrics['total_pred'])
        matched_gt = int(metrics['matched_gt'])
        matched_pred = int(metrics['matched_pred'])
        hand_ious = list(metrics.get('hand_ious', []))
        object_ious = list(metrics.get('object_ious', []))
        frames_with_gt = int(metrics['frames_with_gt'])
        frames_with_gt_but_zero_pred = int(metrics['frames_with_gt_but_zero_pred'])

        metrics['interaction_recall'] = matched_gt / total_gt if total_gt > 0 else 0.0
        metrics['interaction_precision'] = matched_pred / total_pred if total_pred > 0 else 0.0
        metrics['hand_iou'] = sum(hand_ious) / len(hand_ious) if hand_ious else None
        metrics['object_iou'] = sum(object_ious) / len(object_ious) if object_ious else None
        metrics['hand_type_accuracy'] = metrics['hand_type_correct'] / matched_gt if matched_gt > 0 else None
        metrics['no_detection_rate'] = frames_with_gt_but_zero_pred / frames_with_gt if frames_with_gt > 0 else 0.0
        # Remove large accumulators before writing to metrics.json
        metrics.pop('hand_ious', None)
        metrics.pop('object_ious', None)

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
            f'Interaction recall: {_fmt_metric(metrics.get("interaction_recall"))}',
            f'Interaction precision: {_fmt_metric(metrics.get("interaction_precision"))}',
            f'Hand IoU (matched): {_fmt_metric(metrics.get("hand_iou"))}',
            f'Object IoU (matched): {_fmt_metric(metrics.get("object_iou"))}',
            f'Hand-type accuracy: {_fmt_metric(metrics.get("hand_type_accuracy"))} '
            f'({metrics.get("hand_type_correct", 0)}/{metrics.get("matched_gt", 0)} matched)',
            f'No-detection rate: {_fmt_metric(metrics.get("no_detection_rate"))}',
            '',
            f'Wall-clock time: {_fmt_latency(metrics.get("wall_clock_ms"))}',
            f'Concurrency: {metrics.get("concurrency", 1)}',
            f'Mean request latency: {_fmt_latency(metrics.get("mean_latency_ms"))}',
            f'Min request latency: {_fmt_latency(metrics.get("min_latency_ms"))}',
            f'Max request latency: {_fmt_latency(metrics.get("max_latency_ms"))}',
        ]
        return '\n'.join(lines)
