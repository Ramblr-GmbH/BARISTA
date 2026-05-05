"""Relation extraction benchmark."""

from __future__ import annotations

import colorsys
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from barista.dataset import FrameAnnotation, ObjectAnnotation, Video, load_videos
from barista.vlm_benchmarks.tasks.base import BenchmarkTask
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    FrameInput,
    ImagePart,
    MessagePart,
    PredictionResult,
    TextPart,
)

# Single definition of the model output shape (system prompt + parser must stay aligned).
RELATION_OUTPUT_JSON_INSTRUCTIONS = (
    'The final response must be a JSON array of relation objects. '
    'If there are no relations, return []. '
    'Each element must be an object with keys: '
    'source_id (integer), target_id (integer), relation_type (string), value (string). '
    'Use only relation_type and value strings that appear together under the same relation_type '
    'in the user message (each value is valid only for its listed type).\n'
    '\n'
    'Example:\n'
    '[{"source_id": 1, "target_id": 2, "relation_type": "position", "value": "part of"}]'
)

RELATION_EXTRACTION_SYSTEM_PROMPT_BASE = (
    'You extract pairwise relations between objects in a coffee-making scene. '
    'Each object has a numeric ID on the set-of-mark image and in the ID-to-category list.\n'
    '\n' + RELATION_OUTPUT_JSON_INSTRUCTIONS
)

_MARKDOWN_FENCE_PATTERN = re.compile(r'```(?:\w*)\n(.*?)```', re.DOTALL)


def _display_id_to_object(frame: FrameAnnotation) -> dict[int, ObjectAnnotation]:
    """Map 1-based display id → object; key order is 1, 2, … and matches SOM / prompt / GT."""
    ordered = sorted(
        frame.objects,
        key=lambda o: (o.category.name.lower(), str(o.object_id)),
    )
    return {i: obj for i, obj in enumerate(ordered, start=1)}


def _ground_truth_relations(
    frame: FrameAnnotation,
    display_id_to_object: dict[int, ObjectAnnotation],
) -> list[dict[str, object]]:
    """GT relation rows with ``source_id`` / ``target_id`` as display ids (1-based)."""
    uuid_to_disp = {obj.object_id: did for did, obj in display_id_to_object.items()}
    rows: list[dict[str, object]] = []
    for rel in frame.relations:
        s = uuid_to_disp.get(rel.source_object_id)
        t = uuid_to_disp.get(rel.target_object_id)
        if s is None or t is None:
            continue
        rows.append(
            {
                'source_id': int(s),
                'target_id': int(t),
                'relation_type': rel.relation_type,
                'value': rel.value,
            }
        )
    return rows


def _set_of_mark_seg_and_id_coords(
    frame: FrameAnnotation,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """Per-pixel object display id (H×W, ``0`` = background) and centroid per id for labels."""
    h, w = frame.height, frame.width
    label_map = np.zeros((h, w), dtype=np.int64)
    id_coords: dict[int, tuple[int, int]] = {}
    for display_id, obj in _display_id_to_object(frame).items():
        m = obj.mask_array(h, w).astype(bool)
        ys, xs = np.where(m)
        label_map[m] = display_id
        id_coords[display_id] = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
    return label_map, id_coords


def _instance_boundaries(label_map: np.ndarray) -> np.ndarray:
    """True where a pixel's 4-neighbour label differs (instance edges)."""
    padded = np.pad(label_map, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    hh, ww = label_map.shape
    center = padded[1 : 1 + hh, 1 : 1 + ww]
    bd = np.zeros((hh, ww), dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neigh = padded[1 + dy : 1 + dy + hh, 1 + dx : 1 + dx + ww]
        bd |= (neigh != center) & (center != 0)
    return bd


def _segmentation_overlay_numpy(
    label_map: np.ndarray,
    base_rgb: np.ndarray,
    alpha_segmentation_boundary: float,
    alpha_segmentation: float,
) -> np.ndarray:
    """Paint each non-zero ``label_map`` id with a distinct hue; blend onto ``base_rgb`` with alpha.

    Boundary pixels use ``alpha_segmentation_boundary`` to mix edge vs fill colour; foreground
    uses ``alpha_segmentation`` to mix overlay vs ``base_rgb``. Background (label ``0``) stays
    identical to ``base_rgb``.
    """
    if label_map.dtype != np.int64:
        label_map = label_map.astype(np.int64)
    background = label_map == 0
    if int(label_map.max()) == 0:
        return base_rgb

    h, w = label_map.shape
    inner = np.zeros((h, w, 3), dtype=np.float32)
    boundary_rgb = np.zeros((h, w, 3), dtype=np.float32)
    boundaries = _instance_boundaries(label_map)
    ab = float(alpha_segmentation_boundary)

    for display_id in np.unique(label_map):
        if display_id == 0:
            continue
        did = int(display_id)
        hue = (did * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        fill_rgb = np.array([r * 255.0, g * 255.0, b * 255.0], dtype=np.float32)
        m = label_map == display_id
        inner[m] = fill_rgb
        boundary_rgb[m & boundaries] = fill_rgb

    mixed_inner = ab * boundary_rgb + (1.0 - ab) * inner
    mixed_inner[background] = 0.0
    a = float(alpha_segmentation)
    out_f = base_rgb.astype(np.float32) * (1.0 - a) + mixed_inner * a
    out = np.clip(np.round(out_f), 0, 255).astype(np.uint8)
    out[background] = base_rgb[background]
    return out


def _add_instance_ids(
    overlay_rgb: np.ndarray,
    id_coords: Mapping[int, tuple[int, int]],
    font_size: int,
) -> np.ndarray:
    """Draw display ids at mask centroids on ``overlay_rgb`` (uint8 RGB, same size in/out)."""
    if not id_coords:
        return overlay_rgb
    im = Image.fromarray(overlay_rgb, mode='RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=font_size)
    stroke = max(1, round(font_size / 14.0))
    for display_id, (cx, cy) in id_coords.items():
        draw.text(
            (cx, cy),
            str(display_id),
            fill='white',
            font=font,
            anchor='mm',
            stroke_width=stroke,
            stroke_fill='black',
        )
    return np.asarray(im, dtype=np.uint8)


def _set_of_marks_rgb(
    frame: FrameAnnotation,
    frame_pil: Image.Image,
    options: SetOfMarkOptions,
) -> np.ndarray:
    """Return uint8 RGB raster: coloured instance overlay on the frame + id text at centroids."""
    if frame_pil.size != (frame.width, frame.height):
        frame_pil = frame_pil.resize((frame.width, frame.height), Image.Resampling.BILINEAR)
    base_rgb = np.asarray(frame_pil, dtype=np.uint8)
    label_map, id_coords = _set_of_mark_seg_and_id_coords(frame)
    overlay_rgb = _segmentation_overlay_numpy(
        label_map,
        base_rgb,
        options.alpha_segmentation_boundary,
        options.mask_alpha,
    )
    return _add_instance_ids(overlay_rgb, id_coords, options.font_size)


def _collect_relation_vocab(videos: list[Video]) -> dict[str, list[str]]:
    """Map each ``relation_type`` to sorted distinct ``value`` strings seen in the dataset."""
    type_to_values: dict[str, set[str]] = {}
    for video in videos:
        for frame in video.frames.values():
            for rel in frame.relations:
                type_to_values.setdefault(rel.relation_type, set()).add(rel.value)
    return {t: sorted(vs) for t, vs in sorted(type_to_values.items())}


def _format_relation_vocab_for_prompt(relation_vocab: dict[str, list[str]]) -> str:
    if not relation_vocab:
        return '  (none)'
    lines: list[str] = []
    for rtype in sorted(relation_vocab.keys()):
        vals = relation_vocab[rtype]
        if not vals:
            lines.append(f'  {rtype}:\n    (none)')
            continue
        indented = '\n'.join(f'    - {v}' for v in vals)
        lines.append(f'  {rtype}:\n{indented}')
    return '\n'.join(lines)


def _relations_to_tuples(rows: list[dict[str, object]]) -> list[tuple[int, int, str, str]]:
    return [
        (
            int(r['source_id']),
            int(r['target_id']),
            str(r['relation_type']).strip().casefold(),
            str(r['value']).strip().casefold(),
        )
        for r in rows
    ]


def _relation_scores_from_counts(tp: int, pred_count: int, gt_count: int) -> dict[str, float | int]:
    """Multiset P/R/F1 from aggregate true-positive and multiset sizes."""
    if pred_count == 0:
        precision = 1.0 if gt_count == 0 else 0.0
    else:
        precision = tp / pred_count
    if gt_count == 0:
        recall = 1.0 if pred_count == 0 else 0.0
    else:
        recall = tp / gt_count
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        'tp': tp,
        'pred_count': pred_count,
        'gt_count': gt_count,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def _relation_multiset_scores(
    pred: list[tuple[int, int, str, str]],
    gt: list[tuple[int, int, str, str]],
) -> dict[str, float | int]:
    cp = Counter(pred)
    cg = Counter(gt)
    tp = int(sum((cp & cg).values()))
    return _relation_scores_from_counts(tp, int(sum(cp.values())), int(sum(cg.values())))


def _get_relation_scores(
    pred: list[tuple[int, int, str, str]],
    gt: list[tuple[int, int, str, str]],
) -> dict[str, object]:
    """Multiset P/R/F1 over all tuples plus the same breakdown per ``relation_type`` (tuple index 2)."""
    out: dict[str, object] = dict(_relation_multiset_scores(pred, gt))
    type_keys = {t[2] for t in pred} | {t[2] for t in gt}
    out['per_relation_type'] = {
        rk: _relation_multiset_scores([t for t in pred if t[2] == rk], [t for t in gt if t[2] == rk])
        for rk in sorted(type_keys)
    }
    return out


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


def _parse_relation_extraction_response(raw_text: str) -> dict[str, object] | None:
    """Parse model JSON into ``predicted_relations`` (same flow as grounding)."""
    text = raw_text.strip()
    fence_match = _MARKDOWN_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = _try_parse_json_array(text)
    if parsed is not None:
        return _validate_predicted_relations(parsed)

    for candidate in _iter_json_arrays(text):
        result = _validate_predicted_relations(candidate)
        if result['predicted_relations'] or candidate == []:
            return result

    return None


def _validate_predicted_relations(parsed: list) -> dict[str, object]:
    """Validate and normalise each array element; drop invalid entries, keep the rest."""
    predicted_relations: list[dict[str, object]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        src = entry.get('source_id')
        tgt = entry.get('target_id')
        rtype = entry.get('relation_type')
        val = entry.get('value')
        if src is None or tgt is None or rtype is None or val is None:
            continue
        try:
            si = int(src)
            ti = int(tgt)
        except (TypeError, ValueError):
            continue
        predicted_relations.append(
            {
                'source_id': si,
                'target_id': ti,
                'relation_type': str(rtype),
                'value': str(val),
            }
        )
    return {'predicted_relations': predicted_relations}


@dataclass(frozen=True)
class SetOfMarkOptions:
    """Set-of-mark rendering options."""

    mask_alpha: float = 0.5
    alpha_segmentation_boundary: float = 0.9
    font_size: int = 15


class RelationExtractionTask(BenchmarkTask):
    """Structured relation extraction with raw frame + set-of-mark and constrained vocabularies.

    Supported ``task_params`` keys:

    - ``frame_step`` (int, default 1): subsample frames by this stride.
    - ``video_id`` (str | null): optional filter to a single video directory id.
    - ``mask_alpha``, ``alpha_segmentation_boundary``, ``font_size``: passed to ``SetOfMarkOptions``.

    Frames with no annotated relations are skipped.
    """

    def __init__(self, dataset_root: Path, task_params: dict[str, object]) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)
        self._som_options = SetOfMarkOptions(
            mask_alpha=self.task_params.get('mask_alpha', 0.5),
            alpha_segmentation_boundary=self.task_params.get('alpha_segmentation_boundary', 0.9),
            font_size=self.task_params.get('font_size', 12),
        )
        self._video_cache: dict[str, Video] = {}
        self._relation_vocab: dict[str, list[str]] = _collect_relation_vocab(load_videos(self.dataset_root))

    def _get_video(self, frame_input: FrameInput) -> Video:
        """Load a single video by id, preferring the example's resolved MP4 path."""
        video_id = frame_input.video_id
        if video_id not in self._video_cache:
            video_dir = frame_input.mp4_path.parent
            if not video_dir.is_dir():
                video_dir = self.dataset_root / video_id
            video = Video.from_dir(video_dir)
            video.video_id = video_id
            self._video_cache[video_id] = video
        return self._video_cache[video_id]

    def iter_examples(self) -> list[BenchmarkExample]:
        videos = load_videos(self.dataset_root)
        for v in videos:
            self._video_cache[v.video_id] = v
        video_id_filter = self.task_params.get('video_id')
        if video_id_filter is not None:
            videos = [v for v in videos if v.video_id == str(video_id_filter)]

        frame_step = int(self.task_params.get('frame_step', 1))
        if frame_step <= 0:
            raise ValueError('frame_step must be >= 1')

        examples: list[BenchmarkExample] = []
        for video in videos:
            frame_indices = video.frame_indices()
            sampled = frame_indices[::frame_step] if frame_step > 1 else frame_indices
            for frame_index in sampled:
                frame = video.frame(frame_index)
                if len(frame.objects) == 0:
                    continue
                display_id_to_object = _display_id_to_object(frame)
                gt_rows = _ground_truth_relations(frame, display_id_to_object)
                if not gt_rows:
                    # Frames with no annotated relations are skipped.
                    continue

                frame_input = FrameInput(
                    video_id=video.video_id,
                    frame_index=frame_index,
                    mp4_path=video.mp4_path,
                )
                example_id = f'relation_extraction:{video.video_id}:{frame_index}'
                examples.append(
                    BenchmarkExample(
                        example_id=example_id,
                        task_name='relation_extraction',
                        label=f'{video.video_id}_{frame_index}',
                        frames=[frame_input],
                        metadata={
                            'video_id': video.video_id,
                            'frame_index': frame_index,
                            'num_gt_relations': len(gt_rows),
                        },
                        task_data={
                            'object_id_category_lines': [
                                f'  {did}: {obj.category.name}' for did, obj in display_id_to_object.items()
                            ],
                            'ground_truth_relations': gt_rows,
                        },
                    )
                )
        return examples

    def default_system_prompt(self) -> str:
        vocab_lines = _format_relation_vocab_for_prompt(self._relation_vocab)
        return (
            RELATION_EXTRACTION_SYSTEM_PROMPT_BASE
            + '\n\nAllowed relation_type and value strings (values are only valid for their type):\n'
            + vocab_lines
        )

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        frame_input = example.frames[0]
        video = self._get_video(frame_input)
        frame_annotation = video.frame(frame_input.frame_index)
        som_rgb = _set_of_marks_rgb(frame_annotation, frame_input.load_pil_image(), self._som_options)
        som_pil = Image.fromarray(som_rgb, mode='RGB')
        som_buf = BytesIO()
        som_pil.save(som_buf, format='PNG')

        object_id_category_lines = cast(list[str], example.task_data['object_id_category_lines'])

        user_text = (
            'First image: original RGB frame.\n'
            'Second image: set-of-mark overlay with numeric object IDs.\n\n'
            'Object ID to category:\n'
            + '\n'.join(object_id_category_lines)
            + '\n\nList every relation you see in this frame as a JSON array.'
        )
        return [
            TextPart(text='Original frame:'),
            example.frames[0].to_image_part(),
            TextPart(text='Set-of-mark frame:'),
            ImagePart(data=som_buf.getvalue(), mime_type='image/png'),
            TextPart(text=user_text),
        ]

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, object] | None:
        return _parse_relation_extraction_response(raw_text)

    def evaluate(self, example: BenchmarkExample, task_result: dict[str, object]) -> dict[str, object]:
        result = dict(task_result)
        pred_rows = cast(list[dict[str, object]], result.get('predicted_relations', []))
        gt_rows = cast(list[dict[str, object]], example.task_data['ground_truth_relations'])
        pred_tuples = _relations_to_tuples(pred_rows)
        gt_tuples = _relations_to_tuples(gt_rows)
        result.update(_get_relation_scores(pred_tuples, gt_tuples))
        result['ground_truth_relations'] = gt_rows
        return result

    def validate_example(self, example: BenchmarkExample) -> None:
        if not example.frames:
            raise ValueError(f'Example {example.example_id} has no frames')
        gt = example.task_data.get('ground_truth_relations', [])
        if not isinstance(gt, list) or not gt:
            raise ValueError(f'Example {example.example_id} has empty ground_truth_relations')

    def init_metrics(self, *, examples_total: int) -> dict[str, object]:
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
            'micro_tp': 0,
            'micro_pred': 0,
            'micro_gt': 0,
            'micro_precision': None,
            'micro_recall': None,
            'micro_f1': None,
            'sum_example_f1': 0.0,
            'example_f1_count': 0,
            'mean_example_f1': None,
            'per_relation_type': {},
        }

    def update_metrics(self, metrics: dict[str, object], prediction: PredictionResult) -> None:
        if prediction.error is not None:
            metrics['micro_gt'] = int(metrics['micro_gt']) + int(prediction.metadata.get('num_gt_relations', 0))
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

        if prediction.error is not None:
            return

        tr = prediction.task_result
        metrics['micro_tp'] = int(metrics['micro_tp']) + int(tr['tp'])
        metrics['micro_pred'] = int(metrics['micro_pred']) + int(tr['pred_count'])
        metrics['micro_gt'] = int(metrics['micro_gt']) + int(tr['gt_count'])
        metrics['sum_example_f1'] = float(metrics['sum_example_f1']) + float(tr['f1'])
        metrics['example_f1_count'] = int(metrics['example_f1_count']) + 1

        prt = tr.get('per_relation_type')
        if isinstance(prt, dict):
            bucket = cast(dict[str, dict[str, int]], metrics['per_relation_type'])
            for rtype, row in prt.items():
                if not isinstance(row, dict):
                    continue
                sub = bucket.setdefault(rtype, {'tp': 0, 'pred_count': 0, 'gt_count': 0})
                sub['tp'] = int(sub['tp']) + int(row['tp'])
                sub['pred_count'] = int(sub['pred_count']) + int(row['pred_count'])
                sub['gt_count'] = int(sub['gt_count']) + int(row['gt_count'])

    def finalize_metrics(self, metrics: dict[str, object]) -> None:
        latency_samples = int(metrics['latency_samples'])
        if latency_samples > 0:
            metrics['mean_latency_ms'] = float(metrics['total_latency_ms']) / latency_samples

        overall = _relation_scores_from_counts(
            int(metrics['micro_tp']),
            int(metrics['micro_pred']),
            int(metrics['micro_gt']),
        )
        metrics['micro_precision'] = overall['precision']
        metrics['micro_recall'] = overall['recall']
        metrics['micro_f1'] = overall['f1']

        ec = int(metrics['examples_total'])
        if ec > 0:
            metrics['mean_example_f1'] = float(metrics['sum_example_f1']) / ec

        raw_by_type = cast(dict[str, dict[str, int]], metrics.get('per_relation_type') or {})
        metrics['per_relation_type'] = {
            rtype: _relation_scores_from_counts(int(sub['tp']), int(sub['pred_count']), int(sub['gt_count']))
            for rtype, sub in sorted(raw_by_type.items())
        }

    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        def _fmt_lat(value: object) -> str:
            if value is None:
                return 'N/A'
            ms = float(value)
            if ms >= 1000.0:
                return f'{ms / 1000.0:.2f}s'
            return f'{ms:.0f}ms'

        lines = [
            f'Run directory: {run_dir}',
            f'Examples total: {metrics["examples_total"]}',
            f'Skipped existing: {metrics["skipped_existing"]}',
            f'Provider successes: {metrics["provider_successes"]}',
            f'Call failures: {metrics["call_failures"]}',
            f'Parse failures: {metrics["parse_failures"]}',
            f'Parsed predictions: {metrics["parsed_predictions"]}',
            '',
            f'Micro precision: {metrics.get("micro_precision")}',
            f'Micro recall: {metrics.get("micro_recall")}',
            f'Micro F1: {metrics.get("micro_f1")}',
            f'Mean per-example F1: {metrics.get("mean_example_f1")}',
        ]
        prt = metrics.get('per_relation_type')
        if isinstance(prt, dict) and prt:
            lines.append('')
            lines.append('Per relation_type (micro over tuples, type-normalised):')
            for rtype in sorted(prt.keys()):
                row = prt[rtype]
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f'  {rtype}: P={row.get("precision")} R={row.get("recall")} F1={row.get("f1")} '
                    f'(tp={row.get("tp")} pred={row.get("pred_count")} gt={row.get("gt_count")})'
                )
        lines.extend(
            [
                '',
                f'Wall-clock time: {_fmt_lat(metrics.get("wall_clock_ms"))}',
                f'Concurrency: {metrics.get("concurrency", 1)}',
                f'Mean request latency: {_fmt_lat(metrics.get("mean_latency_ms"))}',
                f'Min request latency: {_fmt_lat(metrics.get("min_latency_ms"))}',
                f'Max request latency: {_fmt_lat(metrics.get("max_latency_ms"))}',
            ]
        )
        return '\n'.join(lines)
