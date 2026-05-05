"""Referring expression benchmark task."""

import json
import logging
import random
import re
from pathlib import Path
from typing import cast

from barista.bbox import BboxFormat, format_description, pixel_to_normalized
from barista.dataset import (
    FrameAnnotation,
    ObjectAnnotation,
    Relation,
    Video,
    discover_video_dirs,
)
from barista.vlm_benchmarks.geval import GEvalCriterionSpec
from barista.vlm_benchmarks.tasks.geval_task_base import GEvalTaskBase
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    FrameInput,
    MessagePart,
    RunConfig,
    TextPart,
)

logger = logging.getLogger(__name__)

_VOCABULARY_CATEGORIES = {
    'bottle': 'a glass or plastic container for liquids; typically cylindrical without handles',
    'box': 'a (usually rectangular) container; may have a lid',
    'button': 'a small, round object that you press to activate or control a machine',
    'capsule': 'a small pre-filled pod or capsule of ground coffee for use in coffee machines',
    'carafe': 'a glass container with a narrow neck and wide base, typically used for serving liquids',
    'clamp': 'a device (generally used by carpenters) that holds things firmly together',
    'cleaning brush': 'Brush for cleaning espresso machine portafilter and group head',
    'coffee': 'a seed of the coffee tree; ground to make coffee',
    'coffee cup': 'a cup from which coffee is drunk',
    'coffee machine': 'A coffee machine is a device that brews and dispenses coffee.',
    'coffee mill': 'a mill that grinds roasted coffee beans',
    'container': 'an object for holding or transporting something',
    'door': 'a swinging, sliding or revolving barrier at the entrance of a building, room or machine',
    'dosing ring': 'a metal ring component inside a portafilter used for dosing and distributing ground coffee',
    'drawer': 'a boxlike container in a piece of furniture; made so as to slide in and out',
    'glass': 'a container made of glass for holding drinkable liquids',
    'handle': 'A handle is a part of an object that you can hold onto or use to control it.',
    'jar': 'a cylindrical vessel, typically made out of glass and used for storing food',
    'knob': 'a round handle',
    'left hand': 'The hand located on the left side of the body.',
    'lid': 'A top or covering that closes a container.',
    'milk': 'white liquid from dairy animals commonly added to coffee',
    'milk carton': 'a rectangular, often recyclable container made of paperboard with a plastic spout, used for milk',
    'milk frother': 'A milk frother is a device used to aerate milk, creating a creamy, foamy...',
    'needle distributor': 'a coffee preparation tool (often called a WDT tool) equipped with fine needles for breaking up coffee grounds',
    'parcel': 'a package or a small box that contains something',
    'pitcher': 'an open vessel with a handle and a spout for pouring',
    'portafilter': 'a metal filter basket with a handle, used in espresso machines to hold and tamp ground coffee',
    'puck screen': 'a fine metal mesh placed on top of an espresso puck to promote even water distribution',
    'right hand': 'The right hand is the hand located on the right side of a persons body.',
    'saucer': 'a small shallow dish for holding a cup at the table',
    'scale': 'a device used to measure weight or determine the amount or proportion of something',
    'spoon': 'a piece of cutlery with a shallow bowl-shaped container and a handle; used for stirring or scooping',
    'steam wand': 'a tube, typically made of metal or heat-resistant materials, on a coffee machine for steaming milk',
    'sugar': 'a white crystalline carbohydrate used as a sweetener and preservative',
    'sugar syrup': 'a mixture of sugar and water and sometimes corn syrup boiled together; used as a sweetening agent',
    'switch': 'a device, usually electromechanical, used to open and close an electric circuit or power flow',
    'tamper': 'a tool with a flat or convex base and ergonomic handle, used for evenly tamping ground coffee',
    'tap': 'a faucet for drawing water from a pipe or cask',
    'tissue': 'a soft thin (usually translucent) paper',
    'towel': 'a rectangular piece of absorbent cloth (or paper) for drying or wiping',
    'trash can': 'a bin that holds rubbish until it is collected',
    'vessel': 'A vessel is a container used for holding liquids or other substances.',
    'water': 'a clear liquid essential for life and used in beverage preparation',
    'water tank': 'the removable or integrated reservoir on a coffee machine that holds water for brewing',
}


def _build_vocabulary_for_evaluator() -> str:
    """Format the vocabulary as a canonical-name reference for the GEval judge."""
    lines = [
        'Domain vocabulary (canonical category names; accept reasonable synonyms, '
        'but prefer more specific names over generic ones):',
    ]
    for name, description in _VOCABULARY_CATEGORIES.items():
        if description:
            lines.append(f'  - {name}: {description}')
        else:
            lines.append(f'  - {name}')
    return '\n'.join(lines)


REFERRING_SYSTEM_PROMPT_TEMPLATE = (
    'You describe objects in video frames of an egocentric coffee-making scenario.\n\n'
    'You will be given a bounding box in {coord_desc} format, '
    'with integer coordinates normalised to 0-{scale}.\n\n'
    'Write one natural sentence that:\n'
    '1. Names the object using the most specific category you can determine.\n'
    '2. Includes key visual attributes you can observe.\n'
    '3. States the most important spatial or functional relation to another object '
    'Avoid incidental proximity (near, next to, beside).\n'
    '4. Does not mention background, surfaces, walls, or scenery.\n\n'
    'Start your final response with "Description:" followed by your sentence.'
)

_QUESTION_TEMPLATE = (
    'There is an object at {box}. Describe this object, its visual properties, and how it relates to other objects.'
)

_QUESTION_TEMPLATE_WITH_FORMAT = 'Describe the object inside bounding box {box} '

_DESCRIPTION_PATTERN = re.compile(r'(?im)^\s*description\s*:\s*(.+)$')


def _parse_description(raw_text: str) -> str:
    matches = list(_DESCRIPTION_PATTERN.finditer(raw_text))
    if matches:
        return matches[-1].group(1).strip()
    return raw_text.strip()


_CRITERIA: dict[str, GEvalCriterionSpec] = {
    'correctness': {
        'name': 'Correctness',
        'criteria': (
            'Evaluate whether the predicted referring expression correctly and specifically '
            'identifies the object in an egocentric coffee-making video frame. '
            'The reference is a structured annotation in the format:\n'
            '  Object: <category>\n'
            '  Attributes: <key=value, ...>\n'
            '  Relations: <direct relations of the object>\n'
            '  Related objects: <attributes and relations of related objects — supplementary context>\n\n'
            'The "Relations" line lists all direct relations, but the prediction is a single '
            'sentence that is not expected to mention every one. '
            'The "Related objects" section is supplementary context for disambiguation.\n'
            'A good prediction:\n'
            '(1) Names the object with the correct or acceptably synonymous category — '
            'accept reasonable synonyms (e.g. "espresso machine" for "portafilter machine"), '
            'but penalise significantly less specific names (e.g. "machine" for "capsule machine").\n'
            '(2) States key visual attributes of the main object correctly (colour, state).\n'
            '(3) Includes at least one primary spatial or functional relation to another object '
            '(e.g. "held by a hand", "under a capsule machine", "attached to a portafilter machine"). '
            '(4) Is appropriately concise — a single focused sentence without scene description, '
            'background, bounding-box coordinates, or brand names.\n\n'
            'Extra correct details, equivalent wording, and minor rephrasing should never '
            'reduce the score.\n\n' + _build_vocabulary_for_evaluator()
        ),
        'rubric': [
            {
                'score_range': [0, 3],
                'expected_outcome': (
                    'The prediction names a completely different object category that contradicts '
                    'the reference (e.g. "cup" vs "capsule", "left hand" vs "right hand"), '
                    'or is unintelligible / empty.'
                ),
            },
            {
                'score_range': [4, 7],
                'expected_outcome': (
                    'The prediction identifies the correct object but has notable errors: '
                    'uses an overly generic category (e.g. "coffee machine" instead of "capsule machine"), '
                    'gets a main-object attribute wrong (wrong colour or state), '
                    'states a wrong relation (e.g. wrong related object or wrong spatial verb), '
                    'omits all relations entirely, or is excessively verbose (multiple sentences, '
                    'scene description).'
                ),
            },
            {
                'score_range': [8, 10],
                'expected_outcome': (
                    'The prediction correctly identifies the object with its key attributes and '
                    'at least one correct primary spatial/functional relation in a concise single '
                    'sentence. Equivalent wording, extra correct details, and minor stylistic '
                    'differences are acceptable. Omitting secondary structural relations '
                    '(e.g. buttons, handles as parts) should not reduce the score.'
                ),
            },
        ],
    },
}


def _touches_edge(bbox_xywh: list[float], frame_width: int, frame_height: int, margin: int = 2) -> bool:
    x, y, w, h = bbox_xywh
    return x <= margin or y <= margin or (x + w) >= frame_width - margin or (y + h) >= frame_height - margin


def _candidate_objects(
    frame: FrameAnnotation,
) -> list[tuple[ObjectAnnotation, list[tuple[Relation, ObjectAnnotation, bool]]]]:
    """Return all objects in the frame that have a bbox, after applying quality filters."""
    objects_by_id = frame.objects_by_id()
    frame_area = frame.width * frame.height

    # Pre-decode all masks once (expensive RLE decode happens here, not inside any inner loop).
    mask_px: dict[object, int | None] = {}
    for obj in frame.objects:
        m = obj.mask_array(frame.height, frame.width)
        mask_px[obj.object_id] = int(m.sum()) if m is not None else None

    bbox_objects = [obj for obj in frame.objects if obj.category.name.lower() != 'unknown']

    candidates: list[tuple[ObjectAnnotation, list[tuple[Relation, ObjectAnnotation, bool]]]] = []

    for obj in bbox_objects:
        x, y, w, h = obj.bbox

        # Filter: tiny object that also touches the image edge (likely mostly out of frame).
        # Prefer mask area when available; fall back to bbox area.
        mpx = mask_px[obj.object_id]
        obj_area: float = float(mpx) if mpx is not None else w * h
        if obj_area / frame_area < 0.05 and _touches_edge(obj.bbox, frame.width, frame.height):
            continue

        obj_relations: list[tuple[Relation, ObjectAnnotation, bool]] = []
        for rel in frame.relations:
            if rel.source_object_id == obj.object_id:
                target = objects_by_id.get(rel.target_object_id)
                if target is not None:
                    obj_relations.append((rel, target, True))
            elif rel.target_object_id == obj.object_id:
                source = objects_by_id.get(rel.source_object_id)
                if source is not None:
                    obj_relations.append((rel, source, False))

        if not obj_relations or not obj.attributes:
            continue

        candidates.append((obj, obj_relations))

    return candidates


def _select_deterministic(
    candidates: list[tuple[ObjectAnnotation, list[tuple[Relation, ObjectAnnotation, bool]]]],
    n: int,
    seed: str,
) -> list[tuple[ObjectAnnotation, list[tuple[Relation, ObjectAnnotation, bool]]]]:
    rng = random.Random(seed)
    candidates = list(candidates)
    rng.shuffle(candidates)
    return candidates[:n]


def _build_ground_truth(
    obj: ObjectAnnotation,
    relations: list[tuple[Relation, ObjectAnnotation, bool]],
    frame: FrameAnnotation,
) -> dict[str, object]:
    """Build ground-truth task_data for a referring example."""
    gt_attributes: list[dict[str, str]] = [
        {'attribute_type': attr.attribute_type, 'value': attr.value} for attr in obj.attributes
    ]
    gt_relations: list[dict[str, str]] = []
    for rel, other_obj, is_source in relations:
        gt_relations.append(
            {
                'relation_type': rel.relation_type,
                'value': rel.value,
                'target_category': other_obj.category.name,
                'target_object_id': str(other_obj.object_id),
                'is_source': is_source,
            }
        )

    return {
        'metadata': {
            'category': obj.category.name,
            'bbox_xywh': obj.bbox,
            'object_id': str(obj.object_id),
            'attributes': gt_attributes,
            'relations': gt_relations,
        },
        'structured': _build_structured(obj, relations, frame),
        'question': _QUESTION_TEMPLATE,
    }


def _format_bbox(
    bbox: tuple[float, float, float, float],
    frame_input: 'FrameInput',
    *,
    fmt: BboxFormat,
    scale: int,
) -> str:
    x, y, w, h = bbox
    img = frame_input.load_pil_image()
    w_img, h_img = img.size
    coords = pixel_to_normalized(x, y, x + w, y + h, w_img, h_img, fmt=fmt, scale=scale)
    return f'[{", ".join(str(c) for c in coords)}]'


def _related_context(
    other: ObjectAnnotation,
    main_object_id: str,
    frame: FrameAnnotation,
    objects_by_id: dict[str, ObjectAnnotation],
) -> str | None:
    """Build a parenthetical context string for a related object.

    Shows the object's attributes and its deduplicated direct relations
    (excluding the back-reference to the main object), e.g.
    ``(color=black; extracts coffee, coffee cup under)``
    """
    ctx_parts: list[str] = []

    if other.attributes:
        ctx_parts.append(', '.join(f'{a.attribute_type}={a.value}' for a in other.attributes))

    # Collect and deduplicate relations of the related object.
    raw_rels: list[tuple[str, str, bool]] = []
    for rel in frame.relations:
        if rel.source_object_id == other.object_id:
            target = objects_by_id.get(rel.target_object_id)
            if target and str(target.object_id) != main_object_id:
                raw_rels.append((rel.value, target.category.name, True))
        elif rel.target_object_id == other.object_id:
            source = objects_by_id.get(rel.source_object_id)
            if source and str(source.object_id) != main_object_id:
                raw_rels.append((rel.value, source.category.name, False))

    if raw_rels:
        counts: dict[tuple[str, str, bool], int] = {}
        for key in raw_rels:
            counts[key] = counts.get(key, 0) + 1
        rel_strs: list[str] = []
        for key in dict.fromkeys(raw_rels):
            value, cat, is_source = key
            n = counts[key]
            cat_str = f'{n}x {cat}' if n > 1 else cat
            if is_source:
                rel_strs.append(f'{other.category.name} {value} {cat_str}')
            else:
                rel_strs.append(f'{cat_str} {value} {other.category.name}')
        ctx_parts.append(', '.join(rel_strs))

    if not ctx_parts:
        return None
    return '; '.join(ctx_parts)


def _build_structured(
    obj: ObjectAnnotation,
    relations: list[tuple[Relation, ObjectAnnotation, bool]],
    frame: FrameAnnotation,
) -> str:
    objects_by_id = frame.objects_by_id()
    parts = [f'Object: {obj.category.name}']
    if obj.attributes:
        attrs = ', '.join(f'{a.attribute_type}={a.value}' for a in obj.attributes)
        parts.append(f'Attributes: {attrs}')

    if relations:
        # Compositional relation values (case-insensitive) where the main object
        # is the target are structural ("the machine HAS buttons") rather than
        # referring ("the button IS ON the machine"), so we exclude them.
        _COMPOSITIONAL = {'part of', 'attached to'}

        # Deduplicate relations and build direct relation strings.
        counts: dict[tuple[str, str, bool], int] = {}
        for rel, other, is_source in relations:
            if not is_source and rel.value.lower().strip() in _COMPOSITIONAL:
                continue
            key = (rel.value, other.category.name, is_source)
            counts[key] = counts.get(key, 0) + 1
        seen: set[tuple[str, str, bool]] = set()
        rel_strs: list[str] = []
        # Track unique related objects for the context section.
        related_objects: list[ObjectAnnotation] = []
        related_seen: set[str] = set()
        for rel, other, is_source in relations:
            if not is_source and rel.value.lower().strip() in _COMPOSITIONAL:
                continue
            key = (rel.value, other.category.name, is_source)
            if key in seen:
                continue
            seen.add(key)
            n = counts[key]
            other_name = f'{n}x {other.category.name}' if n > 1 else other.category.name
            if is_source:
                rel_strs.append(f'{obj.category.name} {rel.value} {other_name}')
            else:
                rel_strs.append(f'{other_name} {rel.value} {obj.category.name}')
            if n == 1 and str(other.object_id) not in related_seen:
                related_seen.add(str(other.object_id))
                related_objects.append(other)
        if rel_strs:
            parts.append(f'Relations: {", ".join(rel_strs)}')

        # Build "Related objects" section with attrs + their own relations.
        related_lines: list[str] = []
        for other in related_objects:
            ctx = _related_context(other, str(obj.object_id), frame, objects_by_id)
            if ctx:
                related_lines.append(f'  {other.category.name}: {ctx}')
        if related_lines:
            parts.append('Related objects:\n' + '\n'.join(related_lines))

    return '\n'.join(parts)


class _ReferringBaseTask(GEvalTaskBase):
    """Referring expression evaluation task."""

    _criteria = _CRITERIA
    _prediction_field = 'description'
    _reference_field = 'structured'

    def __init__(
        self,
        dataset_root: Path,
        task_params: dict[str, object],
        *,
        run_config: RunConfig | None,
    ) -> None:
        super().__init__(dataset_root, task_params, run_config=run_config)
        self._bbox_format = BboxFormat(self.task_params.get('bbox_format', 'yxyx'))
        self._bbox_scale = int(self.task_params.get('bbox_scale', 1000))

    def iter_examples(self) -> list[BenchmarkExample]:
        raise NotImplementedError('iter_examples must be called on ReferringBuildTask')

    def default_system_prompt(self) -> str:
        coord_desc = format_description(self._bbox_format)
        return REFERRING_SYSTEM_PROMPT_TEMPLATE.format(coord_desc=coord_desc, scale=self._bbox_scale)

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        if self._run_config is None:
            raise ValueError('run_config is required to render referring prompts')
        frame_input = example.frames[0]
        coord_desc = format_description(self._bbox_format)
        question = _QUESTION_TEMPLATE_WITH_FORMAT.format(box='{box}', coord_desc=coord_desc, scale=self._bbox_scale)

        bbox = example.metadata.get('bbox_xywh')
        if bbox is not None:
            bbox_str = _format_bbox(
                cast(tuple[int | float, int | float, int | float, int | float], bbox),
                frame_input,
                fmt=self._bbox_format,
                scale=self._bbox_scale,
            )
            question = question.format(box=bbox_str)

        return [frame_input.to_image_part(), TextPart(text=question)]

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, object] | None:
        text = raw_text.strip()
        if not text:
            return None
        return {'description': _parse_description(text)}

    def _enrich_eval_result(self, example: BenchmarkExample, result: dict[str, object]) -> None:
        result['reference_structured'] = str(example.task_data.get('structured', ''))

    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        base = super().format_summary(metrics, run_dir=run_dir)

        # Append best/worst per-criterion samples from predictions.
        predictions_path = run_dir / 'predictions.jsonl'
        if not predictions_path.exists():
            return base

        records = []
        for raw_line in predictions_path.read_text(encoding='utf-8').splitlines():
            raw_line = raw_line.strip()
            if raw_line:
                records.append(json.loads(raw_line))

        lines: list[str] = []
        for criterion_key in self._active_criteria:
            spec = _CRITERIA.get(criterion_key)
            label = spec['name'] if spec else criterion_key
            score_key = f'geval_{criterion_key}_score'
            reason_key = f'geval_{criterion_key}_reason'

            scored = [r for r in records if r.get('task_result', {}).get(score_key) is not None]
            if not scored:
                continue

            sorted_asc = sorted(scored, key=lambda r: float(r['task_result'][score_key]))
            cutoff = max(1, len(sorted_asc) // 10)
            bottom_pool = sorted_asc[:cutoff]
            top_pool = sorted_asc[-cutoff:]
            sample_n = min(10, len(top_pool))
            best_sample = random.sample(top_pool, sample_n)
            best_sample.sort(key=lambda r: float(r['task_result'][score_key]), reverse=True)
            worst_sample = random.sample(bottom_pool, min(10, len(bottom_pool)))
            worst_sample.sort(key=lambda r: float(r['task_result'][score_key]))

            lines.append(f'\n── {label} ──')
            lines.append(f'  Top 10% best (n={len(top_pool)}), sample of {len(best_sample)}:')
            for r in best_sample:
                lines.append(f'    ({float(r["task_result"][score_key]):.2f})  [{r["example_id"]}]')
                lines.append(f'      prediction: {r["task_result"].get("description", "")}')
                lines.append(f'      reference:  {r["task_result"].get("reference_structured", "")}')
                lines.append(f'      reason:     {r["task_result"].get(reason_key, "")}')
            lines.append(f'  Bottom 10% worst (n={len(bottom_pool)}), sample of {len(worst_sample)}:')
            for r in worst_sample:
                lines.append(f'    ({float(r["task_result"][score_key]):.2f})  [{r["example_id"]}]')
                lines.append(f'      prediction: {r["task_result"].get("description", "")}')
                lines.append(f'      reference:  {r["task_result"].get("reference_structured", "")}')
                lines.append(f'      reason:     {r["task_result"].get(reason_key, "")}')

        if lines:
            return base + '\n' + '\n'.join(lines)
        return base


class ReferringBuildTask(_ReferringBaseTask):
    def __init__(self, dataset_root: Path, task_params: dict[str, object]) -> None:
        super().__init__(dataset_root, task_params, run_config=None)

    def iter_examples(self) -> list[BenchmarkExample]:
        video_dirs = discover_video_dirs(self.dataset_root)
        video_id_filter = self.task_params.get('video_id')
        if video_id_filter is not None:
            video_dirs = [(d, vid) for d, vid in video_dirs if vid == str(video_id_filter)]

        max_objects_raw = self.task_params.get('max_objects_per_frame')
        max_objects = None if max_objects_raw is None else int(cast(int | str, max_objects_raw))
        max_per_cat_raw = self.task_params.get('max_objects_per_category')
        max_per_category = None if max_per_cat_raw is None else int(cast(int | str, max_per_cat_raw))
        frame_step = int(cast(int | str, self.task_params.get('frame_step', 10)))
        activity_only = bool(self.task_params.get('activity_filter'))

        examples: list[BenchmarkExample] = []
        for video_dir, video_id in video_dirs:
            logger.info('Processing video %s', video_id)
            video = Video.from_dir(video_dir)
            video.video_id = video_id
            frame_indices = video.frame_indices()
            sampled_indices = frame_indices[::frame_step] if frame_step > 1 else frame_indices
            if activity_only:
                activity_set = set(video.activity_frame_indices())
                sampled_indices = [i for i in sampled_indices if i in activity_set]
            for frame_index in sampled_indices:
                frame = video.frame(frame_index)
                candidates = _candidate_objects(frame)
                if not candidates:
                    continue

                if max_objects is not None and len(candidates) > max_objects:
                    candidates = _select_deterministic(
                        candidates,
                        max_objects,
                        f'{video.video_id}:{frame_index}',
                    )

                for obj, obj_relations in candidates:
                    gt = _build_ground_truth(obj, obj_relations, frame)
                    example_id = f'referring:{video.video_id}:frame_{frame_index}:obj_{obj.object_id}'
                    examples.append(
                        BenchmarkExample(
                            example_id=example_id,
                            task_name='referring',
                            label=obj.category.name,
                            frames=[
                                FrameInput(
                                    video_id=video.video_id,
                                    frame_index=frame_index,
                                    mp4_path=video.mp4_path,
                                ),
                            ],
                            metadata={
                                'video_id': video.video_id,
                                'frame_index': frame_index,
                                'object_id': str(obj.object_id),
                                'category_name': obj.category.name,
                                'bbox_xywh': obj.bbox,
                            },
                            task_data=gt,
                        ),
                    )

        if max_per_category is not None:
            shuffle_seed = self.task_params.get('shuffle_seed')
            by_cat: dict[str, list[BenchmarkExample]] = {}
            for ex in examples:
                by_cat.setdefault(ex.label, []).append(ex)
            capped: list[BenchmarkExample] = []
            for cat, cat_examples in by_cat.items():
                if len(cat_examples) > max_per_category:
                    rng = random.Random(f'{shuffle_seed}:{cat}')
                    rng.shuffle(cat_examples)
                    cat_examples = cat_examples[:max_per_category]
                capped.extend(cat_examples)
            logger.info(
                'Category cap %d: %d -> %d examples across %d categories',
                max_per_category,
                len(examples),
                len(capped),
                len(by_cat),
            )
            examples = capped

        return examples


class ReferringRunTask(_ReferringBaseTask):
    def __init__(self, dataset_root: Path, task_params: dict[str, object], *, run_config: RunConfig) -> None:
        super().__init__(dataset_root, task_params, run_config=run_config)
