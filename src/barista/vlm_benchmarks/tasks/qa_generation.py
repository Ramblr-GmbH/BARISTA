"""Multi-frame QA pair generation.

Generates LLM-based, category-tagged QA pairs from multi-frame video clips.
All categories are grounded in annotation diffs between frames.

Categories:

- ``state_change``         – how a machine's state changed (open→closed,
  started extracting, etc.).
- ``action_transition``    – how the person's hand actions progressed
  across the clip.
- ``relation_change``      – how a spatial relationship between objects changed.
- ``activity_progression`` – what activity step transitioned.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from barista.dataset import FrameAnnotation, ObjectAnnotation, Video
from barista.vlm_benchmarks.providers.base import VlmProviderClient
from barista.vlm_benchmarks.types import RunConfig, TextPart

# ── Category definitions ─────────────────────────────────────────────────────

QA_CATEGORIES = [
    'state_change',
    'action_transition',
    'relation_change',
    'activity_progression',
]

# ── Change detection ─────────────────────────────────────────────────────────

_UUID_RE = re.compile(r'\s*\(#[0-9a-f-]+\)')
_HAND_CATEGORIES = frozenset({'left hand', 'right hand'})
_KEYFRAME_NEIGHBORHOOD = 2  # annotated frames checked on either side of each keyframe


# ── Annotation helpers ────────────────────────────────────────────────────────


def _relation_index(frame: FrameAnnotation) -> dict[uuid.UUID, list[tuple[str, str, uuid.UUID]]]:
    """Map each source object to its ``(relation_type, value, target_id)`` tuples."""
    index: dict[uuid.UUID, list[tuple[str, str, uuid.UUID]]] = defaultdict(list)
    for rel in frame.relations:
        index[rel.source_object_id].append((rel.relation_type, rel.value, rel.target_object_id))
    return index


def _position_rels(
    obj_id: uuid.UUID,
    rel_idx: dict[uuid.UUID, list[tuple[str, str, uuid.UUID]]],
    objects_by_id: dict[uuid.UUID, ObjectAnnotation],
) -> set[tuple[str, str]]:
    """Return ``(relation_value, target_category)`` pairs for position relations of *obj_id*."""
    result: set[tuple[str, str]] = set()
    for rel_type, value, target_id in rel_idx.get(obj_id, []):
        if rel_type != 'position':
            continue
        if value == 'part of':
            continue
        tgt = objects_by_id.get(target_id)
        if tgt:
            result.add((value, tgt.category.name))
    return result


def _action_rels(
    obj_id: uuid.UUID,
    rel_idx: dict[uuid.UUID, list[tuple[str, str, uuid.UUID]]],
    objects_by_id: dict[uuid.UUID, ObjectAnnotation],
) -> set[str]:
    """Return contacted target categories for human-action relations of *obj_id*."""
    result: set[str] = set()
    for rel_type, _value, target_id in rel_idx.get(obj_id, []):
        if rel_type == 'human_actions':
            tgt = objects_by_id.get(target_id)
            if tgt:
                result.add(tgt.category.name)
    return result


def _activities_at(video: Video, frame_index: int) -> frozenset[str]:
    """Return activity display names covering *frame_index*, with UUIDs stripped."""
    return frozenset(
        _UUID_RE.sub('', act.display_name.strip())
        for act in video.activities
        if act.frame_start <= frame_index <= act.frame_end and act.display_name.strip()
    )


def _majority_value(values: list[str]) -> str | None:
    """Return the most frequent value in *values*, or ``None`` if empty."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _normalize_state_transition(from_value: str, to_value: str) -> str | None:
    """Map a raw attribute transition to a visual description, or ``None`` if irrelevant."""
    fv, tv = from_value.lower().strip(), to_value.lower().strip()
    if fv == 'open' and tv == 'close':
        return 'closed'
    if fv == 'close' and tv == 'open':
        return 'opened'
    if tv == 'extracting' and fv != 'extracting':
        return 'started extracting'
    if fv == 'extracting' and tv != 'extracting':
        return 'stopped extracting'
    if tv == 'rinsing' and fv != 'rinsing':
        return 'started rinsing'
    if fv == 'rinsing' and tv != 'rinsing':
        return 'stopped rinsing'
    return None


# ── Presence / frame-cache helpers ───────────────────────────────────────────

_FrameCache = tuple[dict[uuid.UUID, ObjectAnnotation], dict[uuid.UUID, list[tuple[str, str, uuid.UUID]]]]


def _presence_count(frames: list[int], video: Video) -> Counter:
    """Count how many frames in *frames* each object appears in."""
    counts: Counter[uuid.UUID] = Counter()
    for fi in frames:
        for obj in video.frames[fi].objects:
            counts[obj.object_id] += 1
    return counts


def _build_frame_caches(frames: list[int], video: Video) -> list[_FrameCache]:
    """Return ``(objects_by_id, relation_index)`` for each frame in *frames*."""
    return [(video.frames[fi].objects_by_id(), _relation_index(video.frames[fi])) for fi in frames]


# ── Hand-action state helpers ─────────────────────────────────────────────────


def _hand_state_at(
    frame: FrameAnnotation,
    hand_ids: list[uuid.UUID],
) -> frozenset[str] | None:
    """Contacted object categories for a hand in *frame*, or ``None`` if the hand is absent."""
    objs_by_id = frame.objects_by_id()
    if not any(oid in objs_by_id for oid in hand_ids):
        return None
    rel_idx = _relation_index(frame)
    targets: set[str] = set()
    for oid in hand_ids:
        if oid in objs_by_id:
            targets |= _action_rels(oid, rel_idx, objs_by_id)
    return frozenset(targets)


def _majority_hand_state(
    center: int,
    video: Video,
    hand_ids: list[uuid.UUID],
) -> frozenset[str] | None:
    """Majority contact-target state in annotated frames within ±_KEYFRAME_NEIGHBORHOOD of *center*."""
    counts: Counter[frozenset[str]] = Counter()
    for fi, frame in video.frames.items():
        if abs(fi - center) <= _KEYFRAME_NEIGHBORHOOD:
            state = _hand_state_at(frame, hand_ids)
            if state is not None:
                counts[state] += 1
    return counts.most_common(1)[0][0] if counts else None


def detect_clip_changes(
    frame_indices: list[int],
    video: Video,
) -> dict[str, list[dict[str, Any]]]:
    """Detect annotation changes across the keyframes of a video clip.

    ``frame_indices`` contains the keyframes that will be shown to the VLM —
    typically 4–8 frames sampled at regular intervals.  All detection is
    grounded exclusively in these frames (and their close annotation neighbors)
    so the generated ground truth reflects only what the model can observe.

    Frame windows
    -------------
    Most detection stages compare a *start window* against an *end window*:

    * **Start window** — annotated frames within ±``_KEYFRAME_NEIGHBORHOOD`` of
      the first keyframe (expanded from ``video.frames``).
      Expanding to annotation neighbors gives noise tolerance: a single bad
      annotation at the clip boundary cannot flip a detection.
    * **End window** — same construction around the last keyframe.
    * A property is considered true in a window if it holds in *at least one*
      of that window's frames.

    Detection stages
    ----------------
    1. **State change** — for objects present in *both* windows, whether the
       majority attribute value (e.g. open/closed) changed between windows.
    2. **Action transition** — the sequence of object categories each hand
       contacts while sweeping the keyframes in order.  A state change is
       recorded only when the *set of contacted objects* changes (the action
       verb — touching, holding, pressing — is irrelevant).  At each keyframe
       the majority contact state across its ±``_KEYFRAME_NEIGHBORHOOD``
       annotation neighbors is used for noise tolerance.
    3. **Relation change** — for objects present in both windows, whether their
       dominant spatial relation to other objects changed (majority threshold ≥
       half the frames the object appears in per window).
    4. **Activity progression** — transitions between annotation activity labels
       while sweeping keyframes in order.

    Returns ``{category: [change_dict, ...]}``.  Only categories with at least
    one detected change are included.
    """
    if len(frame_indices) < 2:
        return {}

    # Build a category lookup across all keyframes (used by every stage below).
    obj_cat: dict[uuid.UUID, str] = {}
    for fi in frame_indices:
        for obj in video.frames[fi].objects:
            obj_cat.setdefault(obj.object_id, obj.category.name)

    changes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # ── Start / end windows ───────────────────────────────────────────────
    # Each boundary keyframe is expanded to all annotated frames within
    # ±_KEYFRAME_NEIGHBORHOOD in video.frames (no separate dense list needed).
    def _boundary_window(kfs: list[int]) -> list[int]:
        return sorted({fi for kf in kfs for fi in video.frames if abs(fi - kf) <= _KEYFRAME_NEIGHBORHOOD})

    start_window = _boundary_window(frame_indices[:1])  # frames near first keyframe
    end_window = _boundary_window(frame_indices[-1:])  # frames near last keyframe

    start_presence = _presence_count(start_window, video)
    end_presence = _presence_count(end_window, video)
    start_caches = _build_frame_caches(start_window, video)
    end_caches = _build_frame_caches(end_window, video)

    # Objects that appear in at least one frame of both windows —
    # used by stages 1 and 3 (state and relation change).
    persistent_ids = {
        oid for oid in set(start_presence) & set(end_presence) if start_presence[oid] >= 2 and end_presence[oid] >= 2
    }

    # ── 1. State change ───────────────────────────────────────────────────
    # For each object persistent across the clip, collect attribute values
    # from both windows and compare the majority value per attribute type.
    seen_state: set[tuple[str, str]] = set()
    for obj_id in persistent_ids:
        cat = obj_cat.get(obj_id, 'unknown')
        if cat == 'unknown':
            continue

        start_attrs: dict[str, list[str]] = defaultdict(list)
        for objs, _ in start_caches:
            if obj := objs.get(obj_id):
                for a in obj.attributes:
                    start_attrs[a.attribute_type].append(a.value)

        end_attrs: dict[str, list[str]] = defaultdict(list)
        for objs, _ in end_caches:
            if obj := objs.get(obj_id):
                for a in obj.attributes:
                    end_attrs[a.attribute_type].append(a.value)

        for attr_type in set(start_attrs) | set(end_attrs):
            v_start = _majority_value(start_attrs.get(attr_type, []))
            v_end = _majority_value(end_attrs.get(attr_type, []))
            if v_start is None or v_end is None or v_start == v_end:
                continue
            desc = _normalize_state_transition(v_start, v_end)
            if desc and (cat, desc) not in seen_state:
                seen_state.add((cat, desc))
                changes['state_change'].append({'object': cat, 'description': desc})

    # ── 2. Action transition ──────────────────────────────────────────────
    # Sweep each keyframe in order.  At each keyframe, determine the majority
    # contact-target state (set of objects being touched) using annotation
    # neighbors within ±_KEYFRAME_NEIGHBORHOOD.  Record a timeline entry only
    # when the set of contacted objects changes.
    hand_ids_by_cat: dict[str, list[uuid.UUID]] = defaultdict(list)
    for oid, cat in obj_cat.items():
        if cat in _HAND_CATEGORIES:
            hand_ids_by_cat[cat].append(oid)

    for hand_cat in sorted(hand_ids_by_cat):
        hand_ids = hand_ids_by_cat[hand_cat]
        timeline: list[frozenset[str]] = []
        prev_committed: frozenset[str] = frozenset()
        current_state: frozenset[str] = frozenset()

        for fi in frame_indices:
            state = _majority_hand_state(fi, video, hand_ids)
            if state is None:
                continue  # hand not annotated at this keyframe
            if state != current_state:
                if current_state != prev_committed:
                    timeline.append(current_state)
                    prev_committed = current_state
                current_state = state

        # Flush the last state.
        if current_state != prev_committed:
            timeline.append(current_state)

        if len(timeline) >= 2:
            changes['action_transition'].append(
                {
                    'hand': hand_cat,
                    'action_timeline': [[{'target': t} for t in sorted(step)] if step else [] for step in timeline],
                }
            )

    # ── 3. Relation change ────────────────────────────────────────────────
    # For each persistent object, collect position relations in each window.
    # A relation is considered "stable" in a window if it appears in at least
    # half the frames that object is annotated in.
    seen_relation: set[tuple[str, frozenset, frozenset]] = set()
    for obj_id in persistent_ids:
        cat = obj_cat.get(obj_id, 'unknown')
        if cat == 'unknown':
            continue

        start_rel_counts: Counter[tuple[str, str]] = Counter()
        start_obj_frames = 0
        for objs, rel_idx in start_caches:
            if obj_id in objs:
                start_obj_frames += 1
                start_rel_counts.update(_position_rels(obj_id, rel_idx, objs))

        end_rel_counts: Counter[tuple[str, str]] = Counter()
        end_obj_frames = 0
        for objs, rel_idx in end_caches:
            if obj_id in objs:
                end_obj_frames += 1
                end_rel_counts.update(_position_rels(obj_id, rel_idx, objs))

        # Keep only relations present in at least half the window frames.
        start_rels = {p for p, c in start_rel_counts.items() if start_obj_frames > 0 and c * 2 >= start_obj_frames}
        end_rels = {p for p, c in end_rel_counts.items() if end_obj_frames > 0 and c * 2 >= end_obj_frames}

        new_rels = end_rels - start_rels
        lost_rels = start_rels - end_rels
        if new_rels or lost_rels:
            key = (cat, frozenset(new_rels), frozenset(lost_rels))
            if key not in seen_relation:
                seen_relation.add(key)
                changes['relation_change'].append(
                    {
                        'object': cat,
                        'new_relations': [{'relation': r, 'target': t} for r, t in new_rels],
                        'lost_relations': [{'relation': r, 'target': t} for r, t in lost_rels],
                    }
                )

    # ── 4. Activity progression ───────────────────────────────────────────
    # Sweep keyframes in order, comparing the activity label set at each step.
    # Transitions are the activities that started or ended between consecutive
    # keyframes.
    prev_acts = _activities_at(video, frame_indices[0])
    seen_transitions: set[tuple[frozenset[str], frozenset[str]]] = set()

    for fi in frame_indices[1:]:
        curr_acts = _activities_at(video, fi)
        if curr_acts == prev_acts:
            continue
        started = curr_acts - prev_acts
        ended = prev_acts - curr_acts
        if started or ended:
            key = (frozenset(started), frozenset(ended))
            if key not in seen_transitions:
                seen_transitions.add(key)
                changes['activity_progression'].append(
                    {
                        'started_activities': sorted(started),
                        'ended_activities': sorted(ended),
                        'continuing_activities': sorted(curr_acts & prev_acts),
                    }
                )
        prev_acts = curr_acts

    return dict(changes)


# ── QA quality filtering ─────────────────────────────────────────────────────

_MIN_ANSWER_WORDS = 16


def _is_weak_qa(answer: str) -> bool:
    return len(answer.split()) < _MIN_ANSWER_WORDS


# ── Prompt construction ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are creating category-tagged question-answer pairs for a visual QA \
benchmark about coffee-making videos captured from an egocentric viewpoint.

You receive:
1. Full clip context (objects, attributes, relations across all timesteps).
2. A summary of detected changes. Multiple changes of the same type may \
be listed — your question and answer should cover all of them together.

Generate exactly 1 question-answer pair for the assigned category. \
The question should naturally encompass all listed changes of that type.

Categories:
- state_change: How did object attributes (open/closed, full/empty, \
on/off) change across the clip?
- action_transition: Ask about the sequence of objects the hand touched, \
in the order they were contacted. Do not anchor the question to "the \
start" or "the end" of the clip — the hand may appear at any point. \
Ask instead: "What objects did the hand contact, in order?" or \
"What did the hand touch and then release?" \
For the answer: describe the sequence of contacts in order \
(e.g. "touched the portafilter, then the grinder, then released"). \
If the hand was idle between contacts, that does not need to be stated \
explicitly — only list the contact targets in sequence. \
A contact state is defined solely by which objects the hand is touching \
— the specific action type (touching, holding, pressing) is irrelevant. \
Do not count reaching, hovering, or gestures without contact.
- relation_change: How did spatial relationships between objects change? \
(e.g., capsule moved inside machine.)
- activity_progression: What coffee-making activity started and/or ended \
during the clip? The question must explicitly name the activities that \
ended and those that started, e.g. "What activity ended and what started \
during the clip?" or "Describe the activity transition visible in the clip." \
The answer must state: (1) what was happening at the start of the clip, \
(2) what stopped (if any), and (3) what began (if any), using natural \
visual language. Do not omit any of the listed transitions.

Rules for questions:
  - Sound natural, as if asked by someone watching the video.
  - Ask about specific, observable content — not metadata or annotations.
  - Use coffee-making vocabulary naturally.
  - For state_change, relation_change, activity_progression: reference \
"the start" and "end" of the clip — never frame numbers.
  - For action_transition: ask about the sequence of contacts — do not \
say "at the start" or "by the end".

Rules for answers:
  - 1-3 sentences, max 50 words. Concrete, visible content.
  - Mention ALL relevant changes — do not omit any listed change.
  - For state_change, relation_change, activity_progression: use natural \
language ("at the start", "by the end") — never frame numbers.
  - For action_transition: describe the sequence of contact targets in \
order — no temporal anchoring required.

Return ONLY a valid JSON object:
{{"question": "...", "answer": "...", "category": "..."}}
"""


# ── Generation ───────────────────────────────────────────────────────────────


def _parse_llm_json(raw: str) -> dict | None:
    """Parse LLM response as a single JSON object, handling fences."""
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f'    [warn] JSON parse error ({exc}); raw={repr(raw[:200])}; skipping')
        return None

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return None


# ── Change summary formatter ─────────────────────────────────────────────────


def _format_change_summary(changes: dict[str, list[dict]]) -> str:
    """Format detected changes into a human-readable string for the LLM prompt."""
    if not changes:
        return 'No significant changes detected.'

    lines: list[str] = []

    for sc in changes.get('state_change', []):
        lines.append(f'- state_change: "{sc["object"]}" {sc["description"]}')

    for at in changes.get('action_transition', []):
        steps = []
        for step_targets in at['action_timeline']:
            if step_targets:
                step_str = ' & '.join(t['target'] for t in step_targets)
            else:
                step_str = 'idle'
            steps.append(step_str)
        lines.append(f'- action_transition: "{at["hand"]}" contacts: {" -> ".join(steps)}')

    for rc in changes.get('relation_change', []):
        new_r = ', '.join(f'{r["relation"]} {r["target"]}' for r in rc['new_relations'])
        lost_r = ', '.join(f'{r["relation"]} {r["target"]}' for r in rc['lost_relations'])
        parts = []
        if new_r:
            parts.append(f'new: {new_r}')
        if lost_r:
            parts.append(f'lost: {lost_r}')
        lines.append(f'- relation_change: "{rc["object"]}" — {"; ".join(parts)}')

    for ap in changes.get('activity_progression', []):
        parts = []
        if ap.get('ended_activities'):
            parts.append('ended: ' + ', '.join(ap['ended_activities']))
        if ap.get('started_activities'):
            parts.append('started: ' + ', '.join(ap['started_activities']))
        if ap.get('continuing_activities'):
            parts.append('continuing: ' + ', '.join(ap['continuing_activities']))
        lines.append(f'- activity_progression: {"; ".join(parts)}')

    return '\n'.join(lines) if lines else 'No significant changes detected.'


def _build_compact_clip_context(
    frame_indices: list[int], video: Video, changes: dict[str, list[dict[str, Any]]]
) -> str:
    """Build a compact context string for the LLM in change-focused QA generation.

    Instead of repeating full frame-by-frame annotation detail, this emits:
    - A list of all object categories present across the clip (deduplicated).
    - The activity progression timeline.
    - The structured change summary (already computed).
    """
    all_obj_cats: set[str] = set()
    for fi in frame_indices:
        for obj in video.frames[fi].objects:
            if obj.category.name != 'unknown':
                all_obj_cats.add(obj.category.name)

    # Activity timeline (overlap-aware, deduplicated transitions).
    activity_steps: list[str] = []
    prev_act_set: frozenset[str] = frozenset()
    for fi in frame_indices:
        act_set = _activities_at(video, fi)
        if act_set != prev_act_set and act_set:
            label = ' + '.join(sorted(act_set))
            if not activity_steps or activity_steps[-1] != label:
                activity_steps.append(label)
            prev_act_set = act_set

    parts: list[str] = []
    if all_obj_cats:
        parts.append('Objects present: ' + ', '.join(sorted(all_obj_cats)))
    if activity_steps:
        parts.append('Activity timeline: ' + ' -> '.join(activity_steps))
    parts.append('Detected changes:\n' + _format_change_summary(changes))
    return '\n\n'.join(parts)


# ── Video processing ──────────────────────────────────────────────────────────


def find_all_change_clips(
    video: Video,
    clip_length_in_frames: int,
    frame_spacing: int,
) -> list[tuple[list[int], dict[str, list]]]:
    """Find all non-overlapping clips that contain at least one change.

    Uses a stride equal to the window length (``clip_length_in_frames *
    frame_spacing``) so each annotated-frame position belongs to exactly one
    clip.
    """
    indices = video.frame_indices()
    step = max(1, clip_length_in_frames * frame_spacing)
    results = []
    for start in range(0, len(indices), step):
        clip = [
            indices[start + j * frame_spacing]
            for j in range(clip_length_in_frames)
            if start + j * frame_spacing < len(indices)
        ]
        if len(clip) < 2:
            continue
        changes = detect_clip_changes(clip, video)
        if changes:
            results.append((clip, changes))
    return results


def scan_video_changes(
    video_dir: Path,
    clip_length_in_frames: int,
    frame_spacing: int,
) -> list[tuple[list[int], dict[str, list]]] | None:
    """Load a video and return all change clips without LLM generation.

    Returns ``None`` if the video cannot be loaded.
    """
    try:
        video = Video.from_dir(video_dir)
    except Exception as exc:
        print(f'  [skip] cannot load video from {video_dir}: {exc}')
        return None
    clips = find_all_change_clips(video, clip_length_in_frames, frame_spacing)
    print(f'  {video_dir.name}: {len(video.frames)} annotated frames -> {len(clips)} clips with changes')
    return clips


def generate_qa_for_selected_clips(
    video_dir: Path,
    selected_by_category: dict[tuple[int, ...], set[str]],
    client: VlmProviderClient,
    config: RunConfig,
    output_path: Path,
) -> None:
    """Generate QA for a pre-selected set of clips and write to *output_path*.

    *selected_by_category* maps ``clip_frame_tuple`` to the set of categories
    to generate for that clip.  The caller is responsible for applying any
    global cap and shuffle before passing this dict.
    """
    try:
        video = Video.from_dir(video_dir)
    except Exception as exc:
        print(f'  [skip] cannot load video from {video_dir}: {exc}')
        return

    results = []
    for clip_tuple in sorted(selected_by_category.keys()):
        clip = list(clip_tuple)
        categories = selected_by_category[clip_tuple]
        changes = detect_clip_changes(clip, video)

        pairs: list[dict[str, Any]] = []
        for category in sorted(categories):
            if category not in changes:
                continue
            category_changes = {category: changes[category]}
            context = _build_compact_clip_context(clip, video, category_changes)
            llm_system = _SYSTEM_PROMPT + f'\n\nIMPORTANT: Only generate a pair for this category: {category}.'
            try:
                response = client.generate(llm_system, [TextPart(text=context)], config)
                raw = response.raw_text
            except Exception as exc:
                print(f'    [warn] generation failed for {category} ({exc}); skipping')
                continue
            item = _parse_llm_json(raw)
            if not item:
                continue
            q = item.get('question', '')
            a = item.get('answer', '')
            cat = item.get('category', '')
            if q and a and cat in QA_CATEGORIES and not _is_weak_qa(a):
                pairs.append({'question': str(q), 'answer': str(a), 'category': str(cat)})

        if pairs:
            idx_str = ','.join(str(fi) for fi in clip)
            cats = {p['category'] for p in pairs}
            print(f'  clip [frames {idx_str}]: {len(pairs)} pair(s) categories={cats}')
            results.append({'clip_index': clip[0], 'frame_indices': clip, 'qa_pairs': pairs})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f'  Saved {len(results)} clip records -> {output_path}')
