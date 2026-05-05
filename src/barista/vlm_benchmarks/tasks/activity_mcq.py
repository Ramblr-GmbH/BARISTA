from __future__ import annotations

import random
import re
from bisect import bisect_left, bisect_right
from pathlib import Path

from barista.dataset import Activity, Video, load_videos
from barista.vlm_benchmarks.tasks.base import BenchmarkTask
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    FrameInput,
    MessagePart,
    PredictionResult,
    TextPart,
    render_frame_image_parts,
)

MCQ_SYSTEM_PROMPT = (
    'You are an expert at fine-grained activity recognition from video frames. '
    'Focus on hand-object interactions and visible state changes across consecutive frames '
    '(e.g., direction of motion, whether an object is being placed or removed). '
    'You may provide a brief rationale, but the final line must be exactly ANSWER: <CHOICE_ID>.'
)

_MCQ_ANSWER_PREFIX = re.compile(r'(?i)\b(?:final\s+)?answer\s*:\s*')
_MCQ_CHOICE_ID_TAG = re.compile(r'<CHOICE_ID[_:\s]+([A-Za-z0-9_-]+)\s*>', re.IGNORECASE)
_MCQ_STRIP_WRAPPERS = re.compile(
    r'^(?:answer\s*:\s*)?'  # repeated "answer:" (e.g. "Final Answer: ANSWER: X")
    r'(?:<CHOICE_ID>\s*)?'  # literal <CHOICE_ID> placeholder
    r'<?(?:CHOICE_ID[_:\s]*|CHOICE_)?',  # CHOICE_ID or CHOICE_ prefix with optional <
    re.IGNORECASE,
)
_MCQ_TOKEN = re.compile(r'([A-Za-z0-9_-]+)')


def _fmt_latency(value: object) -> str:
    if value is None:
        return 'N/A'
    ms = float(value)
    if ms >= 1000.0:
        return f'{ms / 1000.0:.2f}s'
    return f'{ms:.0f}ms'


def _render_mcq_user_prompt(example: BenchmarkExample) -> str:
    choices = example.task_data['choices']
    question = str(example.task_data['question'])
    choice_lines = [f'{c["choice_id"]}. {c["text"]}' for c in choices]

    sections = [
        'Question:',
        question,
        '',
        'Choices:',
        *choice_lines,
    ]
    return '\n'.join(sections)


def _collect_activity_vocab(videos: list[Video]) -> list[str]:
    return sorted({str(activity.activity_class_id) for video in videos for activity in video.activities})


def _cap_per_class(
    examples: list[BenchmarkExample], max_per_class: int, seed: int | None = None
) -> list[BenchmarkExample]:
    by_class: dict[str, dict[str, list[BenchmarkExample]]] = {}
    for ex in examples:
        class_id = str(ex.metadata['activity_class_id'])
        video_id = str(ex.metadata['video_id'])
        by_class.setdefault(class_id, {}).setdefault(video_id, []).append(ex)

    rng = random.Random(seed)
    kept_ids: set[str] = set()
    for class_id, videos_map in by_class.items():
        video_queues = [list(exs) for exs in videos_map.values()]
        rng.shuffle(video_queues)
        count = 0
        while count < max_per_class:
            added_any = False
            for queue in video_queues:
                if count >= max_per_class:
                    break
                if queue:
                    kept_ids.add(queue.pop(0).example_id)
                    count += 1
                    added_any = True
            if not added_any:
                break

    return [ex for ex in examples if ex.example_id in kept_ids]


def _collect_activity_verb_map(videos: list[Video]) -> dict[str, str]:
    result: dict[str, str] = {}
    for video in videos:
        for activity in video.activities:
            class_id = str(activity.activity_class_id)
            if class_id not in result:
                result[class_id] = activity.verb
    return result


def _filter_non_overlapping(activities: list[Activity]) -> list[tuple[int, Activity]]:
    overlapping: set[int] = set()
    sorted_indices = sorted(range(len(activities)), key=lambda i: activities[i].frame_start)
    for pos in range(len(sorted_indices) - 1):
        i = sorted_indices[pos]
        for next_pos in range(pos + 1, len(sorted_indices)):
            j = sorted_indices[next_pos]
            if activities[j].frame_start >= activities[i].frame_end:
                break
            overlapping.add(i)
            overlapping.add(j)
    return [(idx, activities[idx]) for idx in range(len(activities)) if idx not in overlapping]


def _collect_activity_display_names(videos: list[Video]) -> dict[str, str]:
    result: dict[str, str] = {}
    for video in videos:
        for activity in video.activities:
            class_id = str(activity.activity_class_id)
            if class_id not in result and activity.display_name:
                result[class_id] = activity.display_name
    return result


def _segment_frame_indices(video_frame_indices: list[int], activity: Activity) -> list[int]:
    left = bisect_left(video_frame_indices, activity.frame_start)
    right = bisect_right(video_frame_indices, activity.frame_end)
    return video_frame_indices[left:right]


def _sample_uniform_segment_frames(segment_frame_indices: list[int], frames_per_example: int) -> list[int]:
    if frames_per_example == 1:
        return [segment_frame_indices[len(segment_frame_indices) // 2]]

    last_index = len(segment_frame_indices) - 1
    positions = [sample_idx * last_index // (frames_per_example - 1) for sample_idx in range(frames_per_example)]
    return [segment_frame_indices[position] for position in positions]


def _build_choice_activity_class_ids(
    *,
    correct_activity_class_id: str,
    activity_vocab: list[str],
    num_choices: int | None,
    seed: str,
    verb_map: dict[str, str] | None = None,
    min_same_verb_distractors: int = 0,
) -> list[str]:
    rng = random.Random(seed)
    correct_verb = (verb_map or {}).get(correct_activity_class_id)

    same_verb_distractors = []
    other_distractors = []
    for aid in activity_vocab:
        if aid == correct_activity_class_id:
            continue
        if correct_verb and verb_map and verb_map.get(aid) == correct_verb:
            same_verb_distractors.append(aid)
        else:
            other_distractors.append(aid)

    rng.shuffle(same_verb_distractors)
    rng.shuffle(other_distractors)

    if num_choices is None:
        selected_distractors = same_verb_distractors + other_distractors
    else:
        budget = max(num_choices - 1, 0)
        same_take = min(min_same_verb_distractors, len(same_verb_distractors), budget)
        selected_same = same_verb_distractors[:same_take]
        remaining_budget = budget - same_take
        remaining_pool = same_verb_distractors[same_take:] + other_distractors
        rng.shuffle(remaining_pool)
        selected_distractors = selected_same + remaining_pool[:remaining_budget]

    candidate_ids = selected_distractors + [correct_activity_class_id]
    if len(candidate_ids) == 1:
        return candidate_ids

    rng.shuffle(candidate_ids)
    return candidate_ids


def _choice_texts(choice_activity_class_ids: list[str], activity_label_map: dict[str, str]) -> list[str]:
    base_texts = [
        activity_label_map.get(activity_class_id, activity_class_id) for activity_class_id in choice_activity_class_ids
    ]
    counts: dict[str, int] = {}
    for text in base_texts:
        counts[text] = counts.get(text, 0) + 1

    rendered_texts: list[str] = []
    for activity_class_id, text in zip(choice_activity_class_ids, base_texts):
        if counts[text] > 1:
            rendered_texts.append(f'{text} ({activity_class_id})')
            continue
        rendered_texts.append(text)
    return rendered_texts


def _choice_id(index: int) -> str:
    label = ''
    current = index
    while True:
        current, remainder = divmod(current, 26)
        label = chr(ord('A') + remainder) + label
        if current == 0:
            return label
        current -= 1


def _parse_mcq_answer(raw_text: str, valid_choice_ids: list[str]) -> str | None:
    canonical = {cid.upper(): cid for cid in valid_choice_ids}

    # Search backwards through all "answer:" occurrences, return first valid match.
    for match in reversed(list(_MCQ_ANSWER_PREFIX.finditer(raw_text))):
        tail = raw_text[match.end() :]
        tail = _MCQ_STRIP_WRAPPERS.sub('', tail).strip()
        token_match = _MCQ_TOKEN.match(tail)
        if token_match:
            candidate = token_match.group(1).upper()
            if candidate in canonical:
                return canonical[candidate]

    # Fallback: <CHOICE_ID_X> or <CHOICE_ID: X> tag anywhere in the text.
    for match in reversed(list(_MCQ_CHOICE_ID_TAG.finditer(raw_text))):
        candidate = match.group(1).upper()
        if candidate in canonical:
            return canonical[candidate]

    return None


class ActivityMcqTask(BenchmarkTask):
    """Multiple-choice activity recognition over multi-frame segments.

    Supported ``task_params`` keys:

    - ``frames_per_example`` (int, default 4): number of uniformly-sampled
      frames per segment.
    - ``num_choices`` (int | null, default 4): number of answer choices.
      ``null`` uses the full activity vocabulary.
    - ``video_id`` (str | null): optional filter to a single video.
    - ``max_per_class`` (int | null, default null): cap examples per
      activity class. ``null`` means no cap.
    - ``min_same_verb_distractors`` (int, default 0): minimum number of
      distractor choices sharing the same verb as the correct activity.
    """

    def __init__(self, dataset_root: Path, task_params: dict[str, object]) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)

    def iter_examples(self) -> list[BenchmarkExample]:
        videos = load_videos(self.dataset_root)
        video_id_filter = self.task_params.get('video_id')
        if video_id_filter is not None:
            videos = [video for video in videos if video.video_id == str(video_id_filter)]

        activity_vocab = _collect_activity_vocab(videos)
        if not activity_vocab:
            return []

        frames_per_example = int(self.task_params.get('frames_per_example', 4))
        if frames_per_example <= 0:
            raise ValueError('frames_per_example must be >= 1')

        num_choices_value = self.task_params.get('num_choices', 4)
        num_choices = None if num_choices_value is None else int(num_choices_value)
        if num_choices is not None and num_choices <= 0:
            raise ValueError('num_choices must be >= 1 or null')

        activity_label_map = _collect_activity_display_names(videos)
        verb_map = _collect_activity_verb_map(videos)

        max_per_class_value = self.task_params.get('max_per_class')
        max_per_class = None if max_per_class_value is None else int(max_per_class_value)
        shuffle_seed_value = self.task_params.get('shuffle_seed')
        shuffle_seed = None if shuffle_seed_value is None else int(shuffle_seed_value)
        min_same_verb_distractors = int(self.task_params.get('min_same_verb_distractors', 0))

        examples: list[BenchmarkExample] = []
        for video in videos:
            video_frame_indices = video.frame_indices()
            non_overlapping = _filter_non_overlapping(video.activities)
            for activity_idx, activity in non_overlapping:
                segment_frame_indices = _segment_frame_indices(video_frame_indices, activity)
                if len(segment_frame_indices) < frames_per_example:
                    continue

                selected_frame_indices = _sample_uniform_segment_frames(segment_frame_indices, frames_per_example)
                correct_activity_class_id = str(activity.activity_class_id)

                example_seed = f'{video.video_id}:{activity.frame_start}:{activity.frame_end}'
                if shuffle_seed is not None:
                    example_seed = f'{shuffle_seed}:{example_seed}'

                choice_activity_class_ids = _build_choice_activity_class_ids(
                    correct_activity_class_id=correct_activity_class_id,
                    activity_vocab=activity_vocab,
                    num_choices=num_choices,
                    seed=example_seed,
                    verb_map=verb_map,
                    min_same_verb_distractors=min_same_verb_distractors,
                )
                if len(choice_activity_class_ids) < 4:
                    continue

                choice_texts = _choice_texts(choice_activity_class_ids, activity_label_map)
                choices = [
                    {'choice_id': _choice_id(i), 'text': choice_text} for i, choice_text in enumerate(choice_texts)
                ]
                correct_choice_id = choices[choice_activity_class_ids.index(correct_activity_class_id)]['choice_id']
                activity_label = activity_label_map.get(correct_activity_class_id, correct_activity_class_id)

                frame_inputs = [
                    FrameInput(
                        video_id=video.video_id,
                        frame_index=frame_index,
                        mp4_path=video.mp4_path,
                    )
                    for frame_index in selected_frame_indices
                ]

                example_id = (
                    f'activity_mcq:{video.video_id}:segment_{activity.frame_start}_{activity.frame_end}:'
                    f'class_{correct_activity_class_id}:idx_{activity_idx}'
                )
                question = (
                    f'The {frames_per_example} frames below are sampled uniformly from an activity segment '
                    f'(Frame 1 earliest, Frame {frames_per_example} latest). '
                    'Which activity label best describes the segment?'
                )

                examples.append(
                    BenchmarkExample(
                        example_id=example_id,
                        task_name='activity_mcq',
                        label=activity_label,
                        frames=frame_inputs,
                        metadata={
                            'video_id': video.video_id,
                            'segment_frame_start': activity.frame_start,
                            'segment_frame_end': activity.frame_end,
                            'activity_index': activity_idx,
                            'segment_num_frames': len(segment_frame_indices),
                            'selected_frame_indices': selected_frame_indices,
                            'activity_class_id': correct_activity_class_id,
                        },
                        task_data={
                            'question': question,
                            'choices': choices,
                            'correct_choice_id': correct_choice_id,
                        },
                    )
                )

        if max_per_class is not None:
            examples = _cap_per_class(examples, max_per_class, seed=shuffle_seed)

        return examples

    def default_system_prompt(self) -> str:
        return MCQ_SYSTEM_PROMPT

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        image_parts = render_frame_image_parts(example.frames)
        parts: list[MessagePart] = []
        for i, image_part in enumerate(image_parts, 1):
            parts.append(TextPart(text=f'Frame {i}:'))
            parts.append(image_part)
        parts.append(TextPart(text=_render_mcq_user_prompt(example)))
        return parts

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, object] | None:
        choices = example.task_data['choices']
        valid_ids = [str(c['choice_id']) for c in choices]
        parsed_choice_id = _parse_mcq_answer(raw_text, valid_ids)
        if parsed_choice_id is None:
            return None
        return {'parsed_choice_id': parsed_choice_id}

    def evaluate(self, example: BenchmarkExample, task_result: dict[str, object]) -> dict[str, object]:
        result = dict(task_result)
        correct_choice_id = str(example.task_data['correct_choice_id'])
        parsed_choice_id = result.get('parsed_choice_id')
        result['is_correct'] = (parsed_choice_id == correct_choice_id) if parsed_choice_id is not None else None
        result['correct_choice_id'] = correct_choice_id
        result['choices'] = example.task_data['choices']
        return result

    def validate_example(self, example: BenchmarkExample) -> None:
        choices = example.task_data.get('choices', [])
        correct_id = example.task_data.get('correct_choice_id')
        valid_ids = [str(c['choice_id']) for c in choices]
        if correct_id not in valid_ids:
            raise ValueError(
                f'Invalid correct_choice_id for example {example.example_id}: {correct_id!r} not in {valid_ids!r}'
            )

    def init_metrics(self, *, examples_total: int) -> dict[str, object]:
        return {
            'examples_total': examples_total,
            'skipped_existing': 0,
            'provider_successes': 0,
            'call_failures': 0,
            'parse_failures': 0,
            'parsed_predictions': 0,
            'correct_predictions': 0,
            'accuracy': None,
            'accuracy_on_parsed': None,
            'latency_samples': 0,
            'total_latency_ms': 0.0,
            'mean_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'per_label': {},
        }

    def update_metrics(self, metrics: dict[str, object], prediction: PredictionResult) -> None:
        if prediction.error is not None:
            if prediction.error.startswith('provider_error:'):
                metrics['call_failures'] = int(metrics['call_failures']) + 1
            else:
                metrics['provider_successes'] = int(metrics['provider_successes']) + 1
                metrics['parse_failures'] = int(metrics['parse_failures']) + 1
        else:
            metrics['provider_successes'] = int(metrics['provider_successes']) + 1

        parsed_choice_id = prediction.task_result.get('parsed_choice_id')
        is_correct = prediction.task_result.get('is_correct')

        if parsed_choice_id is not None:
            metrics['parsed_predictions'] = int(metrics['parsed_predictions']) + 1
            if is_correct:
                metrics['correct_predictions'] = int(metrics['correct_predictions']) + 1

        if prediction.latency_ms is not None:
            latency = prediction.latency_ms
            metrics['latency_samples'] = int(metrics['latency_samples']) + 1
            metrics['total_latency_ms'] = float(metrics['total_latency_ms']) + latency
            current_min = metrics.get('min_latency_ms')
            current_max = metrics.get('max_latency_ms')
            metrics['min_latency_ms'] = latency if current_min is None else min(float(current_min), latency)
            metrics['max_latency_ms'] = latency if current_max is None else max(float(current_max), latency)

        if prediction.label is not None:
            per_label = dict(metrics.get('per_label') or {})
            label_stats = dict(per_label.get(prediction.label) or {'total': 0, 'correct': 0, 'parsed': 0})
            label_stats['total'] = int(label_stats['total']) + 1
            if parsed_choice_id is not None:
                label_stats['parsed'] = int(label_stats['parsed']) + 1
                if is_correct:
                    label_stats['correct'] = int(label_stats['correct']) + 1
            per_label[prediction.label] = label_stats
            metrics['per_label'] = per_label

    def finalize_metrics(self, metrics: dict[str, object]) -> None:
        total_examples = int(metrics['examples_total'])
        parsed_predictions = int(metrics['parsed_predictions'])
        correct_predictions = int(metrics['correct_predictions'])
        if total_examples > 0:
            metrics['accuracy'] = correct_predictions / total_examples
        if parsed_predictions > 0:
            metrics['accuracy_on_parsed'] = correct_predictions / parsed_predictions

        latency_samples = int(metrics['latency_samples'])
        if latency_samples > 0:
            metrics['mean_latency_ms'] = float(metrics['total_latency_ms']) / latency_samples

        per_label = dict(metrics.get('per_label') or {})
        for label, stats in per_label.items():
            stats = dict(stats)
            total = int(stats.get('total', 0))
            correct = int(stats.get('correct', 0))
            stats['accuracy'] = correct / total if total > 0 else None
            per_label[label] = stats
        metrics['per_label'] = per_label

    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        lines = [
            f'Run directory: {run_dir}',
            f'Examples total: {metrics["examples_total"]}',
            f'Skipped existing: {metrics["skipped_existing"]}',
            f'Provider successes: {metrics["provider_successes"]}',
            f'Call failures: {metrics["call_failures"]}',
            f'Parse failures: {metrics["parse_failures"]}',
            f'Parsed predictions: {metrics["parsed_predictions"]}',
            f'Correct predictions: {metrics["correct_predictions"]}',
            f'Accuracy: {metrics["accuracy"]}',
            f'Accuracy on parsed: {metrics["accuracy_on_parsed"]}',
            '',
            f'Wall-clock time: {_fmt_latency(metrics.get("wall_clock_ms"))}',
            f'Concurrency: {metrics.get("concurrency", 1)}',
            f'Mean request latency: {_fmt_latency(metrics.get("mean_latency_ms"))}',
            f'Min request latency: {_fmt_latency(metrics.get("min_latency_ms"))}',
            f'Max request latency: {_fmt_latency(metrics.get("max_latency_ms"))}',
        ]
        per_label = dict(metrics.get('per_label') or {})
        if per_label:
            lines.append('')
            lines.append('Per-label accuracy:')
            for label in sorted(per_label):
                stats = per_label[label]
                acc = stats.get('accuracy')
                acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
                lines.append(f'  {label}: {stats.get("correct", 0)}/{stats.get("total", 0)} ({acc_str})')
        return '\n'.join(lines)
