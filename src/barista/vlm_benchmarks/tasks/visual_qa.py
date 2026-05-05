"""Visual question-answering task evaluation.

QA pairs are generated from annotations using an LLM (via
``categorized_qa_generation``) and cached under ``<output_dir>/cache/qa_pairs/``.
Subsequent runs skip regeneration.  Pre-generated JSONL files can also
be loaded directly via ``qa_pairs_dir``.

VLM predictions are evaluated against reference answers using deepeval's
G-Eval framework (accuracy, completeness).

Supported ``task_params`` keys:

- ``qa_pairs_dir`` (str | null): directory with per-video JSONL files.
  When *null* (default), QA pairs are generated inline.
- ``data_generation_model`` (dict | null): ``ModelSpec`` for the LLM that
  generates QA pairs.  Defaults to the main model with ``temperature=0.7``.
- ``clip_length_in_frames`` (int): frames per clip (default 4).
- ``frame_spacing`` (int): spacing between frames in a clip (default 30).
- ``video_id`` (str | null): optional single-video filter.
- ``max_frames_per_example`` (int | null): cap frames per clip.
- ``max_examples_per_category`` (int | null): cap examples per change category
  per video, applied during asset preparation (not at build time).
- ``shuffle_seed`` (int | null): seed for shuffling candidate clips before the
  per-category cap is applied during preparation.
- ``judge_model`` (dict | null): ``ModelSpec`` for the G-Eval judge.
- ``geval_criteria`` (list[str]): criteria to run (default: accuracy, completeness).
- ``geval_strict_mode`` (bool): 0-or-1 scoring (default: false).
- ``geval_verbose`` (bool): verbose deepeval output (default: false).
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

from barista.dataset import discover_light_videos, discover_video_dirs
from barista.vlm_benchmarks.geval import GEvalCriterionSpec
from barista.vlm_benchmarks.providers import create_provider_client
from barista.vlm_benchmarks.tasks.base import BenchmarkAssetPreparer
from barista.vlm_benchmarks.tasks.geval_task_base import GEvalTaskBase, select_frames
from barista.vlm_benchmarks.tasks.qa_generation import (
    generate_qa_for_selected_clips,
    scan_video_changes,
)
from barista.vlm_benchmarks.types import (
    AssetPreparationResult,
    BenchmarkExample,
    DatasetPrepareConfig,
    FrameInput,
    MessagePart,
    ModelSpec,
    PredictionResult,
    RunConfig,
    TextPart,
    render_frame_image_parts,
)

logger = logging.getLogger(__name__)

VISUAL_QA_SYSTEM_PROMPT = (
    'You are an expert at visual question answering for coffee-making videos. '
    'You will receive a sequence of ordered frames sampled from a video clip, '
    'along with a question about the content. '
    'Analyze the frames carefully and provide a detailed, accurate answer. '
    'Your answer should be specific, clear, and based solely on what is visible in the frames. '
    'Start your final response with "Answer:" followed by your answer.'
)

_ANSWER_PATTERN = re.compile(r'(?im)^\s*answer\s*:\s*(.+)$')


# ── G-Eval criterion definitions ─────────────────────────────────────────────

_CRITERIA: dict[str, GEvalCriterionSpec] = {
    'accuracy': {
        'name': 'Accuracy',
        'criteria': (
            'Are all claims in the actual_output factually correct per the expected_output? '
            'Only penalise claims that are directly contradicted — wrong change direction, '
            'wrong object name, or wrong contact state. '
            'Do not penalise omissions, extra details, or equivalent wording. '
            'Contact-action synonyms (touching/holding/gripping the same object) are the same state. '
            'Machine synonyms (coffee machine / espresso machine / capsule machine) are always acceptable. '
            'For action-transition questions, evaluate whether the contact targets in the actual_output '
            'are correct — do not penalise for omitting or including temporal anchors like '
            '"at the start" or "by the end", since the hand may appear at any point in the clip.'
        ),
        'rubric': [
            {
                'score_range': [0, 3],
                'expected_outcome': (
                    'The actual_output contains clearly wrong claims that directly '
                    'contradict the expected_output — reversed change directions, '
                    'wrong object names, or a stated contact state that is inconsistent '
                    'with the reference.'
                ),
            },
            {
                'score_range': [4, 7],
                'expected_outcome': (
                    'The actual_output is mostly correct but includes at least one '
                    'claim that directly contradicts the expected_output — e.g. wrong '
                    'change direction or wrong object name. Additional details not in '
                    'the expected_output should NOT place the answer here.'
                ),
            },
            {
                'score_range': [8, 10],
                'expected_outcome': (
                    'All key claims in the actual_output are correct and consistent '
                    'with the expected_output. Additional correct details, plausible '
                    'intermediate steps, or non-contact observations not in the '
                    'expected_output are acceptable. Equivalent terminology is acceptable.'
                ),
            },
        ],
    },
    'completeness': {
        'name': 'Completeness',
        'criteria': (
            'Does the actual_output cover every change listed in the expected_output? '
            'Do not penalise incorrect claims — only missing ones. '
            'Idle hand states between contact events do not need to be stated explicitly. '
            'Per-hand attribution (left vs right) is not required. '
            'Semantically equivalent descriptions are acceptable. '
            'For action-transition questions, evaluate whether every contact target in the expected_output '
            'is mentioned in the actual_output — ignore temporal anchors like '
            '"at the start" or "by the end" since the hand may appear at any point in the clip.'
        ),
        'rubric': [
            {
                'score_range': [0, 3],
                'expected_outcome': (
                    'The actual_output omits most changes — it gives a vague or generic '
                    'response that skips the specific objects, states, or transitions '
                    'from the expected_output.'
                ),
            },
            {
                'score_range': [4, 7],
                'expected_outcome': (
                    'The actual_output mentions some changes but misses at least one '
                    'important object, state transition, or activity step present in '
                    'the expected_output.'
                ),
            },
            {
                'score_range': [8, 10],
                'expected_outcome': (
                    'The actual_output covers all changes from the expected_output. Minor rephrasing is acceptable.'
                ),
            },
        ],
    },
}


def _load_qa_jsonl(path: Path) -> list[dict[str, object]]:
    entries = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _parse_answer(raw_text: str) -> str:
    matches = list(_ANSWER_PATTERN.finditer(raw_text))
    if matches:
        return matches[-1].group(1).strip()
    return raw_text.strip()


class VisualQAPrepareTask(BenchmarkAssetPreparer):
    def __init__(
        self,
        dataset_root: Path,
        task_params: dict[str, object],
        *,
        config: DatasetPrepareConfig,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)
        self.config = config

    def _resolve_qa_pairs_dir(self) -> Path:
        qa_pairs_dir_raw = self.task_params.get('qa_pairs_dir')
        if qa_pairs_dir_raw is not None:
            return Path(str(qa_pairs_dir_raw))
        return self.config.output_dir / 'cache' / 'qa_pairs'

    def _data_generation_model_spec(self) -> ModelSpec:
        data_gen_model_raw = self.task_params.get('data_generation_model')
        if data_gen_model_raw is not None:
            return ModelSpec.model_validate(dict(data_gen_model_raw))
        return self.config.model

    def prepare_assets(self) -> AssetPreparationResult:
        qa_pairs_dir = self._resolve_qa_pairs_dir()
        qa_pairs_dir.mkdir(parents=True, exist_ok=True)

        clip_length = int(self.task_params.get('clip_length_in_frames', 4))
        frame_spacing = int(self.task_params.get('frame_spacing', 30))

        max_per_cat_raw = self.task_params.get('max_examples_per_category')
        max_per_category = None if max_per_cat_raw is None else int(max_per_cat_raw)
        shuffle_seed_raw = self.task_params.get('shuffle_seed')
        shuffle_seed = None if shuffle_seed_raw is None else int(shuffle_seed_raw)

        data_gen_spec = self._data_generation_model_spec()
        client = create_provider_client(data_gen_spec)
        gen_config = RunConfig(
            dataset_root=Path('.'), task='_text_generation', task_params={}, model=data_gen_spec, output_dir=Path('.')
        )

        video_dirs = discover_video_dirs(self.dataset_root)
        video_id_filter = self.task_params.get('video_id')
        if video_id_filter is not None:
            video_dirs = [(d, vid) for d, vid in video_dirs if vid == str(video_id_filter)]

        # Split into already-cached and pending videos.
        cached_count = 0
        pending_video_dirs: list[tuple[Path, str]] = []
        for video_dir, video_id in video_dirs:
            output_path = qa_pairs_dir / f'{video_id}_qa_pairs.jsonl'
            if output_path.exists():
                logger.info('QA pairs cache exists for %s, skipping', video_id)
                cached_count += 1
            else:
                pending_video_dirs.append((video_dir, video_id))

        # Phase 1: scan all pending videos for change clips (annotation-only, no LLM).
        per_category: dict[str, list[tuple[Path, str, list[int]]]] = defaultdict(list)
        for video_dir, video_id in pending_video_dirs:
            logger.info('Scanning change clips for video %s', video_id)
            clips = scan_video_changes(video_dir, clip_length, frame_spacing)
            if clips is None:
                continue
            for clip, changes in clips:
                for category in changes:
                    per_category[category].append((video_dir, video_id, clip))

        # Phase 2: apply global max_per_category cap with shuffle across all videos.
        rng = random.Random(shuffle_seed)
        selected: dict[tuple[str, tuple[int, ...]], set[str]] = defaultdict(set)
        for category, candidates in per_category.items():
            shuffled: list[tuple[Path, str, list[int]]] = list(candidates)
            rng.shuffle(shuffled)
            if max_per_category is not None:
                shuffled = shuffled[:max_per_category]
            for _video_dir, video_id, clip in shuffled:
                selected[(video_id, tuple(clip))].add(category)

        # Group selected clips by video for generation.
        by_video: dict[str, dict[tuple[int, ...], set[str]]] = defaultdict(lambda: defaultdict(set))
        for (video_id, clip_tuple), categories in selected.items():
            by_video[video_id][clip_tuple] |= categories

        # Phase 3: generate QA for each pending video's selected clips.
        generated_count = 0
        for video_dir, video_id in pending_video_dirs:
            output_path = qa_pairs_dir / f'{video_id}_qa_pairs.jsonl'
            clips_for_video = by_video.get(video_id)
            if not clips_for_video:
                output_path.write_text('')
                generated_count += 1
                continue
            logger.info('Generating QA pairs for video %s (%d clips)', video_id, len(clips_for_video))
            generate_qa_for_selected_clips(video_dir, clips_for_video, client, gen_config, output_path)
            generated_count += 1

        prepared_videos = cached_count + generated_count

        summary_text = '\n'.join(
            [
                f'QA pairs dir: {qa_pairs_dir}',
                f'Prepared videos: {prepared_videos}',
                'Task: visual_qa',
            ]
        )
        return AssetPreparationResult(
            task_name='visual_qa',
            artifact_path=qa_pairs_dir,
            artifacts_total=prepared_videos,
            summary_text=summary_text,
        )


class _VisualQABaseTask(GEvalTaskBase):
    """Visual question-answering task for video frame understanding.

    Loads generated QA pairs and presents them
    to the VLM with corresponding video frames. The task evaluates how well
    the model can answer questions about video content.

    Result keys added per example:

    - ``predicted_answer`` – the model's answer to the question.
    - ``reference_answer`` – the ground-truth answer.
    - ``question`` – the question that was asked.
    """

    _criteria = _CRITERIA
    _prediction_field = 'predicted_answer'
    _reference_field = 'reference_answer'

    def _enrich_eval_result(self, example: BenchmarkExample, result: dict[str, object]) -> None:
        result['question'] = str(example.task_data['question'])
        category = example.task_data.get('category')
        if category is not None:
            result['category'] = str(category)

    def _resolve_qa_pairs_dir(self) -> Path:
        qa_pairs_dir_raw = self.task_params.get('qa_pairs_dir')
        if qa_pairs_dir_raw is None:
            raise ValueError(
                'Visual QA requires task_params.qa_pairs_dir. '
                'Run barista-vlm-prepare with a visual_qa prepare config first.'
            )
        return Path(str(qa_pairs_dir_raw))

    def iter_examples(self) -> list[BenchmarkExample]:
        qa_pairs_dir = self._resolve_qa_pairs_dir()
        if not qa_pairs_dir.exists():
            raise ValueError(
                f'Visual QA prepared assets not found at {qa_pairs_dir}. '
                'Run barista-vlm-prepare with a visual_qa prepare config first.'
            )

        video_id_filter = self.task_params.get('video_id')
        max_frames_raw = self.task_params.get('max_frames_per_example')
        max_frames = None if max_frames_raw is None else int(max_frames_raw)  # ty:ignore[invalid-argument-type]

        videos = discover_light_videos(self.dataset_root)
        if video_id_filter is not None:
            videos = [v for v in videos if v.video_id == str(video_id_filter)]

        video_by_id = {v.video_id: v for v in videos}

        examples: list[BenchmarkExample] = []
        for video_id, video in sorted(video_by_id.items()):
            qa_pairs_path = qa_pairs_dir / f'{video_id}_qa_pairs.jsonl'
            if not qa_pairs_path.exists():
                continue

            for entry in _load_qa_jsonl(qa_pairs_path):
                frame_indices = [int(fi) for fi in entry.get('frame_indices', [])]
                clip_index = entry.get('clip_index')
                qa_pairs = entry.get('qa_pairs', [])
                if clip_index is None or not frame_indices:
                    continue

                valid_frame_indices = [fi for fi in frame_indices if fi in video.frame_indices]
                if not valid_frame_indices:
                    continue
                selected_indices = select_frames(valid_frame_indices, max_frames)

                frame_inputs = [
                    FrameInput(video_id=video.video_id, frame_index=fi, mp4_path=video.mp4_path)
                    for fi in selected_indices
                ]

                for qa_idx, qa_pair in enumerate(qa_pairs):
                    question = str(qa_pair.get('question', ''))
                    answer = str(qa_pair.get('answer', ''))
                    category = qa_pair.get('category')
                    if not question or not answer:
                        continue

                    cat_suffix = f':{category}' if category else ''
                    example_id = (
                        f'visual_qa:{video_id}:clip_{clip_index}'
                        f':frames_{selected_indices[0]}_{selected_indices[-1]}'
                        f':qa_{qa_idx}{cat_suffix}'
                    )

                    metadata: dict[str, object] = {
                        'video_id': video_id,
                        'clip_index': int(clip_index),
                        'qa_index': qa_idx,
                        'original_frame_indices': frame_indices,
                        'selected_frame_indices': selected_indices,
                    }
                    task_data: dict[str, object] = {
                        'question': question,
                        'reference_answer': answer,
                        'num_frames': len(selected_indices),
                    }
                    if category is not None:
                        metadata['category'] = str(category)
                        task_data['category'] = str(category)

                    examples.append(
                        BenchmarkExample(
                            example_id=example_id,
                            task_name='visual_qa',
                            label=answer,
                            frames=frame_inputs,
                            metadata=metadata,
                            task_data=task_data,
                        )
                    )

        return examples

    def default_system_prompt(self) -> str:
        return VISUAL_QA_SYSTEM_PROMPT

    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        image_parts = render_frame_image_parts(example.frames)
        parts: list[MessagePart] = []
        for i, image_part in enumerate(image_parts, 1):
            parts.append(TextPart(text=f'Frame {i}:'))
            parts.append(image_part)

        num_frames = len(example.frames)
        question = str(example.task_data['question'])
        user_prompt = '\n'.join(
            [
                'Task: Visual question answering.',
                f'You will receive {num_frames} ordered frames sampled from a video clip '
                f'(Frame 1 is earliest, Frame {num_frames} is latest).',
                '',
                'Question:',
                question,
                '',
                'Provide a clear, specific answer based on the frames shown.',
            ]
        )
        parts.append(TextPart(text=user_prompt))
        return parts

    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, object] | None:
        return {'predicted_answer': _parse_answer(raw_text)}

    def validate_example(self, example: BenchmarkExample) -> None:
        if not example.task_data.get('question'):
            raise ValueError(f'Missing question in task_data for {example.example_id}')
        if not example.task_data.get('reference_answer'):
            raise ValueError(f'Missing reference_answer in task_data for {example.example_id}')

    # ── Per-category metrics ──────────────────────────────────────────────────

    def init_metrics(self, *, examples_total: int) -> dict[str, object]:
        metrics = super().init_metrics(examples_total=examples_total)
        # {category: {score_sum, count}} — populated dynamically
        metrics['_category_scores'] = {}
        return metrics

    def update_metrics(self, metrics: dict[str, object], prediction: PredictionResult) -> None:
        super().update_metrics(metrics, prediction)

        category = prediction.task_result.get('category')
        mean_score = prediction.task_result.get('geval_mean_score')
        if category is None or mean_score is None:
            return

        cat_key = str(category)
        cat_scores: dict = metrics['_category_scores']  # type: ignore[assignment]
        if cat_key not in cat_scores:
            cat_scores[cat_key] = {'score_sum': 0.0, 'count': 0}
        cat_scores[cat_key]['score_sum'] += float(mean_score)
        cat_scores[cat_key]['count'] += 1

    def finalize_metrics(self, metrics: dict[str, object]) -> None:
        super().finalize_metrics(metrics)

        cat_scores: dict = metrics.get('_category_scores', {})  # type: ignore[assignment]
        cat_means: dict[str, float | None] = {}
        for cat_key, data in cat_scores.items():
            if data['count'] > 0:
                cat_means[cat_key] = data['score_sum'] / data['count']
            else:
                cat_means[cat_key] = None
        metrics['category_mean_scores'] = cat_means

    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        base = super().format_summary(metrics, run_dir=run_dir)

        cat_means: dict = metrics.get('category_mean_scores', {})  # type: ignore[assignment]
        if not cat_means:
            return base

        cat_scores: dict = metrics.get('_category_scores', {})  # type: ignore[assignment]
        lines = ['', 'Per-category G-Eval scores:']
        for cat_key in sorted(cat_means):
            score = cat_means[cat_key]
            count = cat_scores.get(cat_key, {}).get('count', 0)
            score_str = f'{score:.4f}' if score is not None else 'N/A'
            lines.append(f'  {cat_key}: {score_str} (n={count})')

        return base + '\n'.join(lines)


class VisualQABuildTask(_VisualQABaseTask):
    def __init__(self, dataset_root: Path, task_params: dict[str, object]) -> None:
        super().__init__(dataset_root, task_params, run_config=None)


class VisualQARunTask(_VisualQABaseTask):
    def __init__(self, dataset_root: Path, task_params: dict[str, object], *, run_config: RunConfig) -> None:
        super().__init__(dataset_root, task_params, run_config=run_config)
