"""Shared base class for G-Eval-based VLM benchmark tasks.

Provides common infrastructure for tasks that evaluate VLM predictions
against reference text using deepeval's G-Eval framework:

- Judge LLM setup and caching
- Evaluation via ``run_geval_criteria``
- Metrics initialisation, accumulation, finalisation, and formatting
- Shared utility helpers (frame selection, score/latency formatting)

Subclasses must set the class attributes ``_criteria``,
``_prediction_field``, and ``_reference_field``, and implement the
remaining abstract methods from :class:`BenchmarkTask`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from barista.vlm_benchmarks.geval import (
    DeepEvalLLMAdapter,
    GEvalCriterionSpec,
    run_geval_criteria,
)
from barista.vlm_benchmarks.providers import create_provider_client
from barista.vlm_benchmarks.tasks.base import BenchmarkTask
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    ModelSpec,
    PredictionResult,
    RunConfig,
)

logger = logging.getLogger(__name__)


# ── Shared utility helpers ────────────────────────────────────────────────────


def select_frames(frame_indices: list[int], max_frames: int | None) -> list[int]:
    """Uniformly sub-sample *frame_indices* down to *max_frames*."""
    if max_frames is None or len(frame_indices) <= max_frames:
        return frame_indices
    if max_frames == 1:
        return [frame_indices[len(frame_indices) // 2]]
    last = len(frame_indices) - 1
    positions = [i * last // (max_frames - 1) for i in range(max_frames)]
    return [frame_indices[p] for p in positions]


def fmt_score(value: object) -> str:
    if value is None:
        return 'N/A'
    return f'{float(value):.4f}'


def fmt_latency(value: object) -> str:
    if value is None:
        return 'N/A'
    ms = float(value)
    if ms >= 1000.0:
        return f'{ms / 1000.0:.2f}s'
    return f'{ms:.0f}ms'


# ── Base class ────────────────────────────────────────────────────────────────


class GEvalTaskBase(BenchmarkTask):
    """Shared base for tasks that evaluate predictions via G-Eval.

    Subclasses must define:

    - ``_criteria`` - ``dict[str, GEvalCriterionSpec]`` with all supported
      evaluation criteria.
    - ``_prediction_field`` - key in ``task_result`` holding the model
      prediction (e.g. ``'predicted_caption'``).
    - ``_reference_field`` - key in ``task_data`` holding the ground truth
      (e.g. ``'reference_caption'``).

    And implement the remaining abstract methods from :class:`BenchmarkTask`:
    :meth:`iter_examples`, :meth:`default_system_prompt`,
    :meth:`render_prompt`, :meth:`parse_response`, :meth:`validate_example`.
    """

    _criteria: dict[str, GEvalCriterionSpec]
    _prediction_field: str
    _reference_field: str

    def __init__(
        self,
        dataset_root: Path,
        task_params: dict[str, object],
        *,
        run_config: RunConfig | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_params = dict(task_params)
        self._run_config = run_config
        self._judge_config: RunConfig | None = None
        self._geval_llm: DeepEvalLLMAdapter | None = None

        raw_criteria = self.task_params.get('geval_criteria')
        if raw_criteria is not None:
            self._active_criteria: list[str] = [str(c) for c in raw_criteria]
        else:
            self._active_criteria = list(self._criteria.keys())

        self._strict_mode: bool = bool(self.task_params.get('geval_strict_mode', False))
        self._verbose_mode: bool = bool(self.task_params.get('geval_verbose', False))

    # ── judge client ──────────────────────────────────────────────────────────

    def _get_geval_llm(self) -> DeepEvalLLMAdapter | None:
        if self._geval_llm is not None:
            return self._geval_llm
        if self._run_config is None:
            logger.warning('GEval judge requested but no run_config supplied; skipping evaluation.')
            return None
        judge_model_raw = self.task_params.get('judge_model')
        if judge_model_raw is not None:
            judge_spec = ModelSpec.model_validate(dict(judge_model_raw))
        else:
            judge_spec = self._run_config.model
        judge_client = create_provider_client(judge_spec)
        self._judge_config = self._run_config.model_copy(update={'model': judge_spec})
        self._geval_llm = DeepEvalLLMAdapter(judge_client, self._judge_config)
        return self._geval_llm

    # ── evaluation ────────────────────────────────────────────────────────────

    def _enrich_eval_result(self, example: BenchmarkExample, result: dict[str, object]) -> None:
        """Hook for subclasses to inject extra fields before G-Eval runs."""

    def evaluate(self, example: BenchmarkExample, task_result: dict[str, object]) -> dict[str, object]:
        result = dict(task_result)
        reference = str(example.task_data[self._reference_field])
        prediction = str(result.get(self._prediction_field, ''))
        result[self._reference_field] = reference
        self._enrich_eval_result(example, result)

        if not self._active_criteria or not prediction:
            return result

        geval_llm = self._get_geval_llm()
        if geval_llm is None:
            return result

        criterion_results = run_geval_criteria(
            criteria=self._criteria,
            reference=reference,
            prediction=prediction,
            geval_llm=geval_llm,
            active_criteria=self._active_criteria,
            strict_mode=self._strict_mode,
            verbose_mode=self._verbose_mode,
            context={k: v for k, v in example.task_data.items() if k != self._reference_field},
        )

        scores: list[float] = []
        for criterion_key, (score, reason) in criterion_results.items():
            result[f'geval_{criterion_key}_score'] = score
            result[f'geval_{criterion_key}_reason'] = reason
            if score is not None:
                scores.append(score)

        result['geval_mean_score'] = (sum(scores) / len(scores)) if scores else None
        return result

    # ── metrics ───────────────────────────────────────────────────────────────

    def init_metrics(self, *, examples_total: int) -> dict[str, object]:
        metrics: dict[str, object] = {
            'examples_total': examples_total,
            'skipped_existing': 0,
            'provider_successes': 0,
            'call_failures': 0,
            'parse_failures': 0,
            'parsed_predictions': 0,
            'geval_mean_score_sum': 0.0,
            'geval_mean_score_count': 0,
            'mean_geval_score': None,
            'latency_samples': 0,
            'total_latency_ms': 0.0,
            'mean_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
        }
        for criterion_key in self._active_criteria:
            metrics[f'geval_{criterion_key}_sum'] = 0.0
            metrics[f'geval_{criterion_key}_count'] = 0
            metrics[f'mean_geval_{criterion_key}'] = None
        return metrics

    def update_metrics(self, metrics: dict[str, object], prediction: PredictionResult) -> None:
        if prediction.error is not None:
            if prediction.error.startswith('provider_error:'):
                metrics['call_failures'] = int(metrics['call_failures']) + 1
            else:
                metrics['provider_successes'] = int(metrics['provider_successes']) + 1
                metrics['parse_failures'] = int(metrics['parse_failures']) + 1
        else:
            metrics['provider_successes'] = int(metrics['provider_successes']) + 1

        has_prediction = prediction.task_result.get(self._prediction_field) is not None
        if has_prediction:
            metrics['parsed_predictions'] = int(metrics['parsed_predictions']) + 1

            mean_score = prediction.task_result.get('geval_mean_score')
            if mean_score is not None:
                metrics['geval_mean_score_sum'] = float(metrics['geval_mean_score_sum']) + float(mean_score)
                metrics['geval_mean_score_count'] = int(metrics['geval_mean_score_count']) + 1

            for criterion_key in self._active_criteria:
                score = prediction.task_result.get(f'geval_{criterion_key}_score')
                if score is not None:
                    metrics[f'geval_{criterion_key}_sum'] = float(metrics[f'geval_{criterion_key}_sum']) + float(score)
                    metrics[f'geval_{criterion_key}_count'] = int(metrics[f'geval_{criterion_key}_count']) + 1

        if prediction.latency_ms is not None:
            latency = prediction.latency_ms
            metrics['latency_samples'] = int(metrics['latency_samples']) + 1
            metrics['total_latency_ms'] = float(metrics['total_latency_ms']) + latency
            current_min = metrics.get('min_latency_ms')
            current_max = metrics.get('max_latency_ms')
            metrics['min_latency_ms'] = latency if current_min is None else min(float(current_min), latency)
            metrics['max_latency_ms'] = latency if current_max is None else max(float(current_max), latency)

    def finalize_metrics(self, metrics: dict[str, object]) -> None:
        mean_score_count = int(metrics['geval_mean_score_count'])
        if mean_score_count > 0:
            metrics['mean_geval_score'] = float(metrics['geval_mean_score_sum']) / mean_score_count

        for criterion_key in self._active_criteria:
            count = int(metrics[f'geval_{criterion_key}_count'])
            if count > 0:
                metrics[f'mean_geval_{criterion_key}'] = float(metrics[f'geval_{criterion_key}_sum']) / count

        latency_samples = int(metrics['latency_samples'])
        if latency_samples > 0:
            metrics['mean_latency_ms'] = float(metrics['total_latency_ms']) / latency_samples

    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        mean_count = int(metrics.get('geval_mean_score_count', 0))
        lines = [
            f'Run directory: {run_dir}',
            f'Examples total: {metrics["examples_total"]}',
            f'Skipped existing: {metrics["skipped_existing"]}',
            f'Provider successes: {metrics["provider_successes"]}',
            f'Call failures: {metrics["call_failures"]}',
            f'Parse failures: {metrics["parse_failures"]}',
            f'Parsed predictions: {metrics["parsed_predictions"]}',
            '',
            f'G-Eval mean score (0-1): {fmt_score(metrics.get("mean_geval_score"))} (n={mean_count})',
        ]
        for criterion_key in self._active_criteria:
            criterion_name = self._criteria[criterion_key]['name']
            count = int(metrics.get(f'geval_{criterion_key}_count', 0))
            lines.append(f'  {criterion_name}: {fmt_score(metrics.get(f"mean_geval_{criterion_key}"))} (n={count})')
        lines += [
            '',
            f'Wall-clock time: {fmt_latency(metrics.get("wall_clock_ms"))}',
            f'Concurrency: {metrics.get("concurrency", 1)}',
            f'Mean request latency: {fmt_latency(metrics.get("mean_latency_ms"))}',
            f'Min request latency: {fmt_latency(metrics.get("min_latency_ms"))}',
            f'Max request latency: {fmt_latency(metrics.get("max_latency_ms"))}',
        ]
        return '\n'.join(lines)


__all__ = [
    'GEvalTaskBase',
    'fmt_latency',
    'fmt_score',
    'select_frames',
]
