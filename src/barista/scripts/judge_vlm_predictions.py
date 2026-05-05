"""Run G-Eval judge evaluation on an existing VLM benchmark run.

This script loads predictions from a completed run directory and applies
(or re-applies) judge evaluation to predictions that have inference results
but are missing judge scores. Useful for decoupling inference from evaluation,
or re-judging with a different model.

Supports: visual_qa, referring.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from barista.vlm_benchmarks.config import load_run_config
from barista.vlm_benchmarks.geval import (
    DeepEvalLLMAdapter,
    GEvalCriterionSpec,
    run_geval_criteria,
)
from barista.vlm_benchmarks.providers import create_provider_client
from barista.vlm_benchmarks.types import (
    BenchmarkExample,
    ModelSpec,
    PredictionResult,
)
from barista.vlm_benchmarks.tasks import create_run_task

logger = logging.getLogger(__name__)


def _load_predictions(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_examples(path: Path) -> dict[str, BenchmarkExample]:
    examples = {}
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                ex = BenchmarkExample.from_dict(json.loads(line))
                examples[ex.example_id] = ex
    return examples


def _needs_judging(record: dict[str, object], *, force: bool) -> bool:
    """Return True if this record should be judge-evaluated."""
    if record.get('error') is not None:
        return False
    task_result = record.get('task_result', {})
    if not task_result:
        return False
    if force:
        return True
    return task_result.get('geval_mean_score') is None


def _judge_visual_qa(
    record: dict[str, object],
    example: BenchmarkExample,
    *,
    criteria: dict[str, GEvalCriterionSpec],
    active_criteria: list[str],
    geval_llm: DeepEvalLLMAdapter,
    strict_mode: bool,
    verbose_mode: bool,
) -> dict[str, object]:
    """Run G-Eval judge on a visual_qa prediction."""
    task_result = dict(record.get('task_result', {}))
    prediction = str(task_result.get('predicted_answer', ''))
    reference = str(example.task_data.get('reference_answer', ''))

    if not prediction or not reference:
        return task_result

    context = {k: v for k, v in example.task_data.items() if k != 'reference_answer'}
    criterion_results = run_geval_criteria(
        criteria=criteria,
        reference=reference,
        prediction=prediction,
        geval_llm=geval_llm,
        active_criteria=active_criteria,
        strict_mode=strict_mode,
        verbose_mode=verbose_mode,
        context=context,
    )

    scores: list[float] = []
    for criterion_key, (score, reason) in criterion_results.items():
        task_result[f'geval_{criterion_key}_score'] = score
        task_result[f'geval_{criterion_key}_reason'] = reason
        if score is not None:
            scores.append(score)

    task_result['geval_mean_score'] = (sum(scores) / len(scores)) if scores else None
    return task_result


def _judge_referring(
    record: dict[str, object],
    example: BenchmarkExample,
    *,
    criteria: dict[str, GEvalCriterionSpec],
    active_criteria: list[str],
    geval_llm: DeepEvalLLMAdapter,
    strict_mode: bool,
    verbose_mode: bool,
) -> dict[str, object]:
    """Run G-Eval judge on a referring prediction."""
    task_result = dict(record.get('task_result', {}))
    prediction = str(task_result.get('description', ''))
    reference = str(example.task_data.get('structured', ''))

    if not prediction or not reference:
        return task_result

    task_result['reference_structured'] = reference

    context = {k: v for k, v in example.task_data.items() if k != 'structured'}
    criterion_results = run_geval_criteria(
        criteria=criteria,
        reference=reference,
        prediction=prediction,
        geval_llm=geval_llm,
        active_criteria=active_criteria,
        strict_mode=strict_mode,
        verbose_mode=verbose_mode,
        context=context,
    )

    scores: list[float] = []
    for criterion_key, (score, reason) in criterion_results.items():
        task_result[f'geval_{criterion_key}_score'] = score
        task_result[f'geval_{criterion_key}_reason'] = reason
        if score is not None:
            scores.append(score)

    task_result['geval_mean_score'] = (sum(scores) / len(scores)) if scores else None
    return task_result


def _get_criteria_for_task(task_name: str) -> dict[str, GEvalCriterionSpec]:
    if task_name == 'visual_qa':
        from barista.vlm_benchmarks.tasks.visual_qa import _CRITERIA

        return _CRITERIA
    if task_name == 'referring':
        from barista.vlm_benchmarks.tasks.referring import _CRITERIA

        return _CRITERIA
    raise ValueError(f'Judge evaluation not supported for task: {task_name}')


def _get_default_criteria(task_name: str) -> list[str]:
    return list(_get_criteria_for_task(task_name).keys())


def judge_run(
    run_dir: Path,
    *,
    judge_model: ModelSpec | None = None,
    concurrency: int = 1,
    force: bool = False,
    criteria: list[str] | None = None,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Run judge evaluation on an existing run directory.

    If *output_dir* is given the original run directory is left untouched:
    the input files are copied there and all outputs (predictions.jsonl,
    metrics.json, summary.txt) are written to *output_dir*.

    Returns the recomputed metrics dict.
    """
    predictions_path = run_dir / 'predictions.jsonl'
    examples_path = run_dir / 'benchmark_examples.jsonl'
    config_path = run_dir / 'run_config.json'

    if not predictions_path.exists():
        raise FileNotFoundError(f'No predictions found at {predictions_path}')
    if not examples_path.exists():
        raise FileNotFoundError(f'No benchmark examples found at {examples_path}')
    if not config_path.exists():
        raise FileNotFoundError(f'No run config found at {config_path}')

    # Set up write directory — copy inputs there if output_dir was requested.
    write_dir = output_dir if output_dir is not None else run_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for fname in ('predictions.jsonl', 'benchmark_examples.jsonl', 'run_config.json'):
            shutil.copy2(run_dir / fname, output_dir / fname)
        predictions_path = output_dir / 'predictions.jsonl'

    run_config = load_run_config(config_path)
    task_name = run_config.task

    task_criteria = _get_criteria_for_task(task_name)
    active_criteria = criteria if criteria is not None else _get_default_criteria(task_name)
    for c in active_criteria:
        if c not in task_criteria:
            raise ValueError(f'Unknown criterion {c!r} for task {task_name}. Available: {list(task_criteria.keys())}')

    # Set up judge LLM
    if judge_model is not None:
        judge_spec = judge_model
    else:
        judge_model_raw = run_config.task_params.get('judge_model')
        if judge_model_raw is not None:
            judge_spec = ModelSpec.model_validate(dict(judge_model_raw))
        else:
            judge_spec = run_config.model

    judge_client = create_provider_client(judge_spec)
    judge_config = run_config.model_copy(update={'model': judge_spec})
    geval_llm = DeepEvalLLMAdapter(judge_client, judge_config)

    # Load data
    records = _load_predictions(predictions_path)
    examples = _load_examples(examples_path)

    # Identify records that need judging
    to_judge = [(i, r) for i, r in enumerate(records) if _needs_judging(r, force=force)]
    logger.info(
        'Judging %d/%d predictions (force=%s, task=%s, judge=%s)',
        len(to_judge),
        len(records),
        force,
        task_name,
        judge_spec.model,
    )

    if not to_judge:
        logger.info('Nothing to judge — recomputing metrics only.')
    else:
        judge_fn = _judge_visual_qa if task_name == 'visual_qa' else _judge_referring

        progress = {'done': 0, 'total': len(to_judge), 'lock': threading.Lock()}

        def _process_one(idx: int, record: dict[str, object]) -> tuple[int, dict[str, object]]:
            example_id = str(record['example_id'])
            example = examples.get(example_id)
            if example is None:
                logger.warning('Example %s not found in benchmark_examples.jsonl, skipping', example_id)
                return idx, record.get('task_result', {})

            short_id = example_id.rsplit(':', 1)[-1] if ':' in example_id else example_id
            t0 = time.perf_counter()
            task_result = judge_fn(
                record,
                example,
                criteria=task_criteria,
                active_criteria=active_criteria,
                geval_llm=geval_llm,
                strict_mode=strict_mode,
                verbose_mode=verbose_mode,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            with progress['lock']:
                progress['done'] += 1
                score = task_result.get('geval_mean_score')
                logger.info(
                    '[judge] [%d/%d] %s  score=%s  %.0fms',
                    progress['done'],
                    progress['total'],
                    short_id,
                    score,
                    elapsed,
                )

            return idx, task_result

        if concurrency <= 1:
            for idx, record in to_judge:
                result_idx, task_result = _process_one(idx, record)
                records[result_idx]['task_result'] = task_result
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_process_one, idx, record): idx for idx, record in to_judge}
                for future in as_completed(futures):
                    result_idx, task_result = future.result()
                    records[result_idx]['task_result'] = task_result

        # Write updated predictions
        with predictions_path.open('w', encoding='utf-8') as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + '\n')

    # Recompute metrics (always write to write_dir) — inject active_criteria so the task picks them up
    # (the original run_config may have geval_criteria=[] to skip judging
    # during inference, but we need the criteria for metrics aggregation).
    metrics_config = run_config.model_copy(
        update={'task_params': {**run_config.task_params, 'geval_criteria': active_criteria}}
    )
    task = create_run_task(metrics_config)
    metrics = task.init_metrics(examples_total=len(records))
    for record in records:
        pred = PredictionResult(
            example_id=str(record['example_id']),
            task_name=str(record['task_name']),
            model=str(record['model']),
            provider=str(record['provider']),
            raw_text=str(record.get('raw_text', '')),
            error=record.get('error'),
            latency_ms=record.get('latency_ms'),
            usage=record.get('usage'),
            metadata=record.get('metadata', {}),
            task_result=record.get('task_result', {}),
            label=record.get('label'),
            finish_reason=record.get('finish_reason'),
            provider_response_id=record.get('provider_response_id'),
        )
        task.update_metrics(metrics, pred)

    task.finalize_metrics(metrics)

    # Preserve runner-only fields from the original metrics file.
    original_metrics_path = run_dir / 'metrics.json'
    _RUNNER_KEYS = (
        'wall_clock_ms',
        'concurrency',
        'total_input_tokens',
        'total_output_tokens',
        'total_thinking_tokens',
    )
    if original_metrics_path.exists():
        original_metrics = json.loads(original_metrics_path.read_text(encoding='utf-8'))
        for key in _RUNNER_KEYS:
            if key in original_metrics and key not in metrics:
                metrics[key] = original_metrics[key]

    (write_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    summary_text = task.format_summary(metrics, run_dir=write_dir)
    (write_dir / 'summary.txt').write_text(summary_text, encoding='utf-8')

    logger.info('Judge evaluation complete. Updated predictions, metrics, and summary.')
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run G-Eval judge evaluation on an existing VLM benchmark run directory.'
    )
    parser.add_argument(
        'run_dir',
        type=Path,
        help='Path to the run directory containing predictions.jsonl and benchmark_examples.jsonl.',
    )
    parser.add_argument(
        '--judge-config',
        type=Path,
        help='Path to a run config JSON to use for the judge model. If not provided, uses the run_config.json from the run directory.',
    )
    parser.add_argument(
        '--judge-model',
        help='Override judge model name (uses provider from run config or judge config).',
    )
    parser.add_argument(
        '--judge-provider',
        help='Override judge provider.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Number of parallel judge API requests.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-judge all predictions, even those that already have scores.',
    )
    parser.add_argument(
        '--criteria',
        nargs='+',
        help='Subset of criteria to evaluate (default: all for the task).',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict mode (0-or-1 scoring).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose deepeval output.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help=(
            'Write judged predictions, metrics, and summary to this directory '
            'instead of the original run directory. The original run is left untouched.'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    logging.getLogger('google_genai').setLevel(logging.WARNING)

    parser = build_parser()
    args = parser.parse_args(argv)

    judge_model: ModelSpec | None = None
    if args.judge_config is not None:
        judge_config = load_run_config(args.judge_config)
        judge_model_raw = judge_config.task_params.get('judge_model')
        if judge_model_raw is not None:
            judge_model = ModelSpec.model_validate(dict(judge_model_raw))
        else:
            judge_model = judge_config.model

    if args.judge_model or args.judge_provider:
        if judge_model is not None:
            updates = {}
            if args.judge_model:
                updates['model'] = args.judge_model
            if args.judge_provider:
                updates['provider'] = args.judge_provider
            judge_model = judge_model.model_copy(update=updates)
        else:
            # Load the run config to get base model spec
            run_config = load_run_config(args.run_dir / 'run_config.json')
            base = run_config.model
            updates = {}
            if args.judge_model:
                updates['model'] = args.judge_model
            if args.judge_provider:
                updates['provider'] = args.judge_provider
            judge_model = base.model_copy(update=updates)

    logging.info('Using judge model: %s', judge_model)
    output_dir = args.output_dir if hasattr(args, 'output_dir') else None
    metrics = judge_run(
        args.run_dir,
        judge_model=judge_model,
        concurrency=args.concurrency,
        force=args.force,
        criteria=args.criteria,
        strict_mode=args.strict,
        verbose_mode=args.verbose,
        output_dir=output_dir,
    )

    if metrics:
        read_dir = output_dir if output_dir is not None else args.run_dir
        summary_path = read_dir / 'summary.txt'
        if summary_path.exists():
            print(summary_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
