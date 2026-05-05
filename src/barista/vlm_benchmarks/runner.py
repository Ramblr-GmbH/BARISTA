from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from barista.vlm_benchmarks.config import write_config_snapshot
from barista.vlm_benchmarks.providers import create_provider_client
from barista.vlm_benchmarks.providers.base import VlmProviderClient
from barista.vlm_benchmarks.tasks import create_build_task, create_prepare_task, create_run_task
from barista.vlm_benchmarks.tasks.base import BenchmarkExampleBuilder, BenchmarkTask
from barista.vlm_benchmarks.tasks.grounding import save_debug_frame as grounding_save_debug_frame
from barista.vlm_benchmarks.tasks.hand_object import save_debug_frame as hand_object_save_debug_frame
from barista.vlm_benchmarks.types import (
    AssetPreparationResult,
    BenchmarkExample,
    DatasetBuildConfig,
    DatasetPrepareConfig,
    ImagePart,
    MessagePart,
    ModelResponse,
    PredictionResult,
    RunConfig,
    TextPart,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, object]
    summary_text: str


@dataclass(frozen=True)
class BenchmarkDatasetBuildResult:
    examples_path: Path
    examples_total: int
    summary_text: str


def _prediction_from_dict(d: dict) -> PredictionResult:
    return PredictionResult(
        example_id=d['example_id'],
        task_name=d['task_name'],
        model=d['model'],
        provider=d['provider'],
        raw_text=d.get('raw_text', ''),
        error=d.get('error'),
        latency_ms=d.get('latency_ms'),
        usage=d.get('usage'),
        metadata=d.get('metadata', {}),
        task_result=d.get('task_result') or {},
        label=d.get('label'),
        finish_reason=d.get('finish_reason'),
        provider_response_id=d.get('provider_response_id'),
        prompt_text=d.get('prompt_text'),
    )


def _load_existing_predictions(predictions_path: Path) -> list[PredictionResult]:
    """Load existing predictions from a JSONL file, returning an empty list if the file doesn't exist."""
    if not predictions_path.exists():
        return []
    results: list[PredictionResult] = []
    with predictions_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(_prediction_from_dict(json.loads(line)))
    return results


def _append_jsonl(file_obj, record: dict[str, object]) -> None:
    file_obj.write(json.dumps(record, sort_keys=True) + '\n')
    file_obj.flush()


def _benchmark_dataset_sidecar_path(examples_path: Path, *, suffix: str) -> Path:
    return examples_path.parent / f'{examples_path.stem}{suffix}'


def _asset_preparation_sidecar_paths(artifact_path: Path) -> tuple[Path, Path]:
    if artifact_path.suffix:
        return (
            artifact_path.with_name(f'{artifact_path.stem}.config.json'),
            artifact_path.with_name(f'{artifact_path.stem}.summary.txt'),
        )
    return artifact_path / 'prepare_config.json', artifact_path / 'prepare_summary.txt'


def _append_usage_block(summary_text: str, metrics: dict[str, object]) -> str:
    input_tokens = metrics.get('total_input_tokens')
    output_tokens = metrics.get('total_output_tokens')
    if input_tokens is None and output_tokens is None:
        return summary_text
    thinking_tokens = metrics.get('total_thinking_tokens')
    total = int(input_tokens or 0) + int(output_tokens or 0) + int(thinking_tokens or 0)
    lines = [
        '',
        'Token usage:',
        f'  Input tokens:  {int(input_tokens or 0):,}',
        f'  Output tokens: {int(output_tokens or 0):,}',
    ]
    if thinking_tokens:
        lines.append(f'  Thinking tokens: {int(thinking_tokens):,}')
    lines.append(f'  Total tokens:  {total:,}')
    return summary_text + '\n' + '\n'.join(lines)


def _validate_benchmark_examples(
    examples: list[BenchmarkExample],
    *,
    task: BenchmarkExampleBuilder | None = None,
    expected_task_name: str | None = None,
) -> str | None:
    if not examples:
        return None

    seen_example_ids: set[str] = set()
    task_name: str | None = None
    for example in examples:
        if task_name is None:
            task_name = example.task_name
        elif example.task_name != task_name:
            raise ValueError(
                f'Frozen benchmark examples contain multiple task names: {task_name!r} and {example.task_name!r}'
            )

        if example.example_id in seen_example_ids:
            raise ValueError(f'Duplicate example_id in frozen benchmark examples: {example.example_id}')
        seen_example_ids.add(example.example_id)

        if task is not None:
            task.validate_example(example)
    if expected_task_name is not None and task_name != expected_task_name:
        raise ValueError(f'Config task {expected_task_name!r} does not match benchmark examples task {task_name!r}')
    return task_name


def _call_with_retries(
    *,
    provider_client: VlmProviderClient,
    system_prompt: str,
    parts: list[MessagePart],
    config: RunConfig,
    response_schema: type | None = None,
) -> ModelResponse | None:
    max_attempts = config.model.max_retries + 1
    for attempt in range(max_attempts):
        try:
            return provider_client.generate(system_prompt, parts, config, response_schema=response_schema)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(2**attempt, 2))


def _write_and_update(
    record: PredictionResult,
    *,
    task: BenchmarkTask,
    metrics: dict[str, object],
    predictions_file: object,
    errors_file: object,
) -> None:
    record_dict = asdict(record)
    _append_jsonl(predictions_file, record_dict)
    if record.error is not None:
        _append_jsonl(errors_file, record_dict)
    task.update_metrics(metrics, record)
    if record.usage is not None:
        metrics['total_input_tokens'] = int(metrics.get('total_input_tokens', 0)) + record.usage.get('input_tokens', 0)
        metrics['total_output_tokens'] = int(metrics.get('total_output_tokens', 0)) + record.usage.get(
            'output_tokens', 0
        )
        if record.usage.get('thinking_tokens'):
            metrics['total_thinking_tokens'] = (
                int(metrics.get('total_thinking_tokens', 0)) + record.usage['thinking_tokens']
            )


def create_run_dir(output_root: Path, *, task_name: str, model_name: str) -> Path:
    """Return a timestamped run directory path under *output_root* (not created yet)."""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe_model = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in model_name)
    run_id = f'{timestamp}_{task_name}_{safe_model}'
    return Path(output_root) / run_id


def prepare_examples(
    examples: list[BenchmarkExample],
    *,
    limit: int | None,
    shuffle_seed: int | None,
) -> list[BenchmarkExample]:
    """Optionally shuffle and/or truncate *examples*, returning a new list."""
    prepared = list(examples)
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(prepared)
    if limit is not None:
        prepared = prepared[:limit]
    return prepared


def write_benchmark_examples(examples: list[BenchmarkExample], path: Path) -> None:
    """Serialize *examples* to a JSONL file at *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_obj:
        for example in examples:
            file_obj.write(json.dumps(example.to_dict(), sort_keys=True) + '\n')


def load_benchmark_examples(path: Path) -> list[BenchmarkExample]:
    """Load and validate benchmark examples from a JSONL file."""
    examples: list[BenchmarkExample] = []
    with path.open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(BenchmarkExample.from_dict(payload))
    _validate_benchmark_examples(examples)
    return examples


def _extract_prompt_text(parts: list[MessagePart]) -> str:
    """Return the concatenation of all TextPart texts in the prompt."""
    return '\n'.join(part.text for part in parts if isinstance(part, TextPart))


def _log_prompt(
    system_prompt: str,
    parts: list[MessagePart],
    *,
    example_id: str,
    response_schema: type | None = None,
) -> None:
    lines = [
        f'=== Prompt for first example ({example_id}) ===',
        '',
        '--- System prompt ---',
        system_prompt,
        '',
        '--- Message parts ---',
    ]
    for i, part in enumerate(parts):
        if isinstance(part, TextPart):
            lines.append(f'[{i}] TextPart:')
            lines.append(part.text)
        elif isinstance(part, ImagePart):
            lines.append(f'[{i}] ImagePart({part.mime_type}, {len(part.data)} bytes)')
        lines.append('')
    if response_schema is not None:
        lines.append(f'--- Response schema: {response_schema.__name__} ---')
        lines.append('')
    lines.append('=== End of prompt ===')
    logger.info('\n'.join(lines))


def _run_one_example(
    example: BenchmarkExample,
    *,
    config: RunConfig,
    task: BenchmarkTask,
    provider_client: VlmProviderClient,
    system_prompt: str,
    progress: dict | None = None,
    log_prompt: bool = False,
) -> PredictionResult:
    short_id = example.example_id.rsplit(':', 1)[-1] if ':' in example.example_id else example.example_id
    label = example.label or ''
    base_kwargs = {
        'example_id': example.example_id,
        'task_name': example.task_name,
        'model': config.model.model,
        'provider': config.model.provider,
        'metadata': example.metadata,
        'label': label,
    }

    parts: list[MessagePart] | None = None
    try:
        parts = task.render_prompt(example)
        if log_prompt:
            _log_prompt(
                system_prompt,
                parts,
                example_id=example.example_id,
                response_schema=task.response_schema(),
            )
        response = _call_with_retries(
            provider_client=provider_client,
            system_prompt=system_prompt,
            parts=parts,
            config=config,
            response_schema=task.response_schema(),
        )
    except Exception as exc:
        logger.warning('[inference] %s FAILED: %s', short_id, exc)
        return PredictionResult(
            **base_kwargs,
            raw_text='',
            error=f'provider_error: {exc.__class__.__name__}: {exc}',
            latency_ms=None,
            usage=None,
            prompt_text=_extract_prompt_text(parts) if parts else None,
        )

    assert response is not None
    task_result = task.parse_response(example, response.raw_text)

    # Retry with doubled max_output_tokens on MAX_TOKENS truncation
    if (
        task_result is None
        and response.finish_reason is not None
        and 'MAX_TOKENS' in response.finish_reason
        and config.model.max_output_tokens is not None
    ):
        new_max = config.model.max_output_tokens * 2
        logger.info(
            '[retry] %s hit MAX_TOKENS, retrying with max_output_tokens=%d',
            short_id,
            new_max,
        )
        retry_config = config.model_copy(
            update={'model': config.model.model_copy(update={'max_output_tokens': new_max})}
        )
        try:
            response = _call_with_retries(
                provider_client=provider_client,
                system_prompt=system_prompt,
                parts=parts,
                config=retry_config,
                response_schema=task.response_schema(),
            )
            assert response is not None
            task_result = task.parse_response(example, response.raw_text)
        except Exception as exc:
            logger.warning('[retry] %s FAILED: %s', short_id, exc)

    if task_result is None:
        logger.warning('[parse] %s FAILED: could not parse response', short_id)
        try:
            failure_task_result = task.evaluate(example, {})
        except Exception:
            failure_task_result = {}
        return PredictionResult(
            **base_kwargs,
            raw_text=response.raw_text,
            error='parse_error: could not parse response',
            task_result=failure_task_result,
            latency_ms=response.latency_ms,
            usage=response.usage,
            finish_reason=response.finish_reason,
            provider_response_id=response.provider_response_id,
            prompt_text=_extract_prompt_text(parts),
        )

    task_result = task.evaluate(example, task_result)

    done_str = ''
    if progress is not None:
        with progress['lock']:
            progress['done'] += 1
            done_str = f' [{progress["done"]}/{progress["total"]}]'
    score = task_result.get('geval_mean_score')
    logger.info('[done]%s %s  score=%s  latency=%dms', done_str, short_id, score, response.latency_ms or 0)

    return PredictionResult(
        **base_kwargs,
        raw_text=response.raw_text,
        error=None,
        latency_ms=response.latency_ms,
        usage=response.usage,
        task_result=task_result,
        finish_reason=response.finish_reason,
        provider_response_id=response.provider_response_id,
        prompt_text=_extract_prompt_text(parts),
    )


def _run_examples(
    examples: list[BenchmarkExample],
    *,
    config: RunConfig,
    task: BenchmarkTask,
    provider_client: VlmProviderClient,
    system_prompt: str,
    run_dir: Path,
    existing_predictions: list[PredictionResult] | None = None,
) -> dict[str, object]:
    """Run inference on *examples* and write predictions/metrics to *run_dir*."""
    predictions_path = run_dir / 'predictions.jsonl'
    errors_path = run_dir / 'errors.jsonl'
    metrics_path = run_dir / 'metrics.json'

    # Resume: skip already-completed examples and replay their metrics
    completed_ids: set[str] = set()
    if existing_predictions:
        completed_ids = {p.example_id for p in existing_predictions}
        examples = [e for e in examples if e.example_id not in completed_ids]
        logger.info('Resuming: %d already completed, %d remaining', len(completed_ids), len(examples))

    total_examples = len(examples) + len(completed_ids)
    metrics = task.init_metrics(examples_total=total_examples)

    # Replay existing predictions into metrics
    if existing_predictions:
        for pred in existing_predictions:
            task.update_metrics(metrics, pred)
            if pred.usage is not None:
                metrics['total_input_tokens'] = int(metrics.get('total_input_tokens', 0)) + pred.usage.get(
                    'input_tokens', 0
                )
                metrics['total_output_tokens'] = int(metrics.get('total_output_tokens', 0)) + pred.usage.get(
                    'output_tokens', 0
                )
                if pred.usage.get('thinking_tokens'):
                    metrics['total_thinking_tokens'] = (
                        int(metrics.get('total_thinking_tokens', 0)) + pred.usage['thinking_tokens']
                    )

    concurrency = max(config.concurrency, 1)
    logger.info('Running %d examples (concurrency=%d)', len(examples), concurrency)
    _progress = {'done': 0, 'total': len(examples), 'lock': threading.Lock()}

    debug_dir = None
    debug_samples = getattr(config, 'debug_samples', 0) or 0
    if debug_samples > 0 and config.task in ('hand_object', 'grounding'):
        debug_dir = run_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f'Debug frames will be saved to {debug_dir}')

    _debug_save_fn = {
        'hand_object': hand_object_save_debug_frame,
        'grounding': grounding_save_debug_frame,
    }.get(config.task or '')
    n_debug_saved = [0]  # mutable for closure

    def _maybe_save_debug(example: BenchmarkExample, record: PredictionResult) -> None:
        if debug_dir is None or _debug_save_fn is None or record.error is not None or n_debug_saved[0] >= debug_samples:
            return
        doc_uuid = str(example.metadata.get('video_id', '')).split('/')[0]
        frame_index = example.metadata.get('frame_index', '')
        _debug_save_fn(example, record, debug_dir / f'{doc_uuid}_{frame_index}.jpg')
        n_debug_saved[0] += 1

    wall_start = time.perf_counter()

    file_mode = 'a' if existing_predictions else 'w'
    with (
        predictions_path.open(file_mode, encoding='utf-8') as predictions_file,
        errors_path.open(file_mode, encoding='utf-8') as errors_file,
    ):
        is_first_example = True
        if concurrency <= 1:
            for example in examples:
                record = _run_one_example(
                    example=example,
                    config=config,
                    task=task,
                    provider_client=provider_client,
                    system_prompt=system_prompt,
                    progress=_progress,
                    log_prompt=is_first_example,
                )
                is_first_example = False
                _write_and_update(
                    record,
                    task=task,
                    metrics=metrics,
                    predictions_file=predictions_file,
                    errors_file=errors_file,
                )
                _maybe_save_debug(example, record)
        else:
            write_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        _run_one_example,
                        example=example,
                        config=config,
                        task=task,
                        provider_client=provider_client,
                        system_prompt=system_prompt,
                        progress=_progress,
                        log_prompt=(i == 0),
                    ): example
                    for i, example in enumerate(examples)
                }
                for future in as_completed(futures):
                    example = futures[future]
                    record = future.result()
                    with write_lock:
                        _write_and_update(
                            record,
                            task=task,
                            metrics=metrics,
                            predictions_file=predictions_file,
                            errors_file=errors_file,
                        )
                        _maybe_save_debug(example, record)

    wall_clock_ms = (time.perf_counter() - wall_start) * 1000.0
    metrics['wall_clock_ms'] = wall_clock_ms
    metrics['concurrency'] = concurrency
    if completed_ids:
        metrics['resumed_examples'] = len(completed_ids)
    if debug_dir is not None:
        metrics['debug_frames_saved'] = n_debug_saved[0]

    task.finalize_metrics(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return metrics


def build_benchmark_examples(
    config: DatasetBuildConfig,
    *,
    task: BenchmarkExampleBuilder | None = None,
) -> list[BenchmarkExample]:
    """Build and return prepared benchmark examples from *config*, applying limit/shuffle."""
    if task is None:
        config.task_params['shuffle_seed'] = config.shuffle_seed
        task = create_build_task(config)
    examples = task.iter_examples()
    return prepare_examples(
        examples,
        limit=config.limit,
        shuffle_seed=config.shuffle_seed,
    )


def build_and_save_benchmark_dataset(config: DatasetBuildConfig, *, examples_path: Path) -> BenchmarkDatasetBuildResult:
    """Build examples, write them to *examples_path*, and save config/summary sidecars."""
    prepared_examples = build_benchmark_examples(config)
    task_name = _validate_benchmark_examples(prepared_examples, expected_task_name=config.task)
    write_benchmark_examples(prepared_examples, examples_path)
    write_config_snapshot(config, _benchmark_dataset_sidecar_path(examples_path, suffix='.config.json'))
    summary_text = '\n'.join(
        [
            f'Examples path: {examples_path}',
            f'Examples total: {len(prepared_examples)}',
            f'Task: {task_name or config.task}',
        ]
    )
    _benchmark_dataset_sidecar_path(examples_path, suffix='.summary.txt').write_text(
        summary_text + '\n',
        encoding='utf-8',
    )
    return BenchmarkDatasetBuildResult(
        examples_path=examples_path,
        examples_total=len(prepared_examples),
        summary_text=summary_text,
    )


def prepare_benchmark_assets(config: DatasetPrepareConfig) -> AssetPreparationResult:
    """Run the asset-preparation step for *config* and write config/summary sidecars."""
    task = create_prepare_task(config)
    result = task.prepare_assets()
    config_path, summary_path = _asset_preparation_sidecar_paths(result.artifact_path)
    write_config_snapshot(config, config_path)
    summary_path.write_text(result.summary_text + '\n', encoding='utf-8')
    return result


def run_benchmark(
    config: RunConfig,
    *,
    examples: list[BenchmarkExample],
    examples_source_path: Path | None = None,
    run_dir: Path | None = None,
) -> RunResult:
    """Run a full benchmark against a pre-built example list, calling the model and computing metrics.

    A timestamped run directory is created under ``config.output_dir`` unless
    *run_dir* is given explicitly.
    """
    task = create_run_task(config)
    prepared_examples = list(examples)
    task_name = _validate_benchmark_examples(prepared_examples, task=task, expected_task_name=config.task)
    if task_name is None:
        task_name = config.task
    run_dir = (
        Path(run_dir)
        if run_dir is not None
        else create_run_dir(
            config.output_dir,
            task_name=task_name,
            model_name=config.model.model,
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, run_dir / 'run_config.json')
    write_benchmark_examples(prepared_examples, run_dir / 'benchmark_examples.jsonl')
    if examples_source_path is not None:
        (run_dir / 'benchmark_examples_source.txt').write_text(str(examples_source_path) + '\n', encoding='utf-8')

    # Load existing predictions for resume
    existing_predictions = _load_existing_predictions(run_dir / 'predictions.jsonl')

    provider_client = create_provider_client(config.model)
    system_prompt = config.system_prompt if config.system_prompt else task.default_system_prompt()
    metrics = _run_examples(
        prepared_examples,
        config=config,
        task=task,
        provider_client=provider_client,
        system_prompt=system_prompt,
        run_dir=run_dir,
        existing_predictions=existing_predictions or None,
    )
    summary_text = task.format_summary(metrics, run_dir=run_dir)
    summary_text = _append_usage_block(summary_text, metrics)
    (run_dir / 'summary.txt').write_text(summary_text, encoding='utf-8')
    return RunResult(run_dir=run_dir, metrics=metrics, summary_text=summary_text)
