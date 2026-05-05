from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from barista.vlm_benchmarks.types import DatasetBuildConfig, DatasetPrepareConfig, RunConfig

_UNSET = object()  # sentinel for unspecified CLI overrides


def _default_api_key_env(provider: object) -> str | None:
    if provider == 'gemini':
        return 'GEMINI_API_KEY'
    if provider in {'openai', 'openai_compat', 'azure_openai'}:
        return 'OPENAI_API_KEY'
    return None


def _load_json_config(config_path: Path) -> dict[str, object]:
    return json.loads(config_path.read_text(encoding='utf-8'))


def _normalize_loaded_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)

    model_payload = cast(dict[str, object], normalized.get('model') or {})
    model_payload = dict(model_payload)
    provider = model_payload.get('provider')
    # Only inject the default key env when the key is absent from the config entirely.
    # An explicit null preserves the intent to use provider-default credentials (e.g. ADC).
    if 'api_key_env' not in model_payload:
        default_api_key_env = _default_api_key_env(provider)
        if default_api_key_env is not None:
            model_payload['api_key_env'] = default_api_key_env
    normalized['model'] = model_payload
    return normalized


def _write_config_snapshot(config: DatasetBuildConfig | DatasetPrepareConfig | RunConfig, path: Path) -> None:
    path.write_text(json.dumps(config.model_dump(mode='json'), indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _set_task_param(payload: dict[str, object], key: str, value: object) -> None:
    task_params = cast(dict[str, object], payload.get('task_params') or {})
    task_params = dict(task_params)
    task_params[key] = value
    payload['task_params'] = task_params


def _apply_shared_cli_overrides(
    payload: dict[str, object],
    *,
    limit: int | None = None,
    task: str | None = None,
    video_id: str | None = None,
) -> None:
    if limit is not None:
        payload['limit'] = limit
    if task is not None:
        payload['task'] = task
    if video_id is not None:
        _set_task_param(payload, 'video_id', video_id)


def _apply_model_overrides(
    payload: dict[str, object],
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> None:
    if provider is None and model is None and base_url is None:
        return

    model_payload = cast(dict[str, object], payload['model'])
    model_payload = dict(model_payload)
    previous_provider = model_payload.get('provider')
    if provider is not None:
        model_payload['provider'] = provider
    if model is not None:
        model_payload['model'] = model
    if base_url is not None:
        model_payload['base_url'] = base_url
    if provider is not None:
        current_api_key_env = model_payload.get('api_key_env')
        if current_api_key_env == _default_api_key_env(previous_provider):
            model_payload['api_key_env'] = _default_api_key_env(model_payload.get('provider'))
    payload['model'] = model_payload


def load_run_config(config_path: Path) -> RunConfig:
    """Load and validate a RunConfig from a JSON file, injecting provider defaults."""
    payload = _load_json_config(config_path)
    payload = _normalize_loaded_payload(payload)
    return RunConfig.model_validate(payload)


def load_dataset_prepare_config(config_path: Path) -> DatasetPrepareConfig:
    """Load and validate a DatasetPrepareConfig from a JSON file, injecting provider defaults."""
    payload = _load_json_config(config_path)
    payload = _normalize_loaded_payload(payload)
    return DatasetPrepareConfig.model_validate(payload)


def load_dataset_build_config(config_path: Path) -> DatasetBuildConfig:
    """Load and validate a DatasetBuildConfig from a JSON file (no model, no normalization)."""
    payload = _load_json_config(config_path)
    return DatasetBuildConfig.model_validate(payload)


def write_config_snapshot(config: DatasetBuildConfig | DatasetPrepareConfig | RunConfig, path: Path) -> None:
    """Serialize *config* to *path* as pretty-printed JSON (sorted keys)."""
    _write_config_snapshot(config, path)


def apply_build_cli_overrides(
    config: DatasetBuildConfig,
    *,
    limit: int | None = None,
    task: str | None = None,
    video_id: str | None = None,
    shuffle_seed: int | None | object = _UNSET,
    sample_fraction: float | None = None,
    activity_filter: bool | object = _UNSET,
) -> DatasetBuildConfig:
    """Apply CLI-level overrides to a DatasetBuildConfig and return a new validated instance."""
    payload = config.model_dump(mode='python')
    _apply_shared_cli_overrides(payload, limit=limit, task=task, video_id=video_id)
    if shuffle_seed is not _UNSET:
        payload['shuffle_seed'] = shuffle_seed
    if sample_fraction is not None:
        _set_task_param(payload, 'sample_fraction', sample_fraction)
    if activity_filter is not _UNSET:
        _set_task_param(payload, 'activity_filter', activity_filter)
    return DatasetBuildConfig.model_validate(payload)


def apply_prepare_cli_overrides(
    config: DatasetPrepareConfig,
    *,
    limit: int | None = None,
    task: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    out_dir: Path | None = None,
    video_id: str | None = None,
    concurrency: int | None = None,
) -> DatasetPrepareConfig:
    """Apply CLI-level overrides to a DatasetPrepareConfig and return a new validated instance."""
    payload = config.model_dump(mode='python')
    _apply_shared_cli_overrides(payload, limit=limit, task=task, video_id=video_id)
    if out_dir is not None:
        payload['output_dir'] = out_dir
    if concurrency is not None:
        payload['concurrency'] = concurrency
    _apply_model_overrides(payload, provider=provider, model=model, base_url=base_url)
    return DatasetPrepareConfig.model_validate(payload)


def apply_cli_overrides(
    config: RunConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    out_dir: Path | None = None,
    concurrency: int | None = None,
    debug_samples: int | None = None,
) -> RunConfig:
    """Apply CLI-level overrides to a RunConfig and return a new validated instance.

    Only non-None values override the existing config fields.
    Model overrides also update ``api_key_env`` when it was previously the provider default.
    """
    payload = config.model_dump(mode='python')
    if debug_samples is not None:
        payload['debug_samples'] = debug_samples
    if out_dir is not None:
        payload['output_dir'] = out_dir
    if concurrency is not None:
        payload['concurrency'] = concurrency
    _apply_model_overrides(payload, provider=provider, model=model, base_url=base_url)
    return RunConfig.model_validate(payload)
