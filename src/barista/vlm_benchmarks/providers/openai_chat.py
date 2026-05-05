from __future__ import annotations

import os
import threading
import time

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from barista.vlm_benchmarks.providers.base import image_bytes_data_url, resolve_api_key
from barista.vlm_benchmarks.types import ImagePart, MessagePart, ModelResponse, ModelSpec, RunConfig, TextPart


def _extract_openai_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
            elif hasattr(item, 'type') and getattr(item, 'type') == 'text':
                parts.append(str(getattr(item, 'text', '')))
        return '\n'.join(part for part in parts if part)
    return str(content)


def _normalize_openai_usage(usage: object) -> dict[str, int] | None:
    if usage is None:
        return None
    prompt_tokens = getattr(usage, 'prompt_tokens', None)
    completion_tokens = getattr(usage, 'completion_tokens', None)
    total_tokens = getattr(usage, 'total_tokens', None)
    details = getattr(usage, 'completion_tokens_details', None)
    reasoning_tokens = getattr(details, 'reasoning_tokens', None) if details is not None else None
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        total_tokens = usage.get('total_tokens')
        details = usage.get('completion_tokens_details', {})
        reasoning_tokens = details.get('reasoning_tokens') if isinstance(details, dict) else None
    thinking = int(reasoning_tokens or 0)
    output = int(completion_tokens or 0) - thinking
    result = {
        'input_tokens': int(prompt_tokens or 0),
        'output_tokens': output,
        'total_tokens': int(total_tokens or 0),
    }
    if thinking:
        result['thinking_tokens'] = thinking
    if result['input_tokens'] == 0 and result['output_tokens'] == 0 and result['total_tokens'] == 0:
        return None
    return result


def _openai_response_format(schema: type, *, strict: bool = True) -> dict[str, object]:
    """Build an OpenAI ``response_format`` dict from a Pydantic model or type.

    When *strict* is True (default for ``openai`` / ``azure_openai``), the
    schema is cleaned for OpenAI strict mode.  For ``openai_compat`` (vLLM /
    SGLang), strict is disabled since not all servers support it.
    """

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
    else:
        raise TypeError(f'response_schema must be a Pydantic BaseModel subclass, got {schema}')

    if strict:
        _clean_schema(json_schema)

    return {
        'type': 'json_schema',
        'json_schema': {
            'name': schema.__name__,
            'schema': json_schema,
            'strict': strict,
        },
    }


def _clean_schema(schema: dict) -> None:
    """Recursively prepare a JSON schema for OpenAI strict mode.

    OpenAI strict mode requires ``additionalProperties: false`` on every
    object and does not allow ``title`` or ``default`` keywords.
    """
    schema.pop('title', None)
    schema.pop('default', None)

    if schema.get('type') == 'object':
        schema['additionalProperties'] = False

    for key in ('items', 'additionalProperties'):
        sub = schema.get(key)
        if isinstance(sub, dict):
            _clean_schema(sub)

    props = schema.get('properties')
    if isinstance(props, dict):
        for prop in props.values():
            _clean_schema(prop)

    defs = schema.get('$defs')
    if isinstance(defs, dict):
        for d in defs.values():
            _clean_schema(d)


class OpenAIChatProviderClient:
    """OpenAI / Azure OpenAI / OpenAI-compatible provider client.

    The SDK client is created once during ``__init__`` and reused across
    ``generate`` calls.
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        self._provider = model_spec.provider
        api_key = resolve_api_key(model_spec.provider, model_spec.api_key_env)
        if model_spec.provider == 'azure_openai':
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=model_spec.base_url,
                api_version=model_spec.api_version,
                timeout=model_spec.timeout_sec,
                max_retries=model_spec.max_retries,
            )
        else:
            self._client = OpenAI(
                **({'api_key': api_key} if api_key is not None else {}),
                base_url=model_spec.base_url,
                timeout=model_spec.timeout_sec,
            )
        self._dp_size = 0
        self._next_dp_rank = 0
        self._dp_lock = threading.Lock()
        self._dp_thread_local = threading.local()
        if model_spec.provider == 'openai_compat':
            try:
                self._dp_size = max(0, int(os.environ.get('BARISTA_VLLM_DP_SIZE', '0')))
            except ValueError:
                self._dp_size = 0

    def _build_messages(self, system_prompt: str, parts: list[MessagePart]) -> list[dict[str, object]]:
        content_blocks: list[dict[str, object]] = []
        for part in parts:
            if isinstance(part, TextPart):
                content_blocks.append({'type': 'text', 'text': part.text})
            elif isinstance(part, ImagePart):
                url = image_bytes_data_url(part.data, part.mime_type)
                content_blocks.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': url},
                    }
                )
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content_blocks},
        ]

    def generate(
        self,
        system_prompt: str,
        parts: list[MessagePart],
        config: RunConfig,
        *,
        response_schema: type | None = None,
    ) -> ModelResponse:
        messages = self._build_messages(system_prompt, parts)
        request: dict[str, object] = {
            'model': config.model.model,
            'messages': messages,
        }
        if config.model.provider == 'openai_compat':
            request['extra_body'] = {'enable_thinking': True}
            request['temperature'] = config.model.temperature
        elif config.model.thinking_level is not None:
            request['reasoning_effort'] = config.model.thinking_level
        else:
            request['temperature'] = config.model.temperature
        if response_schema is not None:
            request['response_format'] = _openai_response_format(
                response_schema,
                strict=self._provider != 'openai_compat',
            )
        if self._provider == 'openai_compat' and self._dp_size > 1:
            dp_rank = getattr(self._dp_thread_local, 'rank', None)
            if dp_rank is None:
                with self._dp_lock:
                    dp_rank = self._next_dp_rank
                    self._next_dp_rank = (self._next_dp_rank + 1) % self._dp_size
                self._dp_thread_local.rank = dp_rank
            request['extra_headers'] = {'X-data-parallel-rank': str(dp_rank)}

        started = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._normalize_response(response, latency_ms=latency_ms)

    def _normalize_response(self, response: object, *, latency_ms: float) -> ModelResponse:
        choices = getattr(response, 'choices', None)
        if not choices:
            response_id = getattr(response, 'id', None)
            suffix = f' for response {response_id}' if response_id else ''
            raise RuntimeError(f'OpenAI-compatible provider returned no choices{suffix}')
        first_choice = choices[0]
        message = getattr(first_choice, 'message')
        raw_text = _extract_openai_text(getattr(message, 'content', ''))
        usage = _normalize_openai_usage(getattr(response, 'usage', None))
        return ModelResponse(
            raw_text=raw_text,
            latency_ms=latency_ms,
            usage=usage,
            provider_response_id=getattr(response, 'id', None),
            finish_reason=getattr(first_choice, 'finish_reason', None),
        )
