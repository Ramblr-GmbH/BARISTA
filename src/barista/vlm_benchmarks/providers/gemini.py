from __future__ import annotations

import time

from google import genai
from google.genai import types as genai_types

from barista.vlm_benchmarks.providers.base import resolve_api_key
from barista.vlm_benchmarks.types import ImagePart, MessagePart, ModelResponse, ModelSpec, RunConfig, TextPart


def _extract_gemini_text(response: object) -> str:
    text = getattr(response, 'text', None)
    if isinstance(text, str) and text:
        return text

    candidates = getattr(response, 'candidates', None)
    if candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, 'content', None)
        parts = getattr(content, 'parts', None) if content is not None else None
        if parts:
            lines: list[str] = []
            for part in parts:
                if hasattr(part, 'text') and getattr(part, 'text'):
                    lines.append(str(getattr(part, 'text')))
                elif isinstance(part, dict) and part.get('text'):
                    lines.append(str(part['text']))
            return '\n'.join(lines)
    return str(response)


def _extract_gemini_finish_reason(response: object) -> str | None:
    candidates = getattr(response, 'candidates', None)
    if not candidates:
        return None
    first_candidate = candidates[0]
    finish_reason = getattr(first_candidate, 'finish_reason', None)
    if finish_reason is None:
        return None
    return str(finish_reason)


def _normalize_gemini_usage(usage_metadata: object) -> dict[str, int] | None:
    if usage_metadata is None:
        return None
    prompt_tokens = getattr(usage_metadata, 'prompt_token_count', None)
    completion_tokens = getattr(usage_metadata, 'candidates_token_count', None)
    total_tokens = getattr(usage_metadata, 'total_token_count', None)
    thoughts_tokens = getattr(usage_metadata, 'thoughts_token_count', None)
    if prompt_tokens is None and isinstance(usage_metadata, dict):
        prompt_tokens = usage_metadata.get('prompt_token_count')
        completion_tokens = usage_metadata.get('candidates_token_count')
        total_tokens = usage_metadata.get('total_token_count')
        thoughts_tokens = usage_metadata.get('thoughts_token_count')
    result = {
        'input_tokens': int(prompt_tokens or 0),
        'output_tokens': int(completion_tokens or 0),
        'total_tokens': int(total_tokens or 0),
    }
    if thoughts_tokens:
        result['thinking_tokens'] = int(thoughts_tokens)
    if result['input_tokens'] == 0 and result['output_tokens'] == 0 and result['total_tokens'] == 0:
        return None
    return result


class GeminiProviderClient:
    """Gemini / Vertex AI provider client.

    The SDK client is created once during ``__init__`` and reused across
    ``generate`` calls.  The system prompt is passed via the SDK's
    ``system_instruction`` parameter rather than being inlined into user
    content.
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        api_key = resolve_api_key(model_spec.provider, model_spec.api_key_env)
        if model_spec.vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=model_spec.project,
                location=model_spec.location,
            )
        elif api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            # google-genai >= 1.x requires explicit credentials; fall back to
            # Vertex AI with Application Default Credentials (gcloud login).
            self._client = genai.Client(vertexai=True)

    def generate(
        self,
        system_prompt: str,
        parts: list[MessagePart],
        config: RunConfig,
        *,
        response_schema: type | None = None,
    ) -> ModelResponse:
        sdk_parts = []
        for part in parts:
            if isinstance(part, TextPart):
                sdk_parts.append(genai_types.Part.from_text(text=part.text))
            elif isinstance(part, ImagePart):
                sdk_parts.append(genai_types.Part.from_bytes(data=part.data, mime_type=part.mime_type))

        config_kwargs: dict[str, object] = {'temperature': config.model.temperature}
        if config.model.max_output_tokens is not None:
            config_kwargs['max_output_tokens'] = config.model.max_output_tokens
        if config.model.thinking_level is not None:
            config_kwargs['thinking_config'] = genai_types.ThinkingConfig(
                thinking_level=config.model.thinking_level.upper()
            )
        if system_prompt:
            config_kwargs['system_instruction'] = system_prompt
        if response_schema is not None:
            config_kwargs['response_mime_type'] = 'application/json'
            config_kwargs['response_schema'] = response_schema

        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=config.model.model,
            contents=sdk_parts,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._normalize_response(response, latency_ms=latency_ms)

    def _normalize_response(self, response: object, *, latency_ms: float) -> ModelResponse:
        raw_text = _extract_gemini_text(response)
        usage = _normalize_gemini_usage(getattr(response, 'usage_metadata', None))
        return ModelResponse(
            raw_text=raw_text,
            latency_ms=latency_ms,
            usage=usage,
            provider_response_id=getattr(response, 'response_id', None) or getattr(response, 'id', None),
            finish_reason=_extract_gemini_finish_reason(response),
        )
