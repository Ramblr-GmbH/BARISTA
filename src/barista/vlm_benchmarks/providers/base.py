from __future__ import annotations

import base64
import os
from typing import Protocol

from barista.vlm_benchmarks.types import MessagePart, ModelResponse, RunConfig


class VlmProviderClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        parts: list[MessagePart],
        config: RunConfig,
        *,
        response_schema: type | None = None,
    ) -> ModelResponse:
        """Send a prompt to the model and return its response.

        *system_prompt* is the system-level instruction.  *parts* is a
        pre-rendered sequence of ``TextPart`` and ``ImagePart`` objects
        produced by the task.  Implementations must convert these into
        the provider SDK's native format.

        *response_schema*, when provided, is a Pydantic model or Python type
        for structured output.  Providers that support it should constrain the
        model response to match this schema.
        """
        ...


def image_bytes_data_url(data: bytes, mime_type: str) -> str:
    payload = base64.b64encode(data).decode('ascii')
    return f'data:{mime_type};base64,{payload}'


def resolve_api_key(provider: str, api_key_env: str | None) -> str | None:
    """Return the API key string, or None to use provider-default credentials (e.g. ADC)."""
    if api_key_env is not None and api_key_env in os.environ:
        return os.environ[api_key_env]
    if provider == 'openai_compat':
        return 'dummy'
    if api_key_env is None:
        return None  # caller should use provider default credentials (e.g. gcloud ADC)
    return os.environ[api_key_env]  # raises KeyError if not set
