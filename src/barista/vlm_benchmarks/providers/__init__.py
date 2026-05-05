"""Provider adapters for VLM benchmark execution."""

from barista.vlm_benchmarks.providers.base import VlmProviderClient
from barista.vlm_benchmarks.providers.gemini import GeminiProviderClient
from barista.vlm_benchmarks.providers.openai_chat import OpenAIChatProviderClient
from barista.vlm_benchmarks.types import ModelSpec


def create_provider_client(model_spec: ModelSpec) -> VlmProviderClient:
    if model_spec.provider in {'openai', 'openai_compat', 'azure_openai'}:
        return OpenAIChatProviderClient(model_spec)
    if model_spec.provider == 'gemini':
        return GeminiProviderClient(model_spec)
    raise ValueError(f'Unsupported provider: {model_spec.provider}')


__all__ = [
    'VlmProviderClient',
    'create_provider_client',
    'GeminiProviderClient',
    'OpenAIChatProviderClient',
]
