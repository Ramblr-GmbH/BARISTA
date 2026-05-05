"""Reusable deepeval G-Eval integration for VLM benchmarks.

Provides an LLM adapter that bridges our :class:`VlmProviderClient` to
deepeval's expected interface, a structured criterion specification, and
helpers to run one or many G-Eval criteria against a reference/prediction pair.

"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

from deepeval.metrics import GEval as _GEval
from deepeval.metrics.g_eval.utils import Rubric as _Rubric
from deepeval.models import DeepEvalBaseLLM as _BaseLLM
from deepeval.test_case import LLMTestCase as _LLMTestCase
from deepeval.test_case import LLMTestCaseParams as _LLMTestCaseParams
from pydantic import BaseModel

from barista.vlm_benchmarks.providers.base import VlmProviderClient
from barista.vlm_benchmarks.types import RunConfig, TextPart

logger = logging.getLogger(__name__)


# ── Criterion specification ───────────────────────────────────────────────────


class GEvalCriterionSpec(TypedDict, total=False):
    """Schema for a single G-Eval evaluation criterion."""

    name: str  # required
    criteria: str  # required
    evaluation_steps: list[str]
    rubric: list[dict[str, object]]  # list of {score_range, expected_outcome}


# ── deepeval LLM adapter ─────────────────────────────────────────────────────


class DeepEvalLLMAdapter(_BaseLLM):  # type: ignore[misc]
    """Wrap a :class:`VlmProviderClient` for use as a deepeval judge LLM.

    deepeval's ``GEval`` accepts any object implementing the
    ``DeepEvalBaseLLM`` interface.  We lazily import deepeval so that the rest
    of the codebase remains importable even without the package installed.
    """

    def __init__(self, client: VlmProviderClient, config: RunConfig) -> None:
        self._client = client
        self._config = config

    def load_model(self) -> VlmProviderClient:  # ty:ignore[invalid-method-override]
        """Called internally by deepeval before ``generate()``."""
        return self._client

    def generate(self, prompt: str, schema: BaseModel | None = None) -> tuple[str, float]:  # ty:ignore[invalid-method-override]
        """Synchronous generation.  Returns ``(response_text, cost)``."""
        if schema is not None:
            prompt = (
                f'{prompt}\n\nRespond ONLY with a valid JSON object matching the schema: {schema.model_json_schema()}'
            )
        parts: list[TextPart] = [TextPart(text=prompt)]
        response = self._client.generate('', parts, self._config)
        return response.raw_text, 0.0

    async def a_generate(self, prompt: str, schema: BaseModel | None = None) -> tuple[str, float]:  # ty:ignore[invalid-method-override]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, schema))

    def get_model_name(self) -> str:
        return self._config.model.model


# ── Single-criterion runner ───────────────────────────────────────────────────


def run_geval_criterion(
    *,
    criterion_key: str,
    spec: GEvalCriterionSpec,
    reference: str,
    prediction: str,
    geval_llm: DeepEvalLLMAdapter,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    context: dict[str, object] | None = None,
) -> tuple[float | None, str]:
    """Run a single G-Eval criterion and return ``(score, reason)``.

    Parameters
    ----------
    criterion_key:
        Short identifier for logging (e.g. ``'semantic_accuracy'``).
    spec:
        The criterion definition (name, criteria text, evaluation steps).
    reference:
        Ground-truth text to evaluate against.
    prediction:
        Model-generated text to evaluate.
    geval_llm:
        A :class:`DeepEvalLLMAdapter` (or compatible) object.
    strict_mode:
        Enable deepeval strict-mode (score is 0 or 1).
    verbose_mode:
        Pass ``verbose_mode=True`` to deepeval.

    Returns
    -------
    tuple[float | None, str]
        ``(score, reason)`` on success, or ``(None, error_message)`` on failure.
    """

    if context and 'question' in context:
        test_input = str(context['question'])
    else:
        test_input = (
            f'Reference: {reference}\n'
            f'Prediction: {prediction}\n'
            'Evaluate the prediction against the reference according to the criterion.'
        )

    test_case = _LLMTestCase(
        input=test_input,
        actual_output=prediction,
        expected_output=reference,
    )

    rubric_raw = spec.get('rubric')
    rubric = None
    if rubric_raw:
        rubric = [
            _Rubric(score_range=tuple(r['score_range']), expected_outcome=str(r['expected_outcome']))
            for r in rubric_raw
        ]

    eval_steps = list(spec['evaluation_steps']) if spec.get('evaluation_steps') else None

    metric = _GEval(
        name=str(spec['name']),
        criteria=str(spec['criteria']),
        evaluation_steps=eval_steps,
        rubric=rubric,
        evaluation_params=[
            _LLMTestCaseParams.INPUT,
            _LLMTestCaseParams.ACTUAL_OUTPUT,
            _LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=geval_llm,
        strict_mode=strict_mode,
        verbose_mode=verbose_mode,
    )

    try:
        metric.measure(test_case)
        score: float = float(metric.score)
        reason: str = str(metric.reason) if metric.reason else ''
        return score, reason
    except Exception as exc:
        logger.warning('G-Eval criterion %r failed: %s', criterion_key, exc)
        return None, f'geval_error: {exc}'


# ── Multi-criterion convenience helper ────────────────────────────────────────


def run_geval_criteria(
    *,
    criteria: dict[str, GEvalCriterionSpec],
    reference: str,
    prediction: str,
    geval_llm: DeepEvalLLMAdapter,
    active_criteria: list[str] | None = None,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    context: dict[str, object] | None = None,
) -> dict[str, tuple[float | None, str]]:
    """Run multiple G-Eval criteria and return per-criterion results.

    Parameters
    ----------
    criteria:
        Mapping of criterion key to its :class:`GEvalCriterionSpec`.
    reference:
        Ground-truth text.
    prediction:
        Model-generated text.
    geval_llm:
        A :class:`DeepEvalLLMAdapter` (or compatible) object.
    active_criteria:
        Subset of *criteria* keys to evaluate.  ``None`` means all.
    strict_mode:
        Enable deepeval strict-mode.
    verbose_mode:
        Pass ``verbose_mode=True`` to deepeval.

    Returns
    -------
    dict[str, tuple[float | None, str]]
        ``{criterion_key: (score, reason)}`` for each evaluated criterion.
    """
    keys = active_criteria if active_criteria is not None else list(criteria.keys())
    results: dict[str, tuple[float | None, str]] = {}
    for key in keys:
        spec = criteria.get(key)
        if spec is None:
            logger.warning('Unknown G-Eval criterion %r; skipping.', key)
            results[key] = (None, f'unknown_criterion: {key}')
            continue
        results[key] = run_geval_criterion(
            criterion_key=key,
            spec=spec,
            reference=reference,
            prediction=prediction,
            geval_llm=geval_llm,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            context=context,
        )
    return results


__all__ = [
    'DeepEvalLLMAdapter',
    'GEvalCriterionSpec',
    'run_geval_criteria',
    'run_geval_criterion',
]
