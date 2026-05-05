from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from barista.vlm_benchmarks.types import AssetPreparationResult, BenchmarkExample, MessagePart, PredictionResult


class BenchmarkExampleBuilder(ABC):
    """Base class for components that materialize benchmark examples."""

    @abstractmethod
    def iter_examples(self) -> list[BenchmarkExample]:
        """Generate all benchmark examples for this task from the dataset."""
        ...

    def validate_example(self, example: BenchmarkExample) -> None:
        """Validate that an example's ``task_data`` is well-formed.

        Override to add task-specific validation.  The default is a no-op.
        """


class BenchmarkAssetPreparer(ABC):
    """Base class for components that prepare task-specific assets."""

    @abstractmethod
    def prepare_assets(self) -> AssetPreparationResult:
        """Prepare task-specific assets needed before build/run."""
        ...


class BenchmarkTask(BenchmarkExampleBuilder, ABC):
    """Base class for runnable VLM benchmark tasks.

    Each concrete task bundles its own example generation, prompt rendering,
    response parsing, evaluation logic, and metrics aggregation.  Implement
    all abstract methods and register the task in ``tasks/__init__.py``.
    """

    @abstractmethod
    def default_system_prompt(self) -> str:
        """Return the default system prompt when none is specified in the run config."""
        ...

    @abstractmethod
    def render_prompt(self, example: BenchmarkExample) -> list[MessagePart]:
        """Convert a benchmark example into provider-agnostic message parts.

        The returned list of ``TextPart`` and ``ImagePart`` objects is handed
        directly to the provider client.  Frame images should be interleaved
        with text as appropriate for the task.
        """
        ...

    @abstractmethod
    def parse_response(self, example: BenchmarkExample, raw_text: str) -> dict[str, object] | None:
        """Extract a structured task result from the model's raw text response.

        Return ``None`` when the response cannot be parsed; the runner will
        record a parse-failure error for it.
        """
        ...

    @abstractmethod
    def evaluate(self, example: BenchmarkExample, task_result: dict[str, object]) -> dict[str, object]:
        """Score a parsed prediction against the ground truth in the example.

        Return an enriched copy of *task_result* with evaluation fields
        (e.g. ``is_correct``) added.
        """
        ...

    def response_schema(self) -> type | None:
        """Return a Pydantic model or Python type for structured output.

        When non-None, providers that support structured output (e.g. Gemini)
        will constrain the model to return JSON matching this schema.
        The default is ``None`` (free-text output).
        """
        return None

    @abstractmethod
    def init_metrics(self, *, examples_total: int) -> dict[str, object]:
        """Create the initial metrics accumulator for a benchmark run."""
        ...

    @abstractmethod
    def update_metrics(self, metrics: dict[str, object], prediction: PredictionResult) -> None:
        """Update the metrics accumulator with a single prediction result."""
        ...

    @abstractmethod
    def finalize_metrics(self, metrics: dict[str, object]) -> None:
        """Compute derived metrics (e.g. accuracy) after all predictions are processed."""
        ...

    @abstractmethod
    def format_summary(self, metrics: dict[str, object], *, run_dir: Path) -> str:
        """Format final metrics into a human-readable summary string."""
        ...
