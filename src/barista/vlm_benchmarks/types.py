from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
from PIL import Image
from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class FrameInput:
    video_id: str
    frame_index: int
    mp4_path: Path

    def load_pil_image(self) -> Image.Image:
        """Return the frame as a PIL RGB image decoded from the MP4."""
        frame_array = iio.imread(str(self.mp4_path), index=self.frame_index)
        return Image.fromarray(frame_array).convert('RGB')

    def to_image_part(self) -> ImagePart:
        """Return an :class:`ImagePart` for this frame (inline JPEG bytes)."""
        img = self.load_pil_image()
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return ImagePart(data=buf.getvalue(), mime_type='image/jpeg')


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """Inline image data (JPEG bytes)."""

    data: bytes
    mime_type: str


MessagePart = TextPart | ImagePart


def render_frame_image_parts(frames: list[FrameInput]) -> list[ImagePart]:
    """Convert a list of FrameInputs to ImageParts, opening each MP4 at most once.

    All frames from the same MP4 are decoded in a single open/close cycle,
    avoiding N redundant file opens when an example contains multiple frames
    from the same video.
    """
    result: list[ImagePart | None] = [None] * len(frames)

    mp4_groups: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for i, frame in enumerate(frames):
        mp4_groups[frame.mp4_path].append((i, frame.frame_index))

    for mp4_path, items in mp4_groups.items():
        with iio.imopen(str(mp4_path), 'r') as reader:
            for i, frame_index in items:
                img = Image.fromarray(reader.read(index=frame_index)).convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                result[i] = ImagePart(data=buf.getvalue(), mime_type='image/jpeg')

    return [part for part in result if part is not None]


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    task_name: str
    label: str
    frames: list[FrameInput]
    metadata: dict[str, object] = field(default_factory=dict)
    task_data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        for frame in d['frames']:
            frame['mp4_path'] = str(frame['mp4_path'])  # convert Path to str for JSON serialization
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BenchmarkExample:
        return cls(
            example_id=str(payload['example_id']),
            task_name=str(payload['task_name']),
            label=str(payload.get('label', '')),
            frames=[
                FrameInput(
                    video_id=str(f['video_id']),
                    frame_index=int(f['frame_index']),
                    mp4_path=Path(str(f['mp4_path'])),
                )
                for f in payload['frames']
            ],
            metadata=dict(payload.get('metadata', {})),
            task_data=dict(payload.get('task_data', {})),
        )


@dataclass(frozen=True)
class AssetPreparationResult:
    task_name: str
    artifact_path: Path
    artifacts_total: int
    summary_text: str


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    provider: Literal['openai', 'gemini', 'openai_compat', 'azure_openai']
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_version: str | None = None
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    timeout_sec: float = 180.0
    max_retries: int = 3
    temperature: float = 0.0
    max_output_tokens: int | None = None
    thinking_level: Literal['low', 'medium', 'high', 'minimal'] | None = None


class BenchmarkConfigBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset_root: Path
    task: str
    task_params: dict[str, object]


class DatasetBuildConfig(BenchmarkConfigBase):
    output_dir: Path | None = None
    limit: int | None = None
    shuffle_seed: int | None = None


class DatasetPrepareConfig(BenchmarkConfigBase):
    model: ModelSpec
    output_dir: Path
    concurrency: int = 1
    limit: int | None = None
    shuffle_seed: int | None = None


class RunConfig(BenchmarkConfigBase):
    model: ModelSpec
    output_dir: Path
    concurrency: int = 1
    system_prompt: str = ''
    debug_samples: int = 0


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    latency_ms: float
    usage: dict[str, int] | None
    provider_response_id: str | None
    finish_reason: str | None


@dataclass(frozen=True)
class PredictionResult:
    example_id: str
    task_name: str
    model: str
    provider: str
    raw_text: str
    error: str | None
    latency_ms: float | None
    usage: dict[str, int] | None
    metadata: dict[str, object]
    task_result: dict[str, object] = field(default_factory=dict)
    label: str | None = None
    finish_reason: str | None = None
    provider_response_id: str | None = None
    prompt_text: str | None = None
