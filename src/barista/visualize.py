from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .dataset import FrameAnnotation, Video


@dataclass(frozen=True)
class RenderOptions:
    show_bboxes: bool = True
    show_masks: bool = True
    max_objects: int | None = None
    alpha: float = 0.35


def _color_for_category(category_id: uuid.UUID) -> tuple[float, float, float]:
    rng = np.random.default_rng(category_id.int % (2**32))
    return tuple(rng.random(3).tolist())


def render_frame(
    video: Video,
    frame: FrameAnnotation,
    options: RenderOptions,
) -> plt.Figure:
    frame_img = video.frame_array(frame.frame_index, mode='RGB')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(frame_img)
    ax.axis('off')

    for idx, obj in enumerate(frame.objects):
        if options.max_objects is not None and idx >= options.max_objects:
            break
        category = obj.category
        color = _color_for_category(category.id)
        label = category.name

        if options.show_masks:
            mask = obj.mask_array(frame.height, frame.width)
            if mask is not None:
                rgba = np.zeros((frame.height, frame.width, 4))
                rgba[..., 0] = color[0]
                rgba[..., 1] = color[1]
                rgba[..., 2] = color[2]
                rgba[..., 3] = mask * options.alpha
                ax.imshow(rgba)

        if options.show_bboxes:
            x, y, w, h = obj.bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                label,
                color='white',
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=1),
            )

    return fig


def render_frame_array(
    video: Video,
    frame: FrameAnnotation,
    options: RenderOptions,
) -> np.ndarray:
    fig = render_frame(video, frame, options)
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    frame = buffer[:, :, :3].copy()
    plt.close(fig)
    return frame


def visualize_frame(
    video: Video,
    frame_index: int,
    options: RenderOptions | None = None,
    out_path: Path | None = None,
) -> Path:
    if options is None:
        options = RenderOptions()
    frame = video.frames[frame_index]
    fig = render_frame(video, frame, options)
    if out_path is None:
        out_path = Path.cwd() / f'{video.video_id}_{frame.frame_index}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_path


def visualize_segment(
    video: Video,
    start_frame: int,
    end_frame: int,
    step: int = 1,
    options: RenderOptions | None = None,
    out_path: Path | None = None,
    fps: int = 6,
) -> Path:
    if options is None:
        options = RenderOptions()

    frames: list[np.ndarray] = []
    for frame_index in range(start_frame, end_frame + 1, step):
        frame = video.frames[frame_index]
        rendered = render_frame_array(video, frame, options)
        frames.append(rendered)
    if out_path is None:
        out_path = Path.cwd() / f'{video.video_id}_{start_frame}_{end_frame}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == '.gif':
        imageio.mimsave(out_path, frames, fps=fps)
    else:
        with imageio.get_writer(out_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    return out_path
