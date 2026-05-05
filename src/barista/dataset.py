"""Dataset model for the Barista COCO-style video annotations.

Layout expected on disk::

    <root>/
        <video_id>/
            coco_annotation.json
            video.mp4

The top-level entry points are:

- :func:`load_videos` — parse every video under a root directory.
- :func:`discover_video_dirs` — enumerate ``(video_dir, video_id)`` pairs
  without parsing any JSON.
- :func:`discover_light_videos` — discover videos while reading *only* frame
  counts; skips annotation parsing entirely.
- :class:`Video` — parsed representation of one video (annotations included).
- :class:`LightVideo` — lightweight stand-in that avoids annotation parsing.
"""

from __future__ import annotations

import collections
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class VideoMetadata:
    """Recording-level metadata extracted from video_metadata."""

    recording_device_type: str = ''
    recording_device_version: str = ''


def decode_segmentation(segmentation: object, height: int, width: int) -> np.ndarray | None:
    """Decode a COCO segmentation (RLE dict or polygon list) to a boolean mask array."""
    if isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
        rle = dict(segmentation)
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        mask = mask_utils.decode(rle)
        return mask if mask.ndim == 2 else np.any(mask, axis=2)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rles)
        return mask if mask.ndim == 2 else np.any(mask, axis=2)
    return None


@dataclass(frozen=True)
class Category:
    """Object category as stored in the COCO ``categories`` list."""

    name: str
    id: uuid.UUID


@dataclass(frozen=True)
class Attribute:
    """A key-value attribute annotation attached to an object on a frame range."""

    attribute_type: str
    value: str

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            'attribute_type': self.attribute_type,
            'value': self.value,
        }


@dataclass(frozen=True)
class ObjectAnnotation:
    """A single object instance annotation on one frame.

    ``bbox`` is in COCO ``[x, y, w, h]`` format.  ``mask`` is the raw
    segmentation field from the JSON (RLE or polygon list); use
    :meth:`mask_array` to decode it.
    """

    category: Category
    bbox: list[float]
    mask: object
    attributes: list[Attribute]
    object_id: uuid.UUID

    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def bbox_normalized(self, frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return (x / frame_width, y / frame_height, w / frame_width, h / frame_height)

    def bbox_area(self) -> float:
        _, _, w, h = self.bbox
        return w * h

    def mask_array(self, frame_height: int, frame_width: int) -> np.ndarray | None:
        return decode_segmentation(self.mask, frame_height, frame_width)

    def to_prompt_dict(self, frame_width: int, frame_height: int) -> dict[str, object]:
        return {
            'object_id': str(self.object_id),
            'category_id': str(self.category.id),
            'category_name': self.category.name,
            'bbox_xywh': self.bbox,
            'bbox_xyxy': self.bbox_xyxy(),
            'bbox_normalized': self.bbox_normalized(frame_width, frame_height),
            'bbox_area': self.bbox_area(),
            'attributes': [attr.to_prompt_dict() for attr in self.attributes],
        }


@dataclass(frozen=True)
class Relation:
    """A directed relation between two object instances, valid over a frame range."""

    source_object_id: uuid.UUID
    target_object_id: uuid.UUID
    relation_type: str
    value: str

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            'source_object_id': str(self.source_object_id),
            'target_object_id': str(self.target_object_id),
            'relation_type': self.relation_type,
            'value': self.value,
        }


@dataclass(frozen=True)
class Activity:
    """An activity segment covering a contiguous range of frames."""

    frame_start: int
    frame_end: int
    activity_class_id: uuid.UUID
    display_name: str
    verb: str
    noun: str

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'activity_class_id': str(self.activity_class_id),
            'display_name': self.display_name,
        }


@dataclass(frozen=True)
class ProcessStep:
    """A process step segment covering a contiguous range of frames."""

    frame_start: int
    frame_end: int
    activity_class_id: uuid.UUID
    display_name: str


@dataclass
class FrameAnnotation:
    """All annotations for a single video frame."""

    frame_index: int
    width: int
    height: int
    objects: list[ObjectAnnotation]
    relations: list[Relation]

    def objects_by_id(self) -> dict[uuid.UUID, ObjectAnnotation]:
        return {obj.object_id: obj for obj in self.objects}

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            'frame_index': self.frame_index,
            'width': self.width,
            'height': self.height,
            'objects': [obj.to_prompt_dict(self.width, self.height) for obj in self.objects],
            'relations': [rel.to_prompt_dict() for rel in self.relations],
        }


def _load_activities_from_data(data: dict, coco_path: Path) -> list[Activity]:
    """Parse activity annotations from a loaded COCO data dict."""
    activities: list[Activity] = []
    for activity in data.get('activities', []):
        image_range = activity['image_range']
        if activity.get('activity_class_id') is None:
            logging.warning(f'Skipping activity with missing activity_class_id in {coco_path}: {activity}')
            continue
        display_name = activity['display_name']
        verb, noun = display_name.split(' ', 1)
        activities.append(
            Activity(
                frame_start=image_range['image_id_start'],
                frame_end=image_range['image_id_end'],
                activity_class_id=uuid.UUID(activity['activity_class_id']),
                display_name=activity.get('display_name', ''),
                verb=verb,
                noun=noun,
            )
        )
    return activities


def _load_process_steps_from_data(data: dict, coco_path: Path) -> list[ProcessStep]:
    """Parse process step annotations from a loaded COCO data dict."""
    steps: list[ProcessStep] = []
    for step in data.get('process_steps', []):
        image_range = step['image_range']
        if step.get('activity_class_id') is None:
            logging.warning(f'Skipping process step with missing activity_class_id in {coco_path}: {step}')
            continue
        steps.append(
            ProcessStep(
                frame_start=image_range['image_id_start'],
                frame_end=image_range['image_id_end'],
                activity_class_id=uuid.UUID(step['activity_class_id']),
                display_name=step.get('display_name', ''),
            )
        )
    return steps


@dataclass
class Video:
    """Fully-parsed representation of one annotated video.
    Instantiate via :meth:`from_dir` or :func:`load_videos`.

    Frame indices start at 0 and run to ``frame_count - 1``.
    """

    video_id: str
    coco_path: Path
    mp4_path: Path
    width: int
    height: int
    fps: float
    length_in_ms: float
    object_id_to_category_id: dict[str, str]
    categories: dict[str, Category]
    frames: dict[int, FrameAnnotation]
    activities: list[Activity]
    process_steps: list[ProcessStep]
    metadata: VideoMetadata

    def _load_mp4_frame(self, frame_index: int) -> Image.Image:
        frame_array = iio.imread(str(self.mp4_path), index=frame_index)
        return Image.fromarray(frame_array)

    def frame(self, frame_index: int) -> FrameAnnotation:
        """Return the :class:`FrameAnnotation` for *frame_index* (KeyError if absent)."""
        return self.frames[frame_index]

    def frame_indices(self) -> list[int]:
        """Return all frame indices in ascending order."""
        return sorted(self.frames)

    def iter_frames(self) -> list[FrameAnnotation]:
        """Return all :class:`FrameAnnotation` objects in frame-index order."""
        return [self.frames[frame_index] for frame_index in self.frame_indices()]

    def activity_frame_indices(self) -> list[int]:
        """Return sorted frame indices that fall within any activity segment."""
        covered: set[int] = set()
        for activity in self.activities:
            for i in range(activity.frame_start, activity.frame_end):
                if i in self.frames:
                    covered.add(i)
        return sorted(covered)

    def iter_activity_frames(self) -> list[FrameAnnotation]:
        """Like ``iter_frames`` but limited to frames inside activity segments."""
        return [self.frames[i] for i in self.activity_frame_indices()]

    def load_frame_image(self, frame_index: int, mode: str = 'RGB') -> Image.Image:
        return self._load_mp4_frame(frame_index).convert(mode)

    def frame_array(self, frame_index: int, mode: str = 'RGB') -> np.ndarray:
        """Return the decoded frame image as a NumPy array."""
        return np.array(self.load_frame_image(frame_index, mode=mode))

    def objects_by_id(self) -> dict[uuid.UUID, ObjectAnnotation]:
        """Return a dict mapping object id to the *last* seen :class:`ObjectAnnotation` across all frames."""
        objects: dict[uuid.UUID, ObjectAnnotation] = {}
        for frame in self.frames.values():
            for obj in frame.objects:
                objects[obj.object_id] = obj
        return objects

    def sample_clips(
        self,
        stride: int,
        clip_length: int,
        frame_spacing: int,
    ) -> list[list[int]]:
        """Return sampled clips of frame indices.

        A new clip starts every *stride* positions in the sorted frame list.
        Within each clip, *clip_length* indices are selected with *frame_spacing*
        steps between them.  Indices beyond the end are silently skipped.
        """
        indices = self.frame_indices()
        clips: list[list[int]] = []
        i = 0
        while i < len(indices):
            clip = [indices[i + j * frame_spacing] for j in range(clip_length) if i + j * frame_spacing < len(indices)]
            if clip:
                clips.append(clip)
            i += stride
        return clips

    @classmethod
    def from_dir(cls, video_dir: Path) -> Video:
        """Parse a video from its directory, loading the COCO annotation JSON."""
        coco_path = video_dir / 'coco_annotation.json'
        with coco_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        categories = {
            c['id']: Category(
                name=c['name'],
                id=uuid.UUID(c['id']),
            )
            for c in data['categories']
        }

        object_id_to_category_id: dict[str, str] = data.get('object_id_to_category_id', {})
        video_metadata_list = data.get('video_metadata', [])

        frames: dict[int, FrameAnnotation] = {}
        vm = video_metadata_list[0]
        width = vm['width']
        height = vm['height']
        for frame_index in range(vm['frame_count']):
            frames[frame_index] = FrameAnnotation(
                frame_index=frame_index,
                width=width,
                height=height,
                objects=[],
                relations=[],
            )

        # Attributes are stored per-object with image_ranges.
        # Build lookup: object_id -> list of (Attribute, image_ranges)
        attributes_by_object: dict[uuid.UUID, list[tuple[Attribute, list[dict]]]] = collections.defaultdict(list)
        for attr in data.get('attributes', []):
            obj_id = uuid.UUID(attr['object_id'])
            attributes_by_object[obj_id].append(
                (
                    Attribute(
                        attribute_type=attr['attribute_type'],
                        value=attr['value'],
                    ),
                    attr['image_ranges'],
                )
            )

        # Track which objects are present on each frame (by image_id)
        objects_on_frame: dict[int, set[uuid.UUID]] = collections.defaultdict(set)
        for ann in data['annotations']:
            frame_index = int(ann['image_id'])
            if 'bbox' not in ann:
                continue
            category_id = object_id_to_category_id[ann['object_id']]
            category = categories[category_id]
            obj_id = uuid.UUID(ann['object_id'])

            frame_attributes = [
                attribute
                for attribute, image_ranges in attributes_by_object.get(obj_id, [])
                if _frame_in_ranges(frame_index, image_ranges)
            ]

            obj = ObjectAnnotation(
                category=category,
                bbox=ann['bbox'],
                mask=ann.get('segmentation'),
                attributes=frame_attributes,
                object_id=obj_id,
            )
            frames[frame_index].objects.append(obj)
            objects_on_frame[frame_index].add(obj_id)

        # Relations are stored per-object-pair with image_ranges.
        for rel in data.get('relations', []):
            src_obj_id = uuid.UUID(rel['source_object_id'])
            tgt_obj_id = uuid.UUID(rel['target_object_id'])
            relation = Relation(
                source_object_id=src_obj_id,
                target_object_id=tgt_obj_id,
                relation_type=rel['relation_type'],
                value=rel['value'],
            )
            for r in rel['image_ranges']:
                for frame_index in range(r['image_id_start'], r['image_id_end']):
                    if frame_index not in frames:
                        continue
                    present = objects_on_frame.get(frame_index, set())
                    if src_obj_id in present and tgt_obj_id in present:
                        frames[frame_index].relations.append(relation)

        activities = _load_activities_from_data(data, coco_path)
        process_steps = _load_process_steps_from_data(data, coco_path)

        mp4_path = video_dir / 'video.mp4'
        return cls(
            video_id=video_dir.name,
            coco_path=coco_path,
            mp4_path=mp4_path,
            width=width,
            height=height,
            fps=float(vm.get('fps', 0.0)),
            length_in_ms=float(vm.get('length_in_ms', 0.0)),
            categories=categories,
            object_id_to_category_id=object_id_to_category_id,
            frames=frames,
            activities=activities,
            process_steps=process_steps,
            metadata=VideoMetadata(
                recording_device_type=vm.get('recording_device_type', ''),
                recording_device_version=vm.get('recording_device_version', ''),
            ),
        )


def _frame_in_ranges(frame_index: int, ranges: list[dict]) -> bool:
    """Return True if *frame_index* falls within any of the image_id ranges."""
    return any(r['image_id_start'] <= frame_index < r['image_id_end'] for r in ranges)


def load_videos(root: Path) -> list[Video]:
    """Parse all videos under *root* and return them as a list of :class:`Video` objects."""
    videos = []
    for video_dir, video_id in discover_video_dirs(root):
        video = Video.from_dir(video_dir)
        video.video_id = video_id
        videos.append(video)
    return videos


def discover_video_dirs(root: Path) -> list[tuple[Path, str]]:
    """Return (video_dir, video_id) pairs without parsing any COCO JSON.
    Expects flat layout: ``root/<video_id>/coco_annotation.json``
    """
    result = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / 'coco_annotation.json').exists():
            continue
        result.append((child, child.name))
    return result


class LightVideo:
    """Lightweight stand-in for :class:`Video` for tasks that do not need full annotation parsing.

    Stores the video id, the set of known frame indices, and the ``mp4_path``.
    Benchmark tasks use this to build
    :class:`~barista.vlm_benchmarks.types.FrameInput` objects without
    loading COCO annotations.
    """

    def __init__(
        self,
        video_id: str,
        frame_indices: frozenset[int],
        mp4_path: Path,
    ) -> None:
        self.video_id = video_id
        self.frame_indices = frame_indices
        self.mp4_path = mp4_path

    def load_frame_image(self, frame_index: int, mode: str = 'RGB') -> Image.Image:
        frame_array = iio.imread(str(self.mp4_path), index=frame_index)
        return Image.fromarray(frame_array).convert(mode)


def discover_light_videos(dataset_root: Path) -> list[LightVideo]:
    """Discover videos under *dataset_root* by reading only the COCO metadata.

    Does **not** parse annotations, attributes, or relations — so it never
    fails on schema mismatches.  Reads ``video_metadata`` for frame counts.
    """
    videos: list[LightVideo] = []
    for video_dir, video_id in discover_video_dirs(dataset_root):
        coco_path = video_dir / 'coco_annotation.json'
        if not coco_path.exists():
            continue
        with coco_path.open(encoding='utf-8') as f:
            data = json.load(f)
        vm_list = data.get('video_metadata', [])
        if not vm_list:
            continue
        frame_indices = set(range(vm_list[0]['frame_count']))
        mp4_path = video_dir / 'video.mp4'
        videos.append(
            LightVideo(
                video_id,
                frozenset(frame_indices),
                mp4_path,
            )
        )
    return videos
