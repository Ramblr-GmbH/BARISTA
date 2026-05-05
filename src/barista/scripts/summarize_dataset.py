"""Summarize per-video annotation statistics across the dataset.

Scans every video under the dataset root in parallel, printing a text summary
for each video and optionally exporting a suite of CSV files (one main table
plus several long-format distribution tables) for downstream plotting.

Example (text only)::

    barista-summarize --root /data/barista

Example (CSV export)::

    barista-summarize \
        --root /data/barista \
        --csv --output-dir /tmp/stats_csv
"""

from __future__ import annotations

import argparse
import csv
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from barista.dataset import Category, Video, discover_video_dirs


def _sum_by_first_key(pair_counts: dict[tuple[str, str], int]) -> dict[str, int]:
    """Collapse ``(type, value) -> count`` into ``type -> total_count``."""
    out: dict[str, int] = {}
    for (key, _), c in pair_counts.items():
        out[key] = out.get(key, 0) + c
    return out


@dataclass
class VideoStats:
    """Aggregated annotation statistics for a single video."""

    video_id: str
    categories: dict[str, Category]
    n_frames: int
    n_activities: int
    category_counts: dict[str, int]
    attribute_value_counts: dict[tuple[str, str], int]
    relation_value_counts: dict[tuple[str, str], int]
    object_count: int
    n_unique_objects: int
    relation_count: int
    n_frames_with_objects: int
    n_images_without_annotations: int
    activity_counts: dict[str, int]
    activity_duration_buckets: dict[int, int]  # segment length (frames) -> count
    objects_per_frame_buckets: dict[int, int]  # k objects -> number of frames
    n_process_steps: int
    process_step_counts: dict[str, int]  # process step label -> count
    process_step_duration_buckets: dict[int, int]  # segment length (frames) -> count
    preparation_type: str
    recording_device: str
    resolution: tuple[int, int]
    fps: float
    length_in_ms: float


def _get_video_stats(video: Video) -> VideoStats:
    """Walk every frame and activity in *video* to compute aggregate counts."""
    category_counts = {cat.name: 0 for cat in video.categories.values()}
    attribute_value_counts: dict[tuple[str, str], int] = {}
    relation_value_counts: dict[tuple[str, str], int] = {}

    object_count = 0
    unique_object_ids: set[uuid.UUID] = set()
    relation_count = 0
    n_frames_with_objects = 0
    n_images_without_annotations = 0
    objects_per_frame_buckets: Counter[int] = Counter()

    for frame in video.frames.values():
        objects_per_frame_buckets[len(frame.objects)] += 1
        if frame.objects:
            n_frames_with_objects += 1
        else:
            n_images_without_annotations += 1

        for obj in frame.objects:
            object_count += 1
            unique_object_ids.add(obj.object_id)
            category_counts[obj.category.name] = category_counts.get(obj.category.name, 0) + 1
            for attr in obj.attributes:
                ak = (attr.attribute_type, attr.value)
                attribute_value_counts[ak] = attribute_value_counts.get(ak, 0) + 1
        for rel in frame.relations:
            relation_count += 1
            key = (rel.relation_type, rel.value)
            relation_value_counts[key] = relation_value_counts.get(key, 0) + 1

    activity_counts: dict[str, int] = {}
    activity_duration_buckets: Counter[int] = Counter()
    for act in video.activities:
        label = act.display_name if act.display_name.strip() else str(act.activity_class_id)
        activity_counts[label] = activity_counts.get(label, 0) + 1
        duration_frames = act.frame_end - act.frame_start + 1
        activity_duration_buckets[duration_frames] += 1

    process_step_counts: dict[str, int] = {}
    process_step_duration_buckets: Counter[int] = Counter()
    for step in video.process_steps:
        label = step.display_name if step.display_name.strip() else str(step.activity_class_id)
        process_step_counts[label] = process_step_counts.get(label, 0) + 1
        duration_frames = step.frame_end - step.frame_start + 1
        process_step_duration_buckets[duration_frames] += 1

    preparation_type = ''
    seen_coffee_machine_ids: set[uuid.UUID] = set()
    for frame in video.frames.values():
        for obj in frame.objects:
            if 'coffee machine' in obj.category.name.lower() and obj.object_id not in seen_coffee_machine_ids:
                seen_coffee_machine_ids.add(obj.object_id)
                for attr in obj.attributes:
                    if attr.attribute_type == 'type':
                        preparation_type = attr.value
                        break
            if preparation_type:
                break
        if preparation_type:
            break

    return VideoStats(
        video_id=video.video_id,
        categories=video.categories,
        n_frames=len(video.frames),
        n_activities=len(video.activities),
        category_counts=category_counts,
        attribute_value_counts=attribute_value_counts,
        relation_value_counts=relation_value_counts,
        object_count=object_count,
        n_unique_objects=len(unique_object_ids),
        relation_count=relation_count,
        n_frames_with_objects=n_frames_with_objects,
        n_images_without_annotations=n_images_without_annotations,
        activity_counts=activity_counts,
        activity_duration_buckets=dict(activity_duration_buckets),
        objects_per_frame_buckets=dict(objects_per_frame_buckets),
        n_process_steps=len(video.process_steps),
        process_step_counts=process_step_counts,
        process_step_duration_buckets=dict(process_step_duration_buckets),
        preparation_type=preparation_type,
        recording_device=' '.join(
            p for p in [video.metadata.recording_device_type, video.metadata.recording_device_version] if p
        ),
        resolution=(video.width, video.height),
        fps=video.fps,
        length_in_ms=video.length_in_ms,
    )


def _format_video_stats_summary(stats: VideoStats) -> str:
    """Return a human-readable multi-line summary for one video."""
    cat_id_name = ', '.join(f'{c.id}={c.name}' for c in sorted(stats.categories.values(), key=lambda c: c.id))
    lines = [
        f'Video: {stats.video_id}',
        f'  Images: {stats.n_frames}',
        f'  Objects: {stats.object_count} instances, {stats.n_unique_objects} unique ids',
        f'  Categories: {len(stats.categories)} ({cat_id_name})',
        f'  Relations: {stats.relation_count}',
        f'  Attributes: {sum(stats.attribute_value_counts.values())}',
        f'  Activities: {stats.n_activities}',
        f'  Process steps: {stats.n_process_steps}',
        f'  Frames without annotations: {stats.n_images_without_annotations}',
    ]
    rel_by_type = _sum_by_first_key(stats.relation_value_counts)
    if rel_by_type:
        top_rel = sorted(rel_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append('  Top relations: ' + ', '.join(f'{k}={v_}' for k, v_ in top_rel))
    if stats.attribute_value_counts:
        top_attr = sorted(stats.attribute_value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append('  Top attribute values: ' + ', '.join(f'{t}={v} ({c})' for ((t, v), c) in top_attr))
    if stats.category_counts:
        top = sorted(stats.category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append('  Top categories: ' + ', '.join(f'{k}={v_}' for k, v_ in top))
    if stats.process_step_counts:
        top_process = sorted(stats.process_step_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append('  Top process steps: ' + ', '.join(f'{k}={v_}' for k, v_ in top_process))
    if stats.resolution:
        lines.append(f'  Resolution: {stats.resolution[0]}x{stats.resolution[1]}')
    if stats.fps > 0:
        lines.append(f'  FPS: {stats.fps:.4g}')
    return '\n'.join(lines)


_VIDEOS_CSV_FIELDS: tuple[str, ...] = (
    'video_id',
    'preparation_type',
    'recording_device',
    'width',
    'height',
    'fps',
    'length_in_ms',
    'n_frames',
    'n_frames_with_objects',
    'n_object_instances',
    'n_unique_objects',
    'n_relations',
    'n_attributes',
    'n_activity_segments',
    'n_process_steps',
    'n_categories_in_vocab',
    'n_images_without_annotations',
)

# Long-format counts: same columns for every distribution file (see README for `value` semantics).
_DISTRIBUTION_HEADER: tuple[str, ...] = ('video_id', 'value', 'count')
_DISTRIBUTION_FILENAMES: tuple[str, ...] = (
    'activity_distribution.csv',
    'activity_duration_distribution.csv',
    'process_step_distribution.csv',
    'process_step_duration_distribution.csv',
    'category_distribution.csv',
    'objects_per_frame_distribution.csv',
    'relation_type_distribution.csv',
    'relation_value_distribution.csv',
    'attribute_distribution.csv',
    'attribute_value_distribution.csv',
)
# In activity_duration_distribution, value is segment length in frames (inclusive range).
# In relation_value_distribution, value is f'{relation_type}{_RELATION_VALUE_SEP}{relation_value}'.
# In attribute_value_distribution, value is f'{attribute_type}{_ATTRIBUTE_VALUE_SEP}{attribute_value}'.
_RELATION_VALUE_SEP = '\x1e'
_ATTRIBUTE_VALUE_SEP = '\x1e'


def _write_distribution_rows(writer: csv.writer, video_id: str, value_counts: dict[str, int]) -> None:
    """Append ``(video_id, value, count)`` rows sorted by value."""
    for val, c in sorted(value_counts.items()):
        writer.writerow([video_id, val, c])


def _write_objects_per_frame_rows(writer: csv.writer, video_id: str, buckets: dict[int, int]) -> None:
    for k in sorted(buckets.keys()):
        writer.writerow([video_id, str(k), buckets[k]])


@dataclass
class CsvWriters:
    writer_main: csv.DictWriter
    writer_activity: csv.writer
    writer_activity_duration: csv.writer
    writer_process_step: csv.writer
    writer_process_step_duration: csv.writer
    writer_category: csv.writer
    writer_objects_per_frame: csv.writer
    writer_relation_type: csv.writer
    writer_relation_value: csv.writer
    writer_attribute: csv.writer
    writer_attribute_value: csv.writer


def _write_video_stats_csv(stats: VideoStats, writers: CsvWriters) -> None:
    """Write one video's stats to all open CSV writers."""
    main_row = {
        'video_id': stats.video_id,
        'preparation_type': stats.preparation_type,
        'recording_device': stats.recording_device,
        'width': stats.resolution[0],
        'height': stats.resolution[1],
        'fps': stats.fps if stats.fps > 0 else '',
        'length_in_ms': stats.length_in_ms if stats.length_in_ms > 0 else '',
        'n_frames': stats.n_frames,
        'n_frames_with_objects': stats.n_frames_with_objects,
        'n_object_instances': stats.object_count,
        'n_unique_objects': stats.n_unique_objects,
        'n_relations': stats.relation_count,
        'n_attributes': sum(stats.attribute_value_counts.values()),
        'n_activity_segments': stats.n_activities,
        'n_process_steps': stats.n_process_steps,
        'n_categories_in_vocab': len(stats.categories),
        'n_images_without_annotations': stats.n_images_without_annotations,
    }

    writers.writer_main.writerow(main_row)

    vid = stats.video_id
    _write_distribution_rows(writers.writer_activity, vid, stats.activity_counts)
    _write_objects_per_frame_rows(writers.writer_activity_duration, vid, stats.activity_duration_buckets)
    _write_distribution_rows(writers.writer_process_step, vid, stats.process_step_counts)
    _write_objects_per_frame_rows(writers.writer_process_step_duration, vid, stats.process_step_duration_buckets)
    _write_distribution_rows(writers.writer_category, vid, stats.category_counts)
    _write_objects_per_frame_rows(writers.writer_objects_per_frame, vid, stats.objects_per_frame_buckets)
    _write_distribution_rows(writers.writer_relation_type, vid, _sum_by_first_key(stats.relation_value_counts))
    relation_values_flat = {
        f'{rtype}{_RELATION_VALUE_SEP}{rval}': c for (rtype, rval), c in stats.relation_value_counts.items()
    }
    _write_distribution_rows(writers.writer_relation_value, vid, relation_values_flat)
    _write_distribution_rows(writers.writer_attribute, vid, _sum_by_first_key(stats.attribute_value_counts))
    attribute_values_flat = {
        f'{atype}{_ATTRIBUTE_VALUE_SEP}{aval}': c for (atype, aval), c in stats.attribute_value_counts.items()
    }
    _write_distribution_rows(writers.writer_attribute_value, vid, attribute_values_flat)


@contextmanager
def _csv_writers(output_dir: Path):
    """Open all export CSVs; closes every file on exit (compound with, no ExitStack)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    open_kw = {'newline': '', 'encoding': 'utf-8'}
    dist_paths = [output_dir / name for name in _DISTRIBUTION_FILENAMES]
    with (
        (output_dir / 'videos.csv').open('w', **open_kw) as f_main,
        dist_paths[0].open('w', **open_kw) as f_act,
        dist_paths[1].open('w', **open_kw) as f_adur,
        dist_paths[2].open('w', **open_kw) as f_ps,
        dist_paths[3].open('w', **open_kw) as f_psdur,
        dist_paths[4].open('w', **open_kw) as f_cat,
        dist_paths[5].open('w', **open_kw) as f_opf,
        dist_paths[6].open('w', **open_kw) as f_rt,
        dist_paths[7].open('w', **open_kw) as f_rv,
        dist_paths[8].open('w', **open_kw) as f_attr,
        dist_paths[9].open('w', **open_kw) as f_attr_val,
    ):
        writer_main = csv.DictWriter(f_main, fieldnames=list(_VIDEOS_CSV_FIELDS))
        writer_main.writeheader()

        dist_files = (f_act, f_adur, f_ps, f_psdur, f_cat, f_opf, f_rt, f_rv, f_attr, f_attr_val)
        dist_writers: list[csv.writer] = []
        for f in dist_files:
            w = csv.writer(f)
            w.writerow(list(_DISTRIBUTION_HEADER))
            dist_writers.append(w)

        (
            writer_activity,
            writer_activity_duration,
            writer_process_step,
            writer_process_step_duration,
            writer_category,
            writer_objects_per_frame,
            writer_relation_type,
            writer_relation_value,
            writer_attribute,
            writer_attribute_value,
        ) = dist_writers
        yield CsvWriters(
            writer_main=writer_main,
            writer_activity=writer_activity,
            writer_activity_duration=writer_activity_duration,
            writer_process_step=writer_process_step,
            writer_process_step_duration=writer_process_step_duration,
            writer_category=writer_category,
            writer_objects_per_frame=writer_objects_per_frame,
            writer_relation_type=writer_relation_type,
            writer_relation_value=writer_relation_value,
            writer_attribute=writer_attribute,
            writer_attribute_value=writer_attribute_value,
        )


def _load_and_stats(args: tuple[Path, str]) -> VideoStats:
    video_dir, video_id = args
    video = Video.from_dir(video_dir)
    video.video_id = video_id
    return _get_video_stats(video)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize dataset videos.')
    parser.add_argument('--root', type=Path, required=True, help='Dataset root directory.')
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Also write videos.csv and distribution CSVs under --output-dir.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for CSV export (required with --csv).',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel worker processes.',
    )
    args = parser.parse_args()

    if args.csv and args.output_dir is None:
        parser.error('--csv requires --output-dir')

    video_dir_pairs = discover_video_dirs(args.root)
    if not video_dir_pairs:
        raise SystemExit(f'No videos found under {args.root}')

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        all_stats = list(executor.map(_load_and_stats, video_dir_pairs))

    if args.csv:
        with _csv_writers(args.output_dir) as writers:
            for stats in all_stats:
                print(_format_video_stats_summary(stats))
                _write_video_stats_csv(stats, writers)
    else:
        for stats in all_stats:
            print(_format_video_stats_summary(stats))


if __name__ == '__main__':
    main()
