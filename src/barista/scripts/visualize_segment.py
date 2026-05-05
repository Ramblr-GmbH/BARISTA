"""Render an annotated video segment as MP4 or GIF.

Draws bounding boxes and segmentation masks on each frame in the range
[start, end] (inclusive) with an optional step.  Output defaults to
``<video_id>_<start>_<end>.mp4`` in the current directory.

Example::

    barista-visualize-segment \
        --root /data/barista \
        --video-id 057feaac-0c7f-48fd-bcea-4ecbf1685ebc \
        --start 120 --end 150 \
        --fps 30 \
        --out /tmp/segment.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from barista.dataset import Video
from barista.visualize import RenderOptions, visualize_segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Dataset root directory.')
    parser.add_argument('--video-id', required=True, help='Video directory name relative to root.')
    parser.add_argument('--start', type=int, required=True, help='Start frame index (inclusive).')
    parser.add_argument('--end', type=int, required=True, help='End frame index (inclusive).')
    parser.add_argument('--step', type=int, default=1, help='Frame step size (default: 1).')
    parser.add_argument('--fps', type=int, default=6, help='Output frames per second (default: 6).')
    parser.add_argument('--out', type=Path, help='Output path (.mp4 or .gif).')
    parser.add_argument('--max-objects', type=int, help='Max number of objects to render.')
    parser.add_argument('--no-bbox', action='store_true', help='Disable bounding boxes.')
    parser.add_argument('--no-mask', action='store_true', help='Disable segmentation masks.')
    args = parser.parse_args()

    video = Video.from_dir(args.root / args.video_id)
    options = RenderOptions(
        show_bboxes=not args.no_bbox,
        show_masks=not args.no_mask,
        max_objects=args.max_objects,
    )
    out_path = visualize_segment(
        video,
        start_frame=args.start,
        end_frame=args.end,
        step=args.step,
        options=options,
        out_path=args.out,
        fps=args.fps,
    )
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
