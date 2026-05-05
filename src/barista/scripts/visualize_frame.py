"""Render a single annotated frame as a PNG image.

Draws bounding boxes and segmentation masks for object annotations on the
selected frame.  Output defaults to ``<video_id>_<frame_index>.png`` in the
current directory.

Example::

    barista-visualize \
        --root /data/barista \
        --video-id 057feaac-0c7f-48fd-bcea-4ecbf1685ebc \
        --frame-index 42 \
        --out /tmp/frame_42.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from barista.dataset import Video
from barista.visualize import RenderOptions, visualize_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Dataset root directory.')
    parser.add_argument('--video-id', required=True, help='Video directory name relative to root.')
    parser.add_argument('--frame-index', type=int, required=True, help='Frame index.')
    parser.add_argument('--out', type=Path, help='Output PNG path (default: ./<video_id>_<frame_index>.png).')
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
    out_path = visualize_frame(video, frame_index=args.frame_index, options=options, out_path=args.out)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
