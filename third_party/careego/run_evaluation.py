"""CaRe-Ego evaluation on the barista-paper hand_object benchmark.

Runs CaRe-Ego inference on the **exact same frames** used by the VLM
hand_object benchmark (read from the built JSONL dataset) and computes
interaction-level metrics: recall/precision, hand/object IoU, hand-type
accuracy, and no-detection rate.

Usage (after running install.sh and activating the careego conda env):

    python third_party/careego/run_evaluation.py \\
        --dataset-root <path/to/dataset> \\
        --benchmark-jsonl runs/vlm_datasets/hand_object_data.jsonl \\
        --checkpoint third_party/careego/weights/best_mIoU_ckpt.pth \\
        --out-dir runs/careego

A timestamped subdir is created under --out-dir (e.g. runs/careego/20250319T120000Z_careego/).

The script writes:
    <out-dir>/predictions.jsonl  — per-frame predictions and GT
    <out-dir>/metrics.json       — interaction recall/precision, hand/object IoU, etc.
    <out-dir>/summary.txt        — human-readable summary
    <out-dir>/debug/             — debug frames with GT (green) and pred (red) bboxes (if --debug-samples N)

Requirements:
    conda activate careego  (see install.sh)
    CaRe-Ego + MMSegmentation installed (via install.sh)
    barista-paper installed in the same env (via install.sh)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# barista-paper dataset loader and bbox utilities (installed via install.sh).
# Import only from hand_object_helpers to avoid pulling in vlm_benchmarks (google-genai).
try:
    from barista.bbox import BboxFormat
    from barista.dataset import Video, discover_video_dirs
    from barista.hand_object_helpers import (
        compute_interaction_metrics_for_frame,
        frame_interactions,
        interaction_to_dict,
        save_debug_frame,
    )
except ImportError as e:
    sys.exit(
        f'barista-paper is not importable: {e}\n'
        'Make sure you ran install.sh and activated the careego conda environment.'
    )

# MMSegmentation / CaRe-Ego — only available after install.sh.
try:
    from mmengine.config import Config  # noqa: F401
    from mmengine.runner import Runner  # noqa: F401
    from mmseg.apis import inference_model, init_model
except ImportError as e:
    sys.exit(f'MMSegmentation is not importable: {e}\nRun install.sh first and activate the careego conda environment.')

log = logging.getLogger('careego_eval')

CAREEGO_CONFIG = Path(__file__).parent / 'mmsegmentation' / 'configs' / 'CaRego.py'


# ---------------------------------------------------------------------------
# Mask → bbox helpers
# ---------------------------------------------------------------------------


def _mask_to_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    """Return ``[x1, y1, x2, y2]`` enclosing all non-zero pixels, or *None*."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    return [int(x_idx[0]), int(y_idx[0]), int(x_idx[-1]), int(y_idx[-1])]


def _scale_bbox(
    bbox: list[int],
    mask_w: int,
    mask_h: int,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Scale *bbox* from mask-pixel coords to image-pixel coords."""
    x1, y1, x2, y2 = bbox
    sx, sy = img_w / mask_w, img_h / mask_h
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# ---------------------------------------------------------------------------
# CaRe-Ego result parsing
# ---------------------------------------------------------------------------


def _extract_masks(result) -> tuple[np.ndarray, ...] | None:
    """Extract CaRe-Ego segmentation masks from an inference result.

    CaregoSegmentor stores predictions in four ``PixelData`` attributes:

        pred_sem_seg_hand      — 0=bg, 1=left_hand, 2=right_hand
        pred_sem_seg_left_obj  — 0=bg, 1=left_object
        pred_sem_seg_right_obj — 0=bg, 1=right_object
        pred_sem_seg_two_obj   — 0=bg, 1=object-touched-by-both-hands

    Returns ``(hand_map, left_obj, right_obj, two_obj)`` as uint8 arrays,
    or *None* when the expected attributes are absent.
    """
    if not hasattr(result, 'pred_sem_seg_hand'):
        return None

    def _squeeze(attr: str) -> np.ndarray:
        return getattr(result, attr).data.squeeze().cpu().numpy().astype(np.uint8)

    return (
        _squeeze('pred_sem_seg_hand'),
        _squeeze('pred_sem_seg_left_obj'),
        _squeeze('pred_sem_seg_right_obj'),
        _squeeze('pred_sem_seg_two_obj'),
    )


def _masks_to_interactions(
    hand_map: np.ndarray,
    left_obj: np.ndarray,
    right_obj: np.ndarray,
    two_obj: np.ndarray,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Convert CaRe-Ego segmentation maps to interaction dicts.

    Output masks are at model crop size (448x448); bboxes are scaled to the
    original image dimensions (*img_w* x *img_h*).
    """
    mask_h, mask_w = hand_map.shape

    left_hand_mask = hand_map == 1
    right_hand_mask = hand_map == 2
    left_obj_mask = left_obj == 1
    right_obj_mask = right_obj == 1
    two_obj_mask = two_obj == 1

    def _s(b: list[int]) -> list[float]:
        return _scale_bbox(b, mask_w, mask_h, img_w, img_h)

    interactions: list[dict] = []

    # Left hand <-> left object
    if left_hand_mask.any() and left_obj_mask.any():
        lh = _mask_to_bbox_xyxy(left_hand_mask)
        lo = _mask_to_bbox_xyxy(left_obj_mask)
        if lh and lo:
            interactions.append(
                {
                    'hand_boxes': [_s(lh)],
                    'object_box': _s(lo),
                    'hand_type': 'left',
                }
            )

    # Right hand <-> right object
    if right_hand_mask.any() and right_obj_mask.any():
        rh = _mask_to_bbox_xyxy(right_hand_mask)
        ro = _mask_to_bbox_xyxy(right_obj_mask)
        if rh and ro:
            interactions.append(
                {
                    'hand_boxes': [_s(rh)],
                    'object_box': _s(ro),
                    'hand_type': 'right',
                }
            )

    # Both hands <-> shared object
    if (left_hand_mask.any() or right_hand_mask.any()) and two_obj_mask.any():
        obj_bbox = _mask_to_bbox_xyxy(two_obj_mask)
        if obj_bbox:
            hand_boxes: list[list[float]] = []
            lh = _mask_to_bbox_xyxy(left_hand_mask)
            rh = _mask_to_bbox_xyxy(right_hand_mask)
            if lh:
                hand_boxes.append(_s(lh))
            if rh:
                hand_boxes.append(_s(rh))
            if hand_boxes:
                interactions.append(
                    {
                        'hand_boxes': hand_boxes,
                        'object_box': _s(obj_bbox),
                        'hand_type': 'both',
                    }
                )

    return interactions


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_benchmark_frames(benchmark_jsonl: Path) -> list[tuple[str, int]]:
    """Read ``(video_id, frame_index)`` pairs from a built VLM benchmark JSONL."""
    pairs: list[tuple[str, int]] = []
    with benchmark_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            vid = row['metadata']['video_id']
            fi = int(row['metadata']['frame_index'])
            pairs.append((vid, fi))
    return pairs


def _decode_video_frames(
    mp4_path: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    """Decode specific frames from *mp4_path* using a single VideoCapture.

    *frame_indices* must be sorted ascending.  Returns a dict mapping each
    successfully decoded frame index to its RGB uint8 numpy array.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    decoded: dict[int, np.ndarray] = {}
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if ok:
            decoded[fi] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            log.warning('Failed to decode frame %d from %s', fi, mp4_path)
    cap.release()
    return decoded


def _load_needed_videos(
    dataset_root: Path,
    needed_ids: set[str],
) -> dict[str, Video]:
    """Load only the videos whose IDs appear in *needed_ids*."""
    videos: dict[str, Video] = {}
    for video_dir, video_id in discover_video_dirs(dataset_root):
        if video_id not in needed_ids:
            continue
        video = Video.from_dir(video_dir)
        video.video_id = video_id
        videos[video_id] = video
    return videos


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_root: Path,
    benchmark_jsonl: Path,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 16,
    debug_samples: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / 'debug' if debug_samples > 0 else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        log.info('Debug frames -> %s', debug_dir)

    # ---- benchmark frame set ------------------------------------------------
    benchmark_frames = _load_benchmark_frames(benchmark_jsonl)
    log.info('Benchmark frames: %d (from %s)', len(benchmark_frames), benchmark_jsonl)

    needed_ids = {vid for vid, _ in benchmark_frames}
    log.info('Loading %d video(s) from %s ...', len(needed_ids), dataset_root)
    video_by_id = _load_needed_videos(dataset_root, needed_ids)

    bbox_fmt = BboxFormat('xyxy')
    bbox_scale = 1000

    # Group by video and sort frames for sequential MP4 access.
    frames_by_video: dict[str, list[int]] = defaultdict(list)
    for video_id, frame_index in benchmark_frames:
        frames_by_video[video_id].append(frame_index)

    frames_to_eval: list[tuple] = []
    for video_id in sorted(frames_by_video):
        video = video_by_id.get(video_id)
        if video is None:
            log.warning('Video %s not in dataset -- skipping', video_id)
            continue
        for fi in sorted(frames_by_video[video_id]):
            frame = video.frame(fi)
            gt = frame_interactions(
                frame=frame,
                interaction_relation_type='human_actions',
                bbox_fmt=bbox_fmt,
                bbox_scale=bbox_scale,
            )
            if gt:
                frames_to_eval.append((video, fi, frame, gt))

    log.info('Frames with GT interactions: %d', len(frames_to_eval))

    # ---- model ---------------------------------------------------------------
    log.info('Loading CaRe-Ego from %s ...', checkpoint)
    model = init_model(str(CAREEGO_CONFIG), str(checkpoint), device=device)
    model.eval()

    # ---- pre-decode frames ---------------------------------------------------
    # Decode all frames using a single cv2.VideoCapture per video (one file
    # open per video instead of one ffmpeg subprocess per frame).
    n_decode = len(frames_to_eval)
    log.info('Decoding %d frames from MP4 ...', n_decode)
    decode_start = time.perf_counter()

    # Collect (video_id -> sorted frame indices) for batch decoding.
    video_frame_indices: dict[str, list[int]] = defaultdict(list)
    for video, fi, _frame, _gt in frames_to_eval:
        video_frame_indices[video.video_id].append(fi)

    # Decode per-video and merge into a flat lookup.
    decoded_frames: dict[tuple[str, int], np.ndarray] = {}
    n_videos = len(video_frame_indices)
    for v_idx, vid_id in enumerate(video_frame_indices):
        video = video_by_id[vid_id]
        indices = sorted(set(video_frame_indices[vid_id]))
        batch = _decode_video_frames(video.mp4_path, indices)
        for fi, arr in batch.items():
            decoded_frames[(vid_id, fi)] = arr
        done = len(decoded_frames)
        elapsed = time.perf_counter() - decode_start
        fps = done / elapsed if elapsed > 0 else 0.0
        eta = (n_decode - done) / fps if fps > 0 else 0.0
        log.info(
            '  video %d/%d  %d frames  total %d/%d  %.1f fps  ETA %.0fs',
            v_idx + 1,
            n_videos,
            len(batch),
            done,
            n_decode,
            fps,
            eta,
        )

    # Build the frame_arrays list in the same order as frames_to_eval.
    frame_arrays: list[np.ndarray] = []
    for video, fi, _frame, _gt in frames_to_eval:
        frame_arrays.append(decoded_frames[(video.video_id, fi)])

    log.info(
        'Decoding done: %d frames in %.1fs',
        n_decode,
        time.perf_counter() - decode_start,
    )

    # ---- inference + metrics -------------------------------------------------
    total_gt = 0
    total_pred = 0
    matched_gt_count = 0
    matched_pred_count = 0
    hand_ious: list[float] = []
    object_ious: list[float] = []
    hand_type_correct = 0
    frames_with_gt = 0
    frames_with_gt_but_zero_pred = 0
    predictions: list[dict] = []

    t_start = time.perf_counter()
    n_total = len(frames_to_eval)
    n_done = 0
    n_warn = 0
    n_debug_saved = 0
    infer_log_every = max(1, n_total // 20)

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_arrays = frame_arrays[batch_start:batch_end]
        batch_meta = frames_to_eval[batch_start:batch_end]

        with torch.no_grad():
            results = inference_model(model, batch_arrays)

        for (video, fi, frame, gt_interactions), arr, result in zip(batch_meta, batch_arrays, results):
            masks = _extract_masks(result)
            if masks is None:
                log.warning('No seg_map for frame %d -- skipping', fi)
                n_warn += 1
                n_done += 1
                continue

            img_h, img_w = frame.height, frame.width
            pred_interactions = _masks_to_interactions(*masks, img_w, img_h)
            gt_dicts = [interaction_to_dict(ia) for ia in gt_interactions]

            fm = compute_interaction_metrics_for_frame(
                gt_dicts,
                pred_interactions,
                img_w,
                img_h,
                bbox_fmt,
                bbox_scale,
                iou_threshold=0.5,
                pred_boxes_in_pixel=True,
            )
            total_gt += fm['total_gt']
            total_pred += fm['total_pred']
            matched_gt_count += fm['matched_gt']
            matched_pred_count += fm['matched_pred']
            hand_ious.extend(fm['hand_ious'])
            object_ious.extend(fm['object_ious'])
            hand_type_correct += fm['hand_type_correct']
            if fm['has_gt']:
                frames_with_gt += 1
            if fm['has_gt_but_zero_pred']:
                frames_with_gt_but_zero_pred += 1

            predictions.append(
                {
                    'video_id': video.video_id,
                    'frame_index': fi,
                    'gt': gt_dicts,
                    'pred': pred_interactions,
                }
            )

            if debug_dir is not None and n_debug_saved < debug_samples:
                save_debug_frame(
                    Image.fromarray(arr),
                    debug_dir / f'{video.video_id}_frame{fi:06d}.jpg',
                    gt_dicts,
                    pred_interactions,
                    img_w,
                    img_h,
                    bbox_fmt,
                    bbox_scale,
                    pred_boxes_in_pixel=True,
                )
                n_debug_saved += 1

            n_done += 1
            if n_done % infer_log_every == 0 or n_done == n_total:
                elapsed = time.perf_counter() - t_start
                fps = n_done / elapsed if elapsed > 0 else 0.0
                eta_m = (n_total - n_done) / fps / 60 if fps > 0 else 0.0
                msg = f'[{n_done}/{n_total}]  {fps:.1f} fps  ETA {eta_m:.1f}min'
                if n_warn:
                    msg += f'  (warn={n_warn})'
                log.info(msg)

    wall_s = time.perf_counter() - t_start

    # ---- aggregate metrics ---------------------------------------------------
    metrics: dict = {
        'wall_clock_s': round(wall_s, 2),
        'frames_evaluated': len(predictions),
        'total_gt_interactions': total_gt,
        'total_pred_interactions': total_pred,
        'matched_gt': matched_gt_count,
        'matched_pred': matched_pred_count,
        'interaction_recall': matched_gt_count / total_gt if total_gt else 0.0,
        'interaction_precision': matched_pred_count / total_pred if total_pred else 0.0,
        'hand_iou': sum(hand_ious) / len(hand_ious) if hand_ious else None,
        'object_iou': sum(object_ious) / len(object_ious) if object_ious else None,
        'hand_type_accuracy': hand_type_correct / matched_gt_count if matched_gt_count else None,
        'no_detection_rate': (frames_with_gt_but_zero_pred / frames_with_gt if frames_with_gt else 0.0),
    }

    # ---- write outputs -------------------------------------------------------
    with (out_dir / 'predictions.jsonl').open('w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    def _f(v: float | None) -> str:
        return f'{v:.4f}' if v is not None else 'N/A'

    summary = (
        '\n'.join(
            [
                f'Frames evaluated       : {metrics["frames_evaluated"]}',
                f'Interaction recall     : {_f(metrics["interaction_recall"])}',
                f'Interaction precision  : {_f(metrics["interaction_precision"])}',
                f'Hand IoU (matched)     : {_f(metrics["hand_iou"])}',
                f'Object IoU (matched)   : {_f(metrics["object_iou"])}',
                f'Hand-type acc.         : {_f(metrics["hand_type_accuracy"])}',
                f'No-detection rate      : {_f(metrics["no_detection_rate"])}',
                f'Wall-clock time        : {wall_s:.1f}s',
            ]
        )
        + '\n'
    )
    (out_dir / 'summary.txt').write_text(summary)

    log.info('\n%s', summary)
    log.info('Results -> %s', out_dir)
    if debug_dir is not None and n_debug_saved:
        log.info('Debug frames -> %s (%d images)', debug_dir, n_debug_saved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run CaRe-Ego on the barista-paper hand_object benchmark.',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        required=True,
        help='Root directory of the barista-paper dataset.',
    )
    parser.add_argument(
        '--benchmark-jsonl',
        type=Path,
        required=True,
        help='Path to the built hand_object_data.jsonl from barista-vlm-build-dataset.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path(__file__).parent / 'weights' / 'best_mIoU_ckpt.pth',
        help='Path to the CaRe-Ego model checkpoint (.pth).',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('runs/careego'),
        help='Output root; a timestamped subdir is created for each run.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar='N',
        help='Inference batch size (default: 16).',
    )
    parser.add_argument(
        '--debug-samples',
        type=int,
        default=0,
        metavar='N',
        help='Save N debug frames with GT (green) and pred (red) bboxes.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device (default: cuda if available).',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    if not args.checkpoint.exists():
        sys.exit(
            f'Checkpoint not found: {args.checkpoint}\n'
            'Download the CaRe-Ego weights -- see third_party/careego/README.md'
        )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = args.out_dir / f'{timestamp}_careego'
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('Run output -> %s', out_dir)

    run_evaluation(
        dataset_root=args.dataset_root,
        benchmark_jsonl=args.benchmark_jsonl,
        checkpoint=args.checkpoint,
        out_dir=out_dir,
        device=args.device,
        batch_size=args.batch_size,
        debug_samples=args.debug_samples,
    )


if __name__ == '__main__':
    main()
