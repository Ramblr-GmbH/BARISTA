"""SAM 3 grounding evaluation on the barista-paper dataset.

Runs SAM 3 text-prompted detection on frozen benchmark examples (one phrase
per example, produced by ``barista-vlm-build-dataset``).  Computes COCO-style
box mAP and segmentation mask mAP via torchmetrics ``MeanAveragePrecision``.

Usage (after running install.sh and activating the sam3 conda env):

    python third_party/sam3/run_evaluation.py \
        --dataset-root <path/to/dataset> \
        --benchmark-jsonl runs/vlm_datasets/grounding_data.jsonl \
        --checkpoint <path/to/sam3.pt> \
        --out-dir runs/sam3

A timestamped subdir is created under --out-dir.

The script writes:
    <out-dir>/predictions.jsonl  — per-example predictions and GT
    <out-dir>/metrics.json       — box mAP, mask mAP, latency, etc.
    <out-dir>/summary.txt        — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou, nms

try:
    from barista.bbox import BboxFormat, normalized_to_pixel
    from barista.dataset import Video, discover_video_dirs
except ImportError as e:
    sys.exit(
        f'barista-paper is not importable: {e}\nMake sure you ran install.sh and activated the sam3 conda environment.'
    )

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    sys.exit(f'SAM 3 is not importable: {e}\nRun install.sh first and activate the sam3 conda environment.')

# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------


def _draw_debug_frame(
    image: Image.Image,
    gt_boxes: list[list[float]],
    phrase: str,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    out_path: Path,
) -> None:
    """Save a side-by-side image: GT (green) on left, predictions (red) on right."""
    from PIL import ImageDraw

    img = image.convert('RGB')

    gt_img = img.copy()
    draw_gt = ImageDraw.Draw(gt_img)
    for box in gt_boxes:
        x0, y0, x1, y1 = box
        draw_gt.rectangle([x0, y0, x1, y1], outline='green', width=6)
        draw_gt.text((x0, max(y0 - 12, 0)), f'GT: {phrase}', fill='green')

    pred_img = img.copy()
    draw_pred = ImageDraw.Draw(pred_img)
    for i in range(pred_boxes.shape[0]):
        x0, y0, x1, y1 = pred_boxes[i].tolist()
        score = pred_scores[i].item()
        draw_pred.rectangle([x0, y0, x1, y1], outline='red', width=6)
        draw_pred.text((x0, max(y0 - 12, 0)), f'{phrase} {score:.2f}', fill='red')

    combined = Image.new('RGB', (gt_img.width + pred_img.width, gt_img.height))
    combined.paste(gt_img, (0, 0))
    combined.paste(pred_img, (gt_img.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)


# ---------------------------------------------------------------------------
# Data loading from frozen JSONL
# ---------------------------------------------------------------------------


def _load_benchmark_examples(benchmark_jsonl: Path) -> list[dict[str, Any]]:
    """Read benchmark examples from the frozen JSONL."""
    examples: list[dict[str, Any]] = []
    with benchmark_jsonl.open() as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


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
# Frame pre-loading
# ---------------------------------------------------------------------------


def _decode_video_frames(
    mp4_path: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    """Decode specific frames from *mp4_path* with a single VideoCapture.

    *frame_indices* should be sorted ascending for minimal seeking.
    Returns a dict mapping frame index → RGB uint8 numpy array.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    decoded: dict[int, np.ndarray] = {}
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if ok:
            decoded[fi] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            print(f'  Warning: failed to decode frame {fi} from {mp4_path}')
    cap.release()
    return decoded


def _preload_frames(
    frame_groups_sorted: list[tuple[tuple[str, int], list[tuple[int, dict]]]],
    video_by_id: dict[str, Any],
) -> dict[tuple[str, int], Image.Image]:
    """Batch-decode all required frames and return a ``{(vid, fi): PIL.Image}`` dict.

    Opens each video's MP4 once with a single cv2.VideoCapture, seeking to
    each required frame in sorted order.  Much faster than spawning an ffmpeg
    subprocess per frame (which is what imageio does).
    """
    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for (vid, fi), _ in frame_groups_sorted:
        if vid in video_by_id:
            indices_by_video[vid].append(fi)

    loaded: dict[tuple[str, int], Image.Image] = {}
    for vid, indices in indices_by_video.items():
        video = video_by_id[vid]
        batch = _decode_video_frames(video.mp4_path, sorted(set(indices)))
        for fi, arr in batch.items():
            loaded[(vid, fi)] = Image.fromarray(arr)
    return loaded


# ---------------------------------------------------------------------------
# SAM 3 inference for one phrase
# ---------------------------------------------------------------------------


def _predict_phrase(
    processor: Sam3Processor,
    state: dict,
    phrase: str,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    """Run SAM 3 on a single phrase using a pre-set image state.

    Returns (boxes, scores, masks) where boxes are xyxy pixel coords.
    """
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=phrase, state=state)

    masks = state.get('masks')
    boxes = state.get('boxes')
    scores = state.get('scores')

    if masks is None or boxes is None or scores is None or boxes.numel() == 0:
        return torch.zeros(0, 4), torch.zeros(0), []

    pred_scores = scores.cpu().float()
    masks_np = masks[:, 0].cpu().numpy()

    # Derive boxes from masks for coordinate consistency.
    derived = []
    mask_list = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        mask_list.append(m)
        ys, xs = np.where(m)
        if len(ys) == 0:
            derived.append([0.0, 0.0, 0.0, 0.0])
        else:
            derived.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    pred_boxes = torch.tensor(derived, dtype=torch.float32) if derived else torch.zeros(0, 4)

    return pred_boxes, pred_scores, mask_list


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_root: Path,
    benchmark_jsonl: Path,
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load benchmark examples ----
    print(f'Loading benchmark examples from {benchmark_jsonl} ...')
    examples = _load_benchmark_examples(benchmark_jsonl)
    print(f'  {len(examples)} examples')

    # ---- Load needed videos ----
    needed_ids = {ex['metadata']['video_id'] for ex in examples}
    print(f'Loading {len(needed_ids)} video(s) from {dataset_root} ...')
    video_by_id = _load_needed_videos(dataset_root, needed_ids)

    # Build category -> id mapping from all examples.
    category_names: set[str] = set()
    for ex in examples:
        category_names.add(ex['task_data']['category'])
    sorted_categories = sorted(category_names)
    category_to_id: dict[str, int] = {name: idx for idx, name in enumerate(sorted_categories)}
    id_to_category: dict[int, str] = {v: k for k, v in category_to_id.items()}
    print(f'  {len(sorted_categories)} categories')

    # ---- Load model ----
    print(f'Loading SAM 3 from {checkpoint} ...')
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        device=device,
        load_from_HF=False,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    print('Model loaded.')

    # ---- Metrics accumulators ----
    box_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    seg_metric = MeanAveragePrecision(box_format='xyxy', iou_type='segm', class_metrics=True)

    predictions_log: list[dict] = []
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Group examples by (video_id, frame_index) so we only load each frame once.
    frame_groups: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        vid = ex['metadata']['video_id']
        fi = ex['metadata']['frame_index']
        frame_groups[(vid, fi)].append((idx, ex))

    n_total = len(examples)
    n_done = 0
    t_start = time.perf_counter()
    _LOG_EVERY = max(1, n_total // 20)

    frame_groups_sorted = sorted(frame_groups.items())

    # ---- Pre-load all frames ----
    print(f'Pre-loading {len(frame_groups_sorted)} unique frame(s) into memory ...')
    t_load = time.perf_counter()
    preloaded = _preload_frames(frame_groups_sorted, video_by_id)
    print(f'  Done in {time.perf_counter() - t_load:.1f}s')

    n_total = len(examples)
    n_done = 0
    t_start = time.perf_counter()
    _LOG_EVERY = max(1, n_total // 20)

    for (vid, fi), group in frame_groups_sorted:
        frame_image = preloaded.get((vid, fi))
        if frame_image is None:
            print(f'  Warning: video {vid} not found, skipping {len(group)} examples')
            n_done += len(group)
            continue

        img_w, img_h = frame_image.size

        # Set image once for all phrases in this frame.
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            state = processor.set_image(frame_image)

        for _ex_idx, ex in group:
            phrase = ex['task_data']['phrase']
            category = ex['task_data']['category']
            gt_boxes_norm = ex['task_data']['ground_truth_boxes']
            bbox_format = BboxFormat(ex['task_data'].get('bbox_format', 'xyxy'))
            bbox_scale = int(ex['task_data'].get('bbox_scale', 1000))

            # Convert GT from normalized to pixel coords.
            gt_boxes_pixel = [
                normalized_to_pixel(box, img_w, img_h, fmt=bbox_format, scale=bbox_scale) for box in gt_boxes_norm
            ]

            cat_id = category_to_id[category]

            # ---- Run SAM 3 for this phrase ----
            t_phrase = time.perf_counter()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                pred_boxes, pred_scores, pred_masks = _predict_phrase(processor, state, phrase)
            latency_ms = (time.perf_counter() - t_phrase) * 1000.0

            # Apply NMS within this phrase's detections.
            if pred_boxes.numel() > 0:
                keep = nms(pred_boxes, pred_scores, nms_iou_threshold)
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_masks = [pred_masks[i] for i in keep.tolist()]

            n_pred = pred_boxes.shape[0]
            n_gt = len(gt_boxes_pixel)

            # ---- Build GT tensors ----
            gt_boxes_t = torch.tensor(gt_boxes_pixel, dtype=torch.float32) if gt_boxes_pixel else torch.zeros(0, 4)
            gt_labels_t = torch.full((n_gt,), cat_id, dtype=torch.int64)

            # ---- Debug ----
            if debug:
                safe_phrase = phrase.replace(' ', '_').replace('/', '_')[:40]
                debug_path = out_dir / 'debug' / f'{vid.replace("/", "_")}_{fi:06d}_{safe_phrase}.png'
                _draw_debug_frame(frame_image, gt_boxes_pixel, phrase, pred_boxes, pred_scores, debug_path)

            # ---- Update box mAP ----
            pred_labels_t = torch.full((n_pred,), cat_id, dtype=torch.int64)
            box_metric.update(
                preds=[{'boxes': pred_boxes.float(), 'scores': pred_scores.float(), 'labels': pred_labels_t}],
                target=[{'boxes': gt_boxes_t, 'labels': gt_labels_t}],
            )

            # ---- Update segmentation mAP (pred masks available from SAM 3) ----
            if pred_masks:
                pred_masks_t = torch.stack([torch.from_numpy(m.astype(bool)) for m in pred_masks])
                # Create GT masks from GT boxes (filled rectangles) for segm mAP.
                # Build all masks at once with a vectorised numpy op.
                if gt_boxes_pixel:
                    boxes_arr = np.array(gt_boxes_pixel, dtype=np.int32).clip(
                        [0, 0, 0, 0], [img_w, img_h, img_w, img_h]
                    )
                    gt_masks_arr = np.zeros((len(gt_boxes_pixel), img_h, img_w), dtype=np.bool_)
                    for i, (x0, y0, x1, y1) in enumerate(boxes_arr):
                        gt_masks_arr[i, y0:y1, x0:x1] = True
                    gt_masks_t = torch.from_numpy(gt_masks_arr)
                    seg_metric.update(
                        preds=[
                            {
                                'boxes': pred_boxes.float(),
                                'scores': pred_scores.float(),
                                'labels': pred_labels_t,
                                'masks': pred_masks_t,
                            }
                        ],
                        target=[{'boxes': gt_boxes_t, 'labels': gt_labels_t, 'masks': gt_masks_t}],
                    )

            # ---- Log prediction ----
            pred_log: list[dict] = []
            if pred_boxes.numel() > 0 and gt_boxes_t.numel() > 0:
                iou_matrix = box_iou(pred_boxes.float(), gt_boxes_t)
            else:
                iou_matrix = torch.zeros(n_pred, n_gt)

            for i in range(n_pred):
                best_iou = iou_matrix[i].max().item() if n_gt > 0 else 0.0
                pred_log.append(
                    {
                        'box_xyxy': pred_boxes[i].tolist(),
                        'score': round(pred_scores[i].item(), 4),
                        'best_iou': round(best_iou, 4),
                    }
                )

            predictions_log.append(
                {
                    'video_id': vid,
                    'frame_index': fi,
                    'phrase': phrase,
                    'category': category,
                    'latency_ms': round(latency_ms, 1),
                    'gt_count': n_gt,
                    'pred_count': n_pred,
                    'gt_boxes': gt_boxes_pixel,
                    'pred': pred_log,
                }
            )

            n_done += 1
            if verbose:
                print(
                    f'  [{n_done}/{n_total}]  video={vid}  frame={fi}'
                    f'  phrase="{phrase}"  gt={n_gt}  pred={n_pred}  {latency_ms:.0f}ms',
                    flush=True,
                )
            elif n_done % _LOG_EVERY == 0 or n_done == n_total:
                elapsed = time.perf_counter() - t_start
                fps = n_done / elapsed if elapsed > 0 else 0.0
                eta_s = (n_total - n_done) / fps if fps > 0 else 0.0
                print(f'  [{n_done}/{n_total}]  {fps:.2f} ex/s  ETA {eta_s / 60:.1f} min', flush=True)

    wall_clock_ms = (time.perf_counter() - t_start) * 1000.0

    # ---- Compute final metrics ----
    def _extract_map_results(metric: MeanAveragePrecision) -> dict[str, Any]:
        result = metric.compute()
        out: dict[str, Any] = {}
        for key in (
            'map',
            'map_50',
            'map_75',
            'map_small',
            'map_medium',
            'map_large',
            'mar_1',
            'mar_10',
            'mar_100',
            'mar_small',
            'mar_medium',
            'mar_large',
        ):
            val = result.get(key)
            out[key] = val.item() if val is not None else None

        per_category: dict[str, dict[str, float | None]] = {}
        map_per_class = result.get('map_per_class')
        mar_100_per_class = result.get('mar_100_per_class')
        classes = result.get('classes')
        if map_per_class is not None and map_per_class.numel() > 0:
            for i in range(map_per_class.shape[0]):
                cat_idx = classes[i].item() if classes is not None else i
                cat_name = id_to_category.get(cat_idx, f'class_{cat_idx}')
                cat_map = map_per_class[i].item()
                cat_mar = mar_100_per_class[i].item() if mar_100_per_class is not None else None
                per_category[cat_name] = {
                    'map': cat_map if cat_map >= 0 else None,
                    'mar_100': cat_mar if cat_mar is not None and cat_mar >= 0 else None,
                }
        out['per_category'] = per_category
        return out

    box_results = _extract_map_results(box_metric)
    seg_results = _extract_map_results(seg_metric)

    metrics: dict[str, Any] = {
        'wall_clock_ms': round(wall_clock_ms, 1),
        'examples_evaluated': n_done,
        'confidence_threshold': confidence_threshold,
        'nms_iou_threshold': nms_iou_threshold,
        'box': box_results,
        'segmentation': seg_results,
    }

    # ---- Write outputs ----
    preds_path = out_dir / 'predictions.jsonl'
    with preds_path.open('w') as f:
        for p in predictions_log:
            f.write(json.dumps(p) + '\n')

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')

    def _fmt(v: float | None) -> str:
        return f'{v:.4f}' if v is not None else 'N/A'

    summary_lines = [
        f'Examples evaluated     : {n_done}',
        f'Confidence threshold   : {confidence_threshold}',
        f'NMS IoU threshold      : {nms_iou_threshold}',
        '',
        '--- Box Detection ---',
        f'mAP @[.5:.95]          : {_fmt(box_results.get("map"))}',
        f'mAP @0.50              : {_fmt(box_results.get("map_50"))}',
        f'mAP @0.75              : {_fmt(box_results.get("map_75"))}',
        f'mAP @small             : {_fmt(box_results.get("map_small"))}',
        f'mAP @medium            : {_fmt(box_results.get("map_medium"))}',
        f'mAP @large             : {_fmt(box_results.get("map_large"))}',
        f'mAR @1                 : {_fmt(box_results.get("mar_1"))}',
        f'mAR @10                : {_fmt(box_results.get("mar_10"))}',
        f'mAR @100               : {_fmt(box_results.get("mar_100"))}',
        f'mAR @small             : {_fmt(box_results.get("mar_small"))}',
        f'mAR @medium            : {_fmt(box_results.get("mar_medium"))}',
        f'mAR @large             : {_fmt(box_results.get("mar_large"))}',
        '',
        '--- Segmentation ---',
        f'mAP @[.5:.95]          : {_fmt(seg_results.get("map"))}',
        f'mAP @0.50              : {_fmt(seg_results.get("map_50"))}',
        f'mAP @0.75              : {_fmt(seg_results.get("map_75"))}',
        f'mAP @small             : {_fmt(seg_results.get("map_small"))}',
        f'mAP @medium            : {_fmt(seg_results.get("map_medium"))}',
        f'mAP @large             : {_fmt(seg_results.get("map_large"))}',
        f'mAR @1                 : {_fmt(seg_results.get("mar_1"))}',
        f'mAR @10                : {_fmt(seg_results.get("mar_10"))}',
        f'mAR @100               : {_fmt(seg_results.get("mar_100"))}',
        f'mAR @small             : {_fmt(seg_results.get("mar_small"))}',
        f'mAR @medium            : {_fmt(seg_results.get("mar_medium"))}',
        f'mAR @large             : {_fmt(seg_results.get("mar_large"))}',
        '',
        f'Wall-clock time        : {wall_clock_ms / 1000:.1f}s',
    ]

    box_per_cat = box_results.get('per_category', {})
    if box_per_cat:
        summary_lines.append('')
        summary_lines.append('Per-category box mAP:')
        for cat_name in sorted(box_per_cat):
            stats = box_per_cat[cat_name]
            summary_lines.append(f'  {cat_name}: mAP={_fmt(stats.get("map"))}  mAR@100={_fmt(stats.get("mar_100"))}')

    seg_per_cat = seg_results.get('per_category', {})
    if seg_per_cat:
        summary_lines.append('')
        summary_lines.append('Per-category segmentation mAP:')
        for cat_name in sorted(seg_per_cat):
            stats = seg_per_cat[cat_name]
            summary_lines.append(f'  {cat_name}: mAP={_fmt(stats.get("map"))}  mAR@100={_fmt(stats.get("mar_100"))}')

    summary = '\n'.join(summary_lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary)

    print('\n' + summary)
    print(f'Results written to {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SAM 3 grounding evaluation on frozen benchmark examples.')
    parser.add_argument('--dataset-root', type=Path, required=True, help='Root directory of the barista-paper dataset.')
    parser.add_argument(
        '--benchmark-jsonl',
        type=Path,
        required=True,
        help='Frozen grounding benchmark JSONL (from barista-vlm-build-dataset).',
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to the SAM 3 checkpoint (.pt).')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('runs/sam3'),
        help='Output root; a timestamped subdir is created for each run.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device (default: cuda if available).',
    )
    parser.add_argument('--confidence-threshold', type=float, default=0.2, help='SAM 3 confidence threshold.')
    parser.add_argument(
        '--nms-iou-threshold', type=float, default=0.5, help='NMS IoU threshold for merging detections.'
    )
    parser.add_argument('--verbose', action='store_true', help='Log details for every example.')
    parser.add_argument('--debug', action='store_true', help='Save side-by-side GT/prediction images to debug/ subdir.')
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(
            f'Checkpoint not found: {args.checkpoint}\nDownload a SAM 3 checkpoint — see third_party/sam3/README.md'
        )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = args.out_dir / f'{timestamp}_sam3'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Run output: {out_dir}')

    run_evaluation(
        dataset_root=args.dataset_root,
        benchmark_jsonl=args.benchmark_jsonl,
        checkpoint=args.checkpoint,
        out_dir=out_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        verbose=args.verbose,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
