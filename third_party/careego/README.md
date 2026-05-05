# CaRe-Ego baseline evaluation

Runs [CaRe-Ego](https://github.com/yuggiehk/CaRe-Ego) on the barista-paper
`hand_object` benchmark and produces metrics comparable to the VLM results:
Interaction-level metrics: recall, precision, hand/object IoU, hand-type accuracy, no-detection rate (same as VLM hand_object for fair comparison).

CaRe-Ego is a **separate conda environment** — it uses MMSegmentation and has
different dependencies from the main barista-paper virtualenv.

---

## Setup

### 1. Install

```bash
bash third_party/careego/install.sh
```

This follows the [official CaRe-Ego setup](https://github.com/yuggiehk/CaRe-Ego):
- Clones CaRe-Ego and creates a conda env from their `mmseg.yml`
- Installs MMSegmentation (`pip install -e .`)
- Copies CaRe-Ego files into MMSegmentation and patches `__init__.py`
- Runs `python setup.py install`
- Installs barista-paper for dataset loading and evaluation

The env is named `careego` (CaRe-Ego's yml uses `mmseg`; we override for clarity).

### 2. Download model weights

Download the best mIoU checkpoint from the CaRe-Ego repository:

- **Google Drive / Hugging Face**: see the
  [CaRe-Ego README](https://github.com/yuggiehk/CaRe-Ego#inference)

Save it to:

```
third_party/careego/weights/best_mIoU_ckpt.pth
```

---

## Run evaluation

Activate the conda environment and run the evaluation script:

```bash
conda activate careego

python third_party/careego/run_evaluation.py \
    --dataset-root <path/to/dataset> \
    --benchmark-jsonl runs/vlm_datasets/hand_object_data.jsonl \
    --checkpoint third_party/careego/weights/best_mIoU_ckpt.pth \
    --out-dir runs/careego
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | required | barista-paper dataset directory |
| `--benchmark-jsonl` | required | Path to the built `hand_object_data.jsonl` from the VLM benchmark build step |
| `--checkpoint` | `weights/best_mIoU_ckpt.pth` | Path to CaRe-Ego .pth checkpoint |
| `--out-dir` | `runs/careego` | Output root; a timestamped subdir is created per run |
| `--device` | `cuda` or `cpu` | PyTorch device |
| `--debug-samples` | `0` | Save N frames with GT (green) and pred (red) bboxes to `<out-dir>/debug/` |

Test without a GPU (slow):

```bash
python third_party/careego/run_evaluation.py \
    --dataset-root test_dataset \
    --benchmark-jsonl runs/vlm_datasets/hand_object_data.jsonl \
    --checkpoint third_party/careego/weights/best_mIoU_ckpt.pth \
    --out-dir runs/careego_test \
    --device cpu
```

---

## Output

Results are written to `--out-dir`:

| File | Contents |
|------|----------|
| `predictions.jsonl` | Per-frame predictions and GT interactions |
| `metrics.json` | Interaction recall/precision, hand/object IoU, hand-type accuracy, no-detection rate (same keys as VLM runs) |
| `summary.txt` | Human-readable summary |

`metrics.json` keys match the VLM pipeline:
- `interaction_recall`, `interaction_precision` — interaction detection
- `hand_iou`, `object_iou` — per-component quality on matched pairs
- `hand_type_accuracy` — fraction of matched interactions with correct hand type
- `no_detection_rate` — fraction of frames with GT but zero predictions
- `wall_clock_ms` — total evaluation time

---

## How it works

1. **Dataset** — loads barista-paper videos via `barista.dataset.load_videos`.
   Frames with at least one `human_actions` relation are evaluated.

2. **Inference** — CaRe-Ego outputs a per-pixel segmentation map with classes:
   - `0` background
   - `1` left hand
   - `2` right hand
   - `3` object (left hand only)
   - `4` object (right hand only)
   - `5` object (both hands)

3. **Mask → bbox** — bounding boxes are computed as the tight rectangle around
   each non-zero mask region.

4. **Metrics** — interaction-level metrics (shared with VLM hand_object task).
   Greedy matching by object-box IoU (threshold 0 by default).

---

## Notes

- The class index constants in `run_evaluation.py` (`CLASS_LEFT_HAND`, etc.)
  are based on the EgoHOS label convention used in CaRe-Ego.  If your model
  was fine-tuned with different indices, update them.
- CaRe-Ego predicts segmentation masks, not bboxes directly.  The bbox is the
  tightest rectangle enclosing the mask, which is a minor difference from the
  VLM task (where the model predicts bboxes directly).
