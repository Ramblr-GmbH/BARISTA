# SAM 3 baseline evaluation

Runs [SAM 3](https://github.com/facebookresearch/sam3) on the barista-paper
grounding benchmark.

SAM 3 is a **separate conda environment** — it has different dependencies from
the main barista-paper virtualenv.

---

## Setup

### 1. Install

```bash
bash third_party/sam3/install.sh
```

This will:
- Create a conda env named `sam3` with PyTorch + CUDA
- Clone SAM 3 into `third_party/sam3/sam3_src` and install it from there
- Install barista-paper for dataset loading and evaluation

### 2. Download model weights

Download a SAM 3 checkpoint from the repository:

- See the [SAM 3 README](https://github.com/facebookresearch/sam3#model-checkpoints)

Save it to:

```
third_party/sam3/weights/
```

---

## Run evaluation

The script evaluates SAM 3 on the **exact same examples** used by the VLM
grounding benchmark (one phrase per example, read from the built JSONL dataset).

```bash
# Step 1: build the grounding benchmark (run in main barista-paper env, not sam3)
barista-vlm-build-dataset --config configs/vlm_benchmarks/grounding/build.json

# Step 2: evaluate with SAM 3
conda activate sam3

python third_party/sam3/run_evaluation.py \
    --dataset-root <path/to/dataset> \
    --benchmark-jsonl runs/vlm_datasets/grounding_data.jsonl \
    --checkpoint third_party/sam3/weights/<checkpoint>.pt \
    --out-dir runs/sam3
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | required | barista-paper dataset directory |
| `--benchmark-jsonl` | required | Frozen grounding JSONL from `barista-vlm-build-dataset` |
| `--checkpoint` | required | Path to SAM 3 checkpoint |
| `--out-dir` | `runs/sam3` | Output root; a timestamped subdir is created per run |
| `--device` | `cuda` or `cpu` | PyTorch device |
| `--confidence-threshold` | `0.5` | SAM 3 confidence threshold |
| `--nms-iou-threshold` | `0.3` | NMS IoU threshold for merging detections |
| `--debug` | off | Save side-by-side GT/prediction images to `debug/` subdir |
| `--verbose` | off | Log details for every example |
