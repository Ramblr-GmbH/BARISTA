# BARISTA: A Multi-Task Egocentric Benchmark for Compositional Visual Understanding

**BARISTA** is a densely annotated egocentric dataset and unified benchmark of 185 real-world coffee-preparation videos.
Each annotated frame includes instance masks, bounding boxes, per-instance attributes, and directed typed relations anchored to persistent object identities, together with activity and process-step labels.

The benchmark evaluates six complementary tasks under a shared zero-shot protocol: phrase grounding, hand-object interaction recognition, referring expression generation, relation extraction, activity recognition, and temporal visual question answering.

## Quickstart

```bash
uv venv --python 3.12
uv sync --extra vlm --extra dev
source .venv/bin/activate
```

## Development

```bash
uv run ruff format
uv run ruff check
```

## Dataset layout

```
<root>/
  <video_id>/
    coco_annotation.json
    video.mp4
```

## CLI

### Dataset tools

```bash
barista-summarize --root test_dataset                       # per-video summary (--csv, --output-dir)
barista-visualize --root test_dataset --video-id VID --frame-index 0 --out /tmp/frame.png
barista-visualize-segment --root test_dataset --video-id VID --start 0 --end 30 --fps 6 --out /tmp/seg.mp4
```

### VLM benchmarking

| Command | Purpose |
|---------|---------|
| `barista-vlm-build-dataset` | Freeze a benchmark dataset to JSONL |
| `barista-vlm-prepare` | Prepare task-specific assets (visual_qa) |
| `barista-vlm-run` | Run inference + evaluation |
| `barista-vlm-judge` | Post-hoc G-Eval judge on a completed run |

**Available tasks:** `activity_mcq`, `grounding`, `hand_object`, `visual_qa`, `referring`, `relation_extraction`.

**Typical workflow:**

```bash
# 1. (asset-backed tasks only) prepare assets
barista-vlm-prepare --config configs/vlm_benchmarks/visual_qa/prepare.json

# 2. build frozen dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/activity_mcq/build.json --out examples.jsonl

# 3. run evaluation
barista-vlm-run --config configs/vlm_benchmarks/activity_mcq/gemini.json --examples-path examples.jsonl
```

Output goes to `output_dir/<run_id>/`: `predictions.jsonl`, `errors.jsonl`, `metrics.json`, `run_config.json`, `summary.txt`.

**Environment variables:** `GEMINI_API_KEY`, `OPENAI_API_KEY`, or GCP Application Default Credentials (for Vertex AI).

For config reference, per-task parameters, and advanced workflows, see [`configs/vlm_benchmarks/README.md`](configs/vlm_benchmarks/README.md).

## Dataset format notes

- Organised as `<root>/<video_id>/`.
- Segmentations are COCO RLE (`counts` + `size`).
- Attributes and relations are stored per-object with `image_ranges`, expanded to per-frame data at load time.
- Activities are per-segment with `image_range` and a `display_name` label.

The loader exposes: `Video` → `frames` → `objects` (`ObjectAnnotation` with `category`, `bbox`, `mask`, `attributes`), plus `Relation` (per-frame) and `Activity` (frame interval).
