# VLM Benchmark Configs

Starter JSON configs for asset preparation, frozen-dataset building, and inference/evaluation.

Configs are grouped by task under `configs/vlm_benchmarks/<task_name>/`.
Each task implements the `BenchmarkTask` protocol; see `src/barista/vlm_benchmarks/tasks/`.

## Config roles

| File | CLI command | Purpose |
|------|------------|---------|
| `prepare.json` | `barista-vlm-prepare` | Generate model-backed reference assets (visual_qa only) |
| `build.json` | `barista-vlm-build-dataset` | Freeze benchmark examples to JSONL (provider-free) |
| `gemini.json` / `azure_openai.json` / `*_openai_compat.json` | `barista-vlm-run` | Inference + evaluation against a provider |

To skip inline judging (visual_qa, referring), set `"geval_criteria": []` in `task_params` and run `barista-vlm-judge` post-hoc.

**Annotation-only tasks** (`activity_mcq`, `grounding`, `hand_object`, `relation_extraction`, `referring`) build directly from `build.json`.

**Asset-backed tasks** (`visual_qa`) require a `barista-vlm-prepare` step before building.

## Fields to edit

- `dataset_root` — path to the dataset (e.g. `/data/barista`)
- `task_params.*` — task-specific parameters (frames per example etc.)
- `model.model` — model name in provider run configs
- `model.base_url` — endpoint URL for `openai_compat` providers
- `concurrency` — parallel API requests (default 1)

## CLI reference

### `barista-vlm-build-dataset`

Freeze a benchmark dataset to a JSONL file.

```bash
barista-vlm-build-dataset \
  --config configs/vlm_benchmarks/activity_mcq/build.json \
  --out runs/vlm_datasets/activity_mcq.jsonl
```

| Flag | Description |
|------|-------------|
| `--config` | Build config JSON (required) |
| `--out` | Output JSONL path (required) |
| `--limit N` | Cap total examples |
| `--video-id ID` | Restrict to one video |
| `--task NAME` | Override task name |
| `--sample-fraction F` | Fraction of frames per video (0–1] |
| `--no-shuffle` | Disable example shuffling |
| `--activity-only` | Only frames inside activity segments |

### `barista-vlm-prepare`

Prepare task-specific assets.

```bash
barista-vlm-prepare --config configs/vlm_benchmarks/visual_qa/prepare.json
```

| Flag | Description |
|------|-------------|
| `--config` | Prepare config JSON (required) |
| `--limit N` | Cap assets to generate |
| `--video-id ID` | Restrict to one video |
| `--model NAME` | Override model |
| `--provider NAME` | Override provider |
| `--base-url URL` | Override endpoint |
| `--out-dir PATH` | Override output directory |
| `--concurrency N` | Parallel generation workers |

### `barista-vlm-run`

Run inference and evaluation.

```bash
barista-vlm-run \
  --config configs/vlm_benchmarks/activity_mcq/gemini.json \
  --examples-path runs/vlm_datasets/activity_mcq.jsonl
```

| Flag | Description |
|------|-------------|
| `--config` | Run config JSON (required) |
| `--examples-path` | Pre-built benchmark JSONL (required) |
| `--model NAME` | Override model |
| `--provider NAME` | Override provider |
| `--base-url URL` | Override endpoint |
| `--out-dir PATH` | Override output directory |
| `--resume-run-dir PATH` | Resume into existing run (cannot combine with `--out-dir`) |
| `--concurrency N` | Parallel API requests |
| `--debug-samples N` | Save N debug images (hand_object only) |

### `barista-vlm-judge`

Run G-Eval judge evaluation on a completed run directory (post-hoc). Supports `visual_qa` and `referring` tasks.

```bash
barista-vlm-judge runs/vlm/<run_dir>/
```

| Flag | Description |
|------|-------------|
| `run_dir` | Path to run directory with predictions.jsonl (required, positional) |
| `--judge-config` | Path to a run config JSON to use for the judge model |
| `--judge-model NAME` | Override judge model name |
| `--judge-provider NAME` | Override judge provider |
| `--concurrency N` | Parallel judge API requests (default 1) |
| `--force` | Re-judge all predictions, even those with existing scores |
| `--criteria C [C ...]` | Subset of criteria to evaluate (default: all for the task) |
| `--strict` | Enable strict mode (0-or-1 scoring) |
| `--verbose` | Enable verbose deepeval output |

## Task-specific workflows

### activity_mcq

```bash
barista-vlm-build-dataset --config configs/vlm_benchmarks/activity_mcq/build.json --out examples.jsonl
barista-vlm-run --config configs/vlm_benchmarks/activity_mcq/gemini.json --examples-path examples.jsonl
```

`task_params`: `frames_per_example` (int), `num_choices` (int or null for full vocab).

Activity labels are read directly from the `display_name` field of each activity in the dataset.

### grounding

> **Bbox format note:** GT boxes are always stored as `xyxy` in the frozen JSONL (canonical format). The `bbox_format` in run configs controls only the prompt instructions and model response parsing — `yxyx` for Gemini/Gemma/GPT, `xyxy` for Qwen. A single built dataset works for all providers.

**Usage:**

`task_params`: `min_bbox_area` (float | null), `video_id` (str | null), `sample_fraction` (float | null).

```bash
# 1. Build frozen examples with referring descriptions (no LLM calls)
barista-vlm-build-dataset \
  --config configs/vlm_benchmarks/grounding/build.json \
  --out examples.jsonl

# 2a. Evaluate with VLM
barista-vlm-run \
  --config configs/vlm_benchmarks/grounding/gemini.json \
  --examples-path examples.jsonl

# 2b. Evaluate with SAM 3 (separate conda env; uses the same frozen JSONL)
conda activate sam3
python third_party/sam3/run_evaluation.py \
  --dataset-root /data/barista \
  --benchmark-jsonl examples.jsonl \
  --checkpoint /path/to/sam3.pt
```

See [`third_party/sam3/README.md`](../../third_party/sam3/README.md) for SAM 3 setup.

### hand_object

Ground truth from `relation_type == "human_actions"` relations.  Hand type inferred from category name (`"left hand"` / `"right hand"`).

> **Bbox format note:** Same as grounding — GT boxes are stored as `xyxy` canonically in the JSONL. The run config's `bbox_format` controls prompt instructions and parsing only.

```bash
barista-vlm-build-dataset --config configs/vlm_benchmarks/hand_object/build.json --out examples.jsonl
barista-vlm-run --config configs/vlm_benchmarks/hand_object/gemini.json --examples-path examples.jsonl
```

**CaRe-Ego specialist baseline** (separate conda env):

```bash
bash third_party/careego/install.sh
conda activate careego
python third_party/careego/run_evaluation.py \
  --dataset-root /data/barista \
  --benchmark-jsonl runs/vlm_datasets/hand_object_data.jsonl \
  --checkpoint third_party/careego/weights/best_mIoU_ckpt.pth \
  --out-dir runs/careego
```

See [`third_party/careego/README.md`](../../third_party/careego/README.md) for details.

### visual_qa

Asset-backed: prepare QA pairs first.

```bash
barista-vlm-prepare --config configs/vlm_benchmarks/visual_qa/prepare.json
barista-vlm-build-dataset --config configs/vlm_benchmarks/visual_qa/build.json --out examples.jsonl
barista-vlm-run --config configs/vlm_benchmarks/visual_qa/gemini.json --examples-path examples.jsonl
```

**Post-hoc judging** (recommended for multi-model comparison): set `"geval_criteria": []` in `task_params` to skip inline judging, then judge separately:

```bash
barista-vlm-judge runs/vlm/<run_dir>/
barista-vlm-judge runs/vlm/<run_dir>/ --judge-model gemini-2.5-pro-preview-05-06 --concurrency 4
```

`task_params` for **prepare**: `stride` (int, 30), `clip_length_in_frames` (int, 4), `frame_spacing` (int, 30), `categorized` (bool, false), `qa_generation_mode` (`"template"` | `"llm"`), `qa_pairs_dir` (str | null), `video_id` (str | null).

`task_params` for **build / run**: `qa_pairs_dir` (str, required), `max_frames_per_example` (int | null), `judge_model` (dict | null), `geval_criteria` (list | null, empty `[]` to skip inline judging), `video_id` (str | null).

### referring

```bash
barista-vlm-build-dataset --config configs/vlm_benchmarks/referring/build.json --out examples.jsonl
barista-vlm-run --config configs/vlm_benchmarks/referring/gemini.json --examples-path examples.jsonl
```

**Post-hoc judging** (recommended for multi-model comparison): set `"geval_criteria": []` in `task_params` to skip inline judging, then judge separately:

```bash
barista-vlm-judge runs/vlm/<run_dir>/
barista-vlm-judge runs/vlm/<run_dir>/ --judge-model gemini-2.5-pro-preview-05-06 --force
```

`task_params` for **build**: `frame_step` (int, 10), `max_objects_per_frame` (int | null), `activity_filter` (bool, false), `video_id` (str | null).

`task_params` for **run**: `judge_model` (dict | null), `geval_criteria` (list | null, empty `[]` to skip inline judging), `geval_strict_mode` (bool), `geval_verbose` (bool).

### relation_extraction

```bash
barista-vlm-build-dataset --config configs/vlm_benchmarks/relation_extraction/build.json --out examples.jsonl
barista-vlm-run --config configs/vlm_benchmarks/relation_extraction/gemini.json --examples-path examples.jsonl
```

## Cross-model comparison workflow

Build once, evaluate with multiple models:

```bash
# Freeze examples
barista-vlm-build-dataset --config configs/vlm_benchmarks/activity_mcq/build.json --out examples.jsonl

# Gemini
barista-vlm-run --config configs/vlm_benchmarks/activity_mcq/gemini.json --examples-path examples.jsonl

# Azure OpenAI
barista-vlm-run --config configs/vlm_benchmarks/activity_mcq/azure_openai.json --examples-path examples.jsonl

# Local vLLM
barista-vlm-run \
  --config configs/vlm_benchmarks/activity_mcq/qwen_openai_compat.json \
  --examples-path examples.jsonl \
  --base-url http://localhost:8000/v1
```

Resume after interruption:

```bash
barista-vlm-run \
  --config configs/vlm_benchmarks/activity_mcq/gemini.json \
  --examples-path examples.jsonl \
  --resume-run-dir runs/vlm/<existing_run_id>
```

## Local vLLM quick start

Use `scripts/serve_vllm.py` to launch a vLLM server with automatic tensor-parallelism and per-model defaults:

```bash
# Qwen3.5-27B
python scripts/serve_vllm.py --model Qwen/Qwen3.5-27B

# Gemma 4 31B (adds --reasoning-parser, --tool-call-parser, --chat-template, etc. automatically)
python scripts/serve_vllm.py --model google/gemma-4-31B-it

# Override GPU count or image limit
python scripts/serve_vllm.py --tp 4 --mm-image-limit 4
```

Then run a benchmark against the local server:

```bash
barista-vlm-run \
  --config configs/vlm_benchmarks/activity_mcq/qwen_openai_compat.json \
  --examples-path examples.jsonl \
  --base-url http://localhost:8000/v1
```

- Set `--mm-image-limit` ≥ `task_params.frames_per_example`.
- Any unrecognised flags are forwarded verbatim to vLLM.

## Run output

Artifacts under `output_dir/<run_id>/`:

| File | Content |
|------|---------|
| `predictions.jsonl` | Per-example parsed + evaluated results |
| `errors.jsonl` | Failed examples |
| `metrics.json` | Accuracy, per-label breakdowns, timing |
| `run_config.json` | Snapshot of the config used |
| `summary.txt` | Human-readable summary |

`metrics.json` timing fields: `wall_clock_ms`, `mean_latency_ms`, `min_latency_ms`, `max_latency_ms`, `concurrency`.
For `visual_qa, referring`: per-criterion G-Eval scores (0–1) and `mean_geval_score`.
