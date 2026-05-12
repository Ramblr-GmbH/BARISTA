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

## Dataset access

The BARISTA dataset (annotations and videos) is available from two sources:

- **Hugging Face**: [ramblr/BARISTA](https://huggingface.co/datasets/ramblr/BARISTA)
- **Harvard Dataverse**: [preview link](https://dataverse.harvard.edu/previewurl.xhtml?token=0f9c47c0-dc2f-40a5-98cd-ac68e6603afc)

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

### Annotation structure

`load_videos(root)` returns a list of `Video` objects. Each `Video` exposes:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | `str` | Unique identifier for the video |
| `width`, `height` | `int` | Frame dimensions in pixels |
| `fps` | `float` | Frames per second |
| `frames` | `dict[int, FrameAnnotation]` | Per-frame annotations keyed by frame index |
| `activities` | `list[Activity]` | Activity segments covering frame ranges |
| `process_steps` | `list[ProcessStep]` | Process-step segments covering frame ranges |
| `categories` | `dict[str, Category]` | All object categories in this video |

Each `FrameAnnotation` (accessible via `video.frames[i]` or `video.iter_frames()`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `frame_index` | `int` | 0-based frame index |
| `width`, `height` | `int` | Frame dimensions |
| `objects` | `list[ObjectAnnotation]` | Detected/annotated object instances |
| `relations` | `list[Relation]` | Directed typed relations between objects |

Each `ObjectAnnotation` provides:

| Field | Type | Description |
|-------|------|-------------|
| `object_id` | `UUID` | Persistent identity across frames |
| `category` | `Category` | Object class (`name`, `id`) |
| `bbox` | `list[float]` | Bounding box in COCO `[x, y, w, h]` format |
| `mask` | `object` | Raw COCO RLE segmentation (`counts` + `size`); decode with `.mask_array(h, w)` |
| `attributes` | `list[Attribute]` | Key-value attributes (`attribute_type`, `value`) |

Each `Relation` (per-frame) provides:

| Field | Type | Description |
|-------|------|-------------|
| `source_object_id` | `UUID` | Subject of the relation |
| `target_object_id` | `UUID` | Object of the relation |
| `relation_type` | `str` | Relation type label |
| `value` | `str` | Relation value/qualifier |

Each `Activity` and `ProcessStep` covers a contiguous frame range (`frame_start`, `frame_end`) with a `display_name` string label. `Activity` additionally exposes `verb` and `noun` fields parsed from the display name.

## License

The **code** in this repository is released under the [Apache License 2.0](LICENSE).

The **dataset** (annotations and videos) is released under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) and is hosted on [Hugging Face](https://huggingface.co/datasets/ramblr/BARISTA) and [Harvard Dataverse](https://dataverse.harvard.edu/previewurl.xhtml?token=0f9c47c0-dc2f-40a5-98cd-ac68e6603afc).
