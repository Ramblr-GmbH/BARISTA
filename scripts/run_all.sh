#!/bin/bash
# Run all VLM benchmark tasks with a specific provider.
# Usage: ./scripts/run_all.sh <provider>
# Provider options: gemini, gemma_openai_compat, qwen_openai_compat, azure_openai
#
# For openai_compat providers (qwen_openai_compat, gemma_openai_compat), start the
# vLLM server in a separate terminal before running this script:
#
#   Qwen3.5-27B
#     python scripts/serve_vllm.py --model Qwen/Qwen3.5-27B
#
#   Gemma-4-31B
#     python scripts/serve_vllm.py --model google/gemma-4-31B-it
#
# Parallelism flags:
#   --tp N   Tensor parallelism: splits one model replica across N GPUs.
#            Auto-detected from model presets; override only when needed.
#
#   --dp N   Data parallelism: runs N independent model replicas.
#            Increases throughput. Total GPUs used = tp × dp.
#            Example (4 GPUs: 2-way TP × 2 replicas):
#              python scripts/serve_vllm.py --model Qwen/Qwen3.5-27B --dp 2
#
# Multi-frame tasks:
#   --mm-image-limit N   Max video frames per prompt (default: 1).
#   Tasks like activity prediction and visual QA send multiple frames, so raise this
#   if the server rejects requests with "too many images":
#     python scripts/serve_vllm.py --model Qwen/Qwen3.5-27B --mm-image-limit 8

PROVIDER=${1:?Usage: $0 <provider>}

# Run activity MCQ task
barista-vlm-run --config configs/vlm_benchmarks/activity_mcq/${PROVIDER}.json --examples-path runs/vlm_datasets/activity_data.jsonl

# Run grounding task
barista-vlm-run --config configs/vlm_benchmarks/grounding/${PROVIDER}.json --examples-path runs/vlm_datasets/grounding_data.jsonl

# Run hand-object interaction task
barista-vlm-run --config configs/vlm_benchmarks/hand_object/${PROVIDER}.json --examples-path runs/vlm_datasets/hand_object_data.jsonl

# Run relation extraction task
barista-vlm-run --config configs/vlm_benchmarks/relation_extraction/${PROVIDER}.json --examples-path runs/vlm_datasets/relation_extraction_data.jsonl

# Run referring task
barista-vlm-run --config configs/vlm_benchmarks/referring/${PROVIDER}.json --examples-path runs/vlm_datasets/referring_data.jsonl

# Run visual QA task
barista-vlm-run --config configs/vlm_benchmarks/visual_qa/${PROVIDER}.json --examples-path runs/vlm_datasets/visual_qa_data.jsonl
