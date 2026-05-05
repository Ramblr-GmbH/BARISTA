#!/bin/bash

# Build the activity MCQ dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/activity_mcq/build.json --out runs/vlm_datasets/activity_data.jsonl

# Build the grounding dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/grounding/build.json --out runs/vlm_datasets/grounding_data.jsonl

# Build the hand-object interaction dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/hand_object/build.json --out runs/vlm_datasets/hand_object_data.jsonl

# Build the relation extraction dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/relation_extraction/build.json --out runs/vlm_datasets/relation_extraction_data.jsonl

# Build the referring dataset
barista-vlm-build-dataset --config configs/vlm_benchmarks/referring/build.json --out runs/vlm_datasets/referring_data.jsonl

# Build the visual QA dataset (uses LLM to generate the data)
barista-vlm-prepare --config configs/vlm_benchmarks/visual_qa/prepare.json
barista-vlm-build-dataset --config configs/vlm_benchmarks/visual_qa/build.json --out runs/vlm_datasets/visual_qa_data.jsonl
