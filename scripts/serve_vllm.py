"""
vLLM Server
===========
Starts an OpenAI-compatible HTTP server backed by vLLM with automatic
tensor parallelism configuration based on model size.

Usage:
    python serve_vllm.py                                              # default: Qwen3.5 9B
    python serve_vllm.py --model Qwen/Qwen3.5-27B
    python serve_vllm.py --model google/gemma-4-31B-it
    python serve_vllm.py --tp 4                                        # override tensor-parallel GPU count
    python serve_vllm.py --dp 2                                        # data-parallel replicas (higher throughput)
    python serve_vllm.py --tp 2 --dp 2                                 # 4 GPUs total: 2 replicas × 2-way TP
    python serve_vllm.py --mm-image-limit 8                            # allow up to 8 frames per prompt
    python serve_vllm.py --mm-processor-kwargs '{"max_soft_tokens": 560}'  # extra vLLM args

    Any unrecognized arguments are forwarded verbatim to vLLM.

Parallelism flags:
    --tp N   Tensor parallelism: splits one model replica across N GPUs.
             Use when a single model does not fit on one GPU.
             Auto-detected from model presets when omitted.

    --dp N   Data parallelism: runs N independent model replicas.
             Each replica uses --tp GPUs, so total GPUs = tp × dp.
             Increases request throughput without changing per-GPU memory.
             Requires vLLM ≥ 0.8.0 and --enable-expert-parallel-size on
             MoE models if applicable.

Available models:

  Qwen3.5:
    Qwen/Qwen3.5-2B                  ~4 GB
    Qwen/Qwen3.5-4B                  ~10 GB
    Qwen/Qwen3.5-9B                  ~20 GB
    Qwen/Qwen3.5-27B                 ~56 GB

  Gemma 4:
    google/gemma-4-31B-it            ~66 GB
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Model HF name → (min TP, estimated VRAM GB total)
MODEL_PRESETS = {
    'Qwen/Qwen3.5-2B': (1, 4),
    'Qwen/Qwen3.5-4B': (1, 10),
    'Qwen/Qwen3.5-9B': (1, 20),
    'Qwen/Qwen3.5-27B': (2, 56),
    'google/gemma-4-31B-it': (2, 66),
}

DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'

_SCRIPT_DIR = Path(__file__).parent

_QWEN3_EXTRA = ['--reasoning-parser', 'qwen3']

# Model-specific extra vLLM args appended after the base command
MODEL_EXTRA_ARGS: dict[str, list[str]] = {
    'Qwen/Qwen3.5-2B': _QWEN3_EXTRA,
    'Qwen/Qwen3.5-4B': _QWEN3_EXTRA,
    'Qwen/Qwen3.5-9B': _QWEN3_EXTRA,
    'Qwen/Qwen3.5-27B': _QWEN3_EXTRA,
    'google/gemma-4-31B-it': [
        '--reasoning-parser',
        'gemma4',
        '--mm-processor-kwargs',
        '{"max_soft_tokens": 1120}',
        '--default-chat-template-kwargs',
        '{"enable_thinking": true}',
        '--tool-call-parser',
        'gemma4',
        '--enable-auto-tool-choice',
        '--chat-template',
        str(_SCRIPT_DIR / 'chat_template_gemma4.jinja'),
    ],
}


def _format_requirements(est_vram_gb_total: int | None) -> str:
    if est_vram_gb_total is None:
        return 'VRAM: unknown'
    return f'est VRAM total: ~{est_vram_gb_total:.0f} GB'


def _build_vllm_cmd(
    *,
    model: str,
    host: str,
    port: int,
    tp: int,
    dp: int,
    gpu_memory_utilization: float,
    served_name: str,
    mm_image_limit: int,
    mm_processor_cache_type: str,
) -> list[str]:
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        model,
        '--host',
        host,
        '--port',
        str(port),
        '--tensor-parallel-size',
        str(tp),
        '--data-parallel-size',
        str(dp),
        '--pipeline-parallel-size',
        '1',
        '--gpu-memory-utilization',
        f'{gpu_memory_utilization:.4f}',
        '--max-model-len',
        '65536',
        '--max-num-seqs',
        '2',
        '--trust-remote-code',
        '--dtype',
        'bfloat16',
        '--limit-mm-per-prompt',
        json.dumps({'image': int(mm_image_limit)}),
        '--served-model-name',
        served_name,
        '--enable-prefix-caching',
        '--mm-processor-cache-type',
        mm_processor_cache_type,
    ]

    return cmd


def main():
    presets_help = ', '.join(sorted(MODEL_PRESETS.keys()))
    parser = argparse.ArgumentParser(description='Launch vLLM server')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--model', type=str, default=None, help=f'Model name/path or shorthand ({presets_help})')
    parser.add_argument(
        '--tp', type=int, default=None, help='Tensor-parallel GPU count (auto-detected from model if omitted)'
    )
    parser.add_argument('--dp', type=int, default=1, help='Data-parallel replicas (default: 1). Total GPUs = tp × dp.')
    parser.add_argument(
        '--mm-image-limit',
        type=int,
        default=1,
        help='Max images per prompt (default: 1). Increase for multi-frame detection/tracking.',
    )
    parser.add_argument(
        '--hf-home',
        type=str,
        default='/data/hf_home',
        help='HuggingFace cache directory (sets HF_HOME, default: /data/hf_home)',
    )
    parser.add_argument(
        '--mm-processor-cache-type',
        choices=['lru', 'shm'],
        default='shm',
        help='Multimodal processor cache type (default: shm)',
    )
    args, extra_vllm_args = parser.parse_known_args()

    os.environ['HF_HOME'] = args.hf_home

    # Resolve model name
    model = args.model or DEFAULT_MODEL
    preset_info = MODEL_PRESETS.get(model)

    # Resolve TP size
    tp = args.tp or (preset_info[0] if preset_info else 1)
    served_name = model.split('/')[-1].lower()

    # Print minimum requirements for the selected preset (if known)
    if preset_info:
        min_gpus, est_vram = preset_info
        print(f'Reqs:   {_format_requirements(est_vram)}')

    cmd_host = args.host
    cmd_port = args.port

    model_extra_args = MODEL_EXTRA_ARGS.get(model, [])

    cmd = (
        _build_vllm_cmd(
            model=model,
            host=cmd_host,
            port=cmd_port,
            tp=tp,
            dp=args.dp,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            served_name=served_name,
            mm_image_limit=args.mm_image_limit,
            mm_processor_cache_type=args.mm_processor_cache_type,
        )
        + model_extra_args
        + extra_vllm_args
    )

    print(f'Model:  {model}')
    print(f'TP:     {tp} GPU(s) × DP: {args.dp} replica(s) = {tp * args.dp} GPU(s) total')
    print(f'Served: {served_name}')

    print('\nStarting vLLM server with command:')
    print('  ' + ' '.join(cmd))
    print(f'\nServer will be available at http://{cmd_host}:{cmd_port}')
    print('Press Ctrl+C to stop.\n')

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
