from __future__ import annotations

import argparse
import logging
from pathlib import Path

from barista.vlm_benchmarks.config import apply_cli_overrides, load_run_config
from barista.vlm_benchmarks.runner import load_benchmark_examples, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a VLM benchmark batch from a JSON config.')
    parser.add_argument('--config', type=Path, required=True, help='Path to benchmark JSON config.')
    parser.add_argument('--model', help='Override model name.')
    parser.add_argument('--provider', help='Override provider name.')
    parser.add_argument('--base-url', help='Override model base_url (useful for local vLLM/SGLang endpoints).')
    parser.add_argument('--out-dir', type=Path, help='Override run output root directory.')
    parser.add_argument('--examples-path', type=Path, required=True, help='Pre-built benchmark examples JSONL file.')
    parser.add_argument(
        '--resume-run-dir',
        type=Path,
        help='Resume/append into an existing run directory and skip completed examples.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        help='Number of parallel API requests (default 1, sequential).',
    )
    parser.add_argument(
        '--debug-samples',
        type=int,
        default=5,
        metavar='N',
        help='Save N frames with GT (green) and pred (red) bboxes to run_dir/debug/ (hand_object task only).',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    logging.getLogger('google_genai').setLevel(logging.WARNING)

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_run_config(args.config)
        if args.resume_run_dir is not None and args.out_dir is not None:
            raise ValueError('--resume-run-dir cannot be combined with --out-dir')

        config = apply_cli_overrides(
            config,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            out_dir=args.out_dir,
            concurrency=args.concurrency,
            debug_samples=args.debug_samples,
        )
        examples = load_benchmark_examples(args.examples_path)
        if not examples:
            raise ValueError(f'No examples found in {args.examples_path}')
        result = run_benchmark(
            config,
            examples=examples,
            examples_source_path=args.examples_path,
            run_dir=args.resume_run_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(result.summary_text)

    n_debug = int(result.metrics.get('debug_frames_saved', 0))
    if n_debug > 0:
        print(f'Debug frames: {result.run_dir / "debug"} ({n_debug} images)')

    if int(result.metrics['examples_total']) == 0:
        raise SystemExit('No examples were generated for this config.')
    if int(result.metrics['provider_successes']) == 0:
        raise SystemExit('All provider calls failed.')


if __name__ == '__main__':
    main()
