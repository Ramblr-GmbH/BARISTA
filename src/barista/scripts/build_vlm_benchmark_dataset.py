from __future__ import annotations

import argparse
import logging
from pathlib import Path

from barista.vlm_benchmarks.config import _UNSET, apply_build_cli_overrides, load_dataset_build_config
from barista.vlm_benchmarks.runner import build_and_save_benchmark_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build and save a frozen VLM benchmark dataset from a JSON config.')
    parser.add_argument('--config', type=Path, required=True, help='Path to benchmark JSON config.')
    parser.add_argument('--out', type=Path, required=True, help='Output JSONL path for frozen benchmark examples.')
    parser.add_argument('--limit', type=int, help='Override max examples to materialize.')
    parser.add_argument('--video-id', help='Override task video_id filter.')
    parser.add_argument('--task', help='Override task name.')
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=None,
        help='Fraction of frames per video to include (0.0-1.0], evenly spaced.',
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable example shuffling (override shuffle_seed to null).',
    )
    parser.add_argument(
        '--activity-only',
        action='store_true',
        help='Restrict examples to frames inside activity segments.',
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
        config = load_dataset_build_config(args.config)
        config = apply_build_cli_overrides(
            config,
            limit=args.limit,
            task=args.task,
            video_id=args.video_id,
            shuffle_seed=None if args.no_shuffle else _UNSET,
            sample_fraction=args.sample_fraction,
            activity_filter=True if args.activity_only else _UNSET,
        )
        result = build_and_save_benchmark_dataset(config, examples_path=args.out)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(result.summary_text)
    if result.examples_total == 0:
        raise SystemExit('No examples were generated for this config.')


if __name__ == '__main__':
    main()
