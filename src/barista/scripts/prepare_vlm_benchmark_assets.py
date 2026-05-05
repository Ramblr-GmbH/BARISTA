from __future__ import annotations

import argparse
import logging
from pathlib import Path

from barista.vlm_benchmarks.config import apply_prepare_cli_overrides, load_dataset_prepare_config
from barista.vlm_benchmarks.runner import prepare_benchmark_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare task-specific VLM benchmark assets from a JSON config.')
    parser.add_argument('--config', type=Path, required=True, help='Path to asset-prepare JSON config.')
    parser.add_argument('--limit', type=int, help='Override max examples/assets to generate.')
    parser.add_argument('--video-id', help='Override task video_id filter.')
    parser.add_argument('--task', help='Override task name.')
    parser.add_argument('--model', help='Override model name.')
    parser.add_argument('--provider', help='Override provider name.')
    parser.add_argument('--base-url', help='Override model base_url.')
    parser.add_argument('--out-dir', type=Path, help='Override working/output root directory.')
    parser.add_argument('--concurrency', type=int, help='Override generation concurrency.')
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
        config = load_dataset_prepare_config(args.config)
        config = apply_prepare_cli_overrides(
            config,
            limit=args.limit,
            task=args.task,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            out_dir=args.out_dir,
            video_id=args.video_id,
            concurrency=args.concurrency,
        )
        result = prepare_benchmark_assets(config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(result.summary_text)
    if result.artifacts_total == 0:
        raise SystemExit('No assets were generated for this config.')


if __name__ == '__main__':
    main()
