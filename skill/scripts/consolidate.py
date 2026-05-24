#!/usr/bin/env python3
"""
skillmem consolidate script.

Runs the consolidation worker:
  intake → cluster → form/reconsolidate → reconciliation → flush.

Requires an LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) for synthesis.

Usage:
    python skill/scripts/consolidate.py \
        [--num-records all] \
        [--method LIFO] \
        [--data-dir .skillmem]

Output: JSON summary dict with run statistics to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run skillmem consolidation worker."
    )
    parser.add_argument(
        "--num-records",
        default="all",
        help='Number of relational deposits to process ("all" or integer N).',
    )
    parser.add_argument(
        "--method",
        default="LIFO",
        choices=["LIFO", "FIFO"],
        help="Ordering for deposit pull (LIFO=newest first, FIFO=oldest first).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the Chroma data directory (default: .skillmem/).",
    )
    args = parser.parse_args()

    # Parse num-records
    num_records: int | str = args.num_records
    if num_records != "all":
        try:
            num_records = int(num_records)
        except ValueError:
            print(
                f"ERROR: --num-records must be 'all' or an integer, got: {args.num_records}",
                file=sys.stderr,
            )
            sys.exit(1)

    from etchmem import Config, Engine

    config_kwargs: dict = {}
    if args.data_dir:
        config_kwargs["data_dir"] = args.data_dir

    engine = Engine(config=Config(**config_kwargs) if config_kwargs else None)

    try:
        summary = engine.consolidate(num_records=num_records, method=args.method)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
