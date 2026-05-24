#!/usr/bin/env python3
"""
skillmem remember script.

Deposits a raw record into the relational collection.
No LLM call required.

Usage:
    python skill/scripts/remember.py \
        --data "Text to remember" \
        [--skill summarizer] \
        [--hint 0.8] \
        [--metadata '{"source": "https://example.com"}']
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deposit text into skillmem's relational memory."
    )
    parser.add_argument("--data", required=True, help="Text content to remember.")
    parser.add_argument("--skill", default=None, help="Optional skill scope name.")
    parser.add_argument(
        "--hint",
        type=float,
        default=None,
        help="Importance prior 0–1 (seed, not truth).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON dict of arbitrary metadata (e.g. source URL).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the Chroma data directory (default: .skillmem/).",
    )
    args = parser.parse_args()

    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"ERROR: --metadata is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    from etchmem import Config, Engine

    config_kwargs: dict = {}
    if args.data_dir:
        config_kwargs["data_dir"] = args.data_dir

    engine = Engine(config=Config(**config_kwargs) if config_kwargs else None)
    engine.remember(
        data=args.data,
        hint=args.hint,
        skill=args.skill,
        metadata=metadata,
    )
    print(json.dumps({"status": "ok", "message": "Record deposited."}))


if __name__ == "__main__":
    main()
