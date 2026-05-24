#!/usr/bin/env python3
"""
skillmem recall script.

Retrieves knowledge and emits a recall-event for future reconsolidation.
No LLM call required.

Usage:
    python skill/scripts/recall.py \
        --query "What do I know about X?" \
        [--skill summarizer] \
        [--top-k 5] \
        [--hint 0.6] \
        [--data-dir .skillmem]

Output: JSON array of SearchResult objects to stdout.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve knowledge from etchmem."
    )
    parser.add_argument("--query", required=True, help="The recall query.")
    parser.add_argument("--skill", default=None, help="Optional skill scope filter.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Max results to return (default from config).",
    )
    parser.add_argument(
        "--hint",
        type=float,
        default=None,
        help="Importance prior seeded into the recall-event.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the Chroma data directory (default: .skillmem/).",
    )
    args = parser.parse_args()

    from etchmem import Config, Engine

    config_kwargs: dict = {}
    if args.data_dir:
        config_kwargs["data_dir"] = args.data_dir

    engine = Engine(config=Config(**config_kwargs) if config_kwargs else None)
    results = engine.recall(
        query=args.query,
        skill=args.skill,
        top_k=args.top_k,
        hint=args.hint,
    )

    output = [dataclasses.asdict(r) for r in results]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
