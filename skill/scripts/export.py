#!/usr/bin/env python3
"""
skillmem export script.

Serializes the entire injected knowledge store to JSON files.

Each synthesized article is written as a separate <id>.json file under
  .etchmem/export/<UTC-timestamp>/
relative to the data directory's parent.

No LLM call required.

Usage:
    python skill/scripts/export.py \
        [--data-dir .skillmem]

Output: JSON summary dict to stdout:
    {
      "export_dir": "/abs/path/to/.etchmem/export/20260525T120000Z",
      "count": 3,
      "documents": [ { "id": ..., "content": ..., ... }, ... ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the etchmem injected knowledge store to JSON files. "
            "Writes one <id>.json file per synthesized article into "
            ".etchmem/export/<timestamp>/."
        )
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

    result = engine.export()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
