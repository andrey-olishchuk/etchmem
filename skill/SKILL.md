# etchmem Skill

> Thin agent-skill wrapper over the `etchmem` Python library.
> All logic lives in the library — these scripts marshal CLI arguments
> into `etchmem` and results back to stdout.

## When to use this skill

Trigger this skill when an agent needs to:
- **Deposit** new information: `remember.py`
- **Retrieve** knowledge (with automatic reconsolidation tracking): `recall.py`
- **Consolidate** the buffer into durable knowledge: `consolidate.py`
- **Export** the full knowledge store to JSON files: `export.py`

## Assumptions

- The `etchmem` Python package is installed (`pip install etchmem`).
- The Chroma data directory defaults to `./.etchmem/` relative to CWD.
  Override with `SKILLMEM_DATA_DIR` environment variable.
- An LLM API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) must be set
  for `consolidate.py` (synthesis step). `remember.py` and `recall.py`
  do not require an LLM key.

## Scripts

### `skill/scripts/remember.py`

Deposits a raw record into the relational collection.

```
python skill/scripts/remember.py \
  --data "Text to remember" \
  [--skill summarizer] \
  [--hint 0.8] \
  [--metadata '{"source": "url"}']
```

### `skill/scripts/recall.py`

Retrieves knowledge and emits a recall-event for future reconsolidation.

```
python skill/scripts/recall.py \
  --query "What do I know about X?" \
  [--skill summarizer] \
  [--top-k 5] \
  [--hint 0.6]
```

Output: JSON array of `SearchResult` objects (id, content, score, source, skill).

### `skill/scripts/consolidate.py`

Runs the consolidation worker: intake → cluster → form/reconsolidate → flush.

```
python skill/scripts/consolidate.py \
  [--num-records all] \
  [--method LIFO]
```

Output: JSON summary dict with counts (formed, reconsolidated, dropped, kept,
superseded, flushed).

### `skill/scripts/export.py`

Serializes the entire injected knowledge store to JSON files. Each synthesized
article is written as a separate `<id>.json` file under
`.etchmem/export/<UTC-timestamp>/` (sibling of the data directory). No LLM
key required. Use this to transfer institutional memory to another agent, or
as fine-tuning data shaped by real task outcomes.

```
python skill/scripts/export.py \
  [--data-dir .skillmem]
```

Output: JSON summary dict:
```json
{
  "export_dir": "/abs/path/.etchmem/export/20260525T120000Z",
  "count": 3,
  "documents": [ { "id": "...", "content": "...", "tags": {}, ... } ]
}
```

## Method-to-script mapping

| Library method | Script |
|----------------|--------|
| `engine.remember()` | `skill/scripts/remember.py` |
| `engine.recall()` | `skill/scripts/recall.py` |
| `engine.consolidate()` | `skill/scripts/consolidate.py` |
| `engine.export()` | `skill/scripts/export.py` |

## Design notes

- The skill is a **presentation layer** only. Debug against the library.
- Both surfaces (direct import and skill scripts) call **identical code**.
- The Chroma data directory is the same whether called from the library
  or the skill scripts, so data is always shared.
