# etchmem

**Where agents grow up — not where they take notes.**

Modern AI agents are built from skills. Every time a skill runs, it leaves a trail: results, feedback, corrections, surprises. That trail is experience — and experience is how an agent learns how the world actually works.

`etchmem` is memory for that maturation. Not a scratchpad for the current session. Not a user profile to personalize replies. Not a long chat log you grep when context runs out. It is the layer where raw observations turn into durable understanding — how things work, what users react to, what failed last time — and keep sharpening every time the agent draws on it again.

Over time, knowledge compounds. The agent stops merely following scenarios and starts carrying a worldview shaped by use.

```bash
pip install etchmem
```

---

## What it is

`etchmem` gives your agent a persistent memory that matures through living. You deposit raw observations — facts, outcomes, feedback — scoped to a skill or left general. You retrieve with natural-language queries. Periodically you consolidate, and the engine synthesizes scattered signal into compact knowledge articles, rewriting anything that has been recalled against fresh context.

It is not a vector database and not an agent runtime. It sits above [ChromaDB](https://www.trychroma.com/), which handles all storage and embedding, and exposes a small, deliberate API: remember, recall, consolidate, export, and stats.

---

## How it works

Memory is kept in three tiers:

- **Relational** — raw, append-only records with a TTL. Fresh signal lives here.
- **Buffer** — a working space for deposits and recall-events, waiting for the next consolidation run.
- **Injected** — synthesized knowledge articles, the primary search target. Only current knowledge is kept; superseded articles are hard-deleted.

The key idea is **reconsolidation**: every `recall()` call emits a recall-event into the buffer. When you later call `consolidate()`, the engine detects which injected articles were used, checks whether fresh signal changes anything, and rewrites them if needed. Knowledge that gets recalled stays current as a side-effect of being recalled. Knowledge nobody asks about simply rests. This is how a skill — or the agent as a whole — accumulates experience instead of starting from zero each run.

Embeddings are computed locally by ChromaDB's built-in model — no external embedding API. Synthesis (the rewrite step inside `consolidate`) calls an LLM via `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

---

## API

```python
from etchmem import Engine
engine = Engine()   # persists to .etchmem/ in the current directory
```

### `remember(data, hint=None, skill=None, metadata=None)`

Deposits a text record into the relational tier. No LLM call — cheap.

- `data` — the text to store.
- `hint` — optional float 0–1, your importance signal.
- `skill` — optional scope name; lets you filter recall by agent skill.
- `metadata` — arbitrary dict (source URL, tags, …).

### `recall(query, skill=None, top_k=None, hint=None) → list[SearchResult]`

Retrieves relevant knowledge and emits a recall-event for future reconsolidation. Results are blended from injected (primary) and relational (fresh signal), ranked by a composite score.

### `consolidate(num_records="all", method="LIFO") → dict`

Runs the consolidation worker: clusters the buffer, forms new injected articles, reconsolidates recalled articles against fresh context, hard-deletes superseded ones. Returns a summary dict with counts (`formed`, `reconsolidated`, `dropped`, `kept`, `flushed`, `superseded`). Requires an LLM API key.

### `export() → dict`

Serializes the entire injected knowledge store to disk. Writes one `<id>.json` file per synthesized article into `.etchmem/export/<UTC-timestamp>/` next to the data directory. No LLM call — cheap.

Returns `{"export_dir": "...", "count": N, "documents": [...]}`.

Use this to transfer institutional memory to another agent, or to produce fine-tuning data shaped by real task outcomes rather than hand-authored examples. Experience becomes a portable artifact.

### `stats() → dict`

Returns live collection sizes for all three tiers: `{"relational": N, "buffer": N, "injected": N}`. Useful for monitoring and debugging.

---

## Hello world

```python
import os
from etchmem import Engine

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # or OPENAI_API_KEY

engine = Engine()

# Add some facts
engine.remember("The Eiffel Tower is 330 metres tall including its antenna.")
engine.remember("The tower was completed in 1889 and was originally intended to be temporary.")
engine.remember("It receives about 7 million visitors per year, making it the world's most visited paid monument.")

# Retrieve relevant knowledge
results = engine.recall("How tall is the Eiffel Tower?")
for r in results:
    print(r.score, r.content)

# Consolidate — synthesizes the raw deposits into a compact knowledge article
summary = engine.consolidate()
print(summary)
# {'formed': 1, 'reconsolidated': 0, 'dropped': 0, 'kept': 0, 'flushed': 3, 'superseded': 0}

# Subsequent recalls now hit the synthesized article,
# and any new signal deposited before the next consolidate()
# will be blended in during reconsolidation.
results = engine.recall("Eiffel Tower visitors")
print(results[0].content)

# Export the full knowledge store — one JSON file per synthesized article.
# Transfer to another agent, or use as fine-tuning data.
result = engine.export()
print(result["export_dir"])   # e.g. /your/project/.etchmem/export/20260525T120000Z
print(result["count"])        # 1
```

---

## License

MIT
