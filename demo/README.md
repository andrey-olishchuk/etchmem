# etchmem demo — pydantic-ai CLI

A minimal CLI that shows how etchmem gives an LLM persistent memory across sessions.

## What it does

1. **Recall** — before every reply, `engine.recall(query)` retrieves the most relevant past exchanges from Chroma and injects them into the system prompt.
2. **Answer** — pydantic-ai runs `gpt-4.1-nano` with the enriched context.
3. **Remember** — after every reply, `engine.remember(exchange)` stores the full Q&A so future sessions can recall it.

The memory database lives in `test/.etchmem/` and persists between runs.

## Quick start

```bash
cd etchmem/test

# install deps
pip install -r requirements.txt

# run
OPENAI_API_KEY=sk-... python main.py
```

## Special commands

| Input | Effect |
|-------|--------|
| `stats` | Print memory store sizes (relational / buffer / injected) |
| `quit` / `exit` / `q` | Exit the CLI |

## How memory consolidates

Raw exchanges land in the **relational** store. Run `engine.consolidate()` (or call
`python -c "from etchmem import Engine; Engine().consolidate()"` from this folder)
to cluster and promote them into the **injected** store — that's when etchmem actually
synthesises durable knowledge from the raw conversations.
