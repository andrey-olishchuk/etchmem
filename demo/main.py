"""
etchmem demo — pydantic-ai CLI with persistent memory.

The etchmem skill is a FunctionToolset whose instructions are loaded directly
from etchmem/skill/SKILL.md. Tools wrap the skill scripts via subprocess.

Run:
    python main.py
    (OPENAI_API_KEY loaded from .env in this folder)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

load_dotenv(Path(__file__).parent / ".env")

# ── paths ─────────────────────────────────────────────────────────────────────

DEMO_DIR    = Path(__file__).parent
SKILL_DIR   = DEMO_DIR.parent / "skill"
SCRIPTS_DIR = SKILL_DIR / "scripts"
DATA_DIR    = str(DEMO_DIR / ".etchmem")


def _run_script(script: str, args: list[str]) -> str:
    """Run an etchmem skill script and return its stdout."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + args + ["--data-dir", DATA_DIR]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"{script} exited {proc.returncode}")
    return proc.stdout.strip()


# ── etchmem skill — instructions from SKILL.md ───────────────────────────────

_skill_md = (SKILL_DIR / "SKILL.md").read_text()

etchmem_skill = FunctionToolset(instructions=_skill_md)


@etchmem_skill.tool_plain
def remember(data: str) -> str:
    """Deposit text into etchmem relational memory (skill/scripts/remember.py)."""
    raw = _run_script("remember.py", ["--data", data, "--skill", "chat"])
    return json.loads(raw).get("message", "Stored.")


@etchmem_skill.tool_plain
def recall(query: str) -> str:
    """Retrieve relevant memories from etchmem (skill/scripts/recall.py).
    Returns a JSON array of SearchResult objects (content + score)."""
    raw = _run_script("recall.py", ["--query", query, "--top-k", "4"])
    results = json.loads(raw) if raw else []
    if not results:
        return "No relevant memories found."
    now = time.time()
    lines = []
    for r in results:
        age_s = now - r.get("created_at", now)
        age_h = age_s / 3600
        if age_h < 1:
            age_str = f"{int(age_s / 60)}m ago"
        elif age_h < 48:
            age_str = f"{age_h:.1f}h ago"
        else:
            age_str = f"{age_h / 24:.1f}d ago"
        lines.append(f"[score={r['score']:.2f} | {age_str} | {r['source']}] {r['content']}")
    return "\n".join(lines)


@etchmem_skill.tool_plain
def consolidate() -> str:
    """Run the etchmem consolidation worker (skill/scripts/consolidate.py).
    Clusters raw deposits into durable knowledge. Requires an LLM API key."""
    raw = _run_script("consolidate.py", [])
    s = json.loads(raw)
    return (
        f"Consolidation complete — "
        f"formed={s.get('formed', 0)}, "
        f"reconsolidated={s.get('reconsolidated', 0)}, "
        f"dropped={s.get('dropped', 0)}, "
        f"kept={s.get('kept', 0)}, "
        f"flushed={s.get('flushed', 0)}"
    )


# ── agent ─────────────────────────────────────────────────────────────────────

agent = Agent(
    "openai:gpt-5.4-nano",
    system_prompt="You are a helpful assistant with persistent memory powered by etchmem.",
    toolsets=[etchmem_skill],
)


# ── CLI loop ──────────────────────────────────────────────────────────────────

async def chat_loop() -> None:
    print()
    print("┌──────────────────────────────────────────────┐")
    print("│  etchmem demo  ·  pydantic-ai + gpt-5.4-nano  │")
    print("│  type 'quit' or Ctrl-C to exit                │")
    print("└──────────────────────────────────────────────┘")
    print(f"  memory store: {DATA_DIR}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        try:
            result = await agent.run(user_input)
            print(f"\nAssistant: {result.output}\n")
        except Exception as exc:
            print(f"  ✗ error: {exc}\n")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set (add it to test/.env).", file=sys.stderr)
        sys.exit(1)
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
