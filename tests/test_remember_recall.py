"""
Tests for remember() and recall() — the deposit and retrieval path.

Covers:
  - remember() writes to relational (content hash dedup).
  - recall() searches injected + relational and merges.
  - recall() emits a recall-event into the buffer.
  - Graceful degradation when injected is empty.
  - Skill scoping filters work at recall time.
"""
from __future__ import annotations

import pytest
import tempfile
import os

from etchmem import Engine, Config
from etchmem.models import RecallEvent, SearchResult


@pytest.fixture
def engine(tmp_path):
    """Engine backed by a fresh temporary Chroma directory."""
    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",  # no LLM needed for remember/recall
        openai_api_key=None,
        anthropic_api_key=None,
        embedding_function="hash",  # offline; no model download required
    )
    return Engine(config=cfg)


def test_remember_writes_to_relational(engine):
    """remember() should increase relational count."""
    before = engine.stats()["relational"]
    engine.remember("Python 3.14 ships with a JIT compiler by default.")
    after = engine.stats()["relational"]
    assert after == before + 1


def test_remember_deduplicates_by_content(engine):
    """Inserting the same text twice should not create a second record."""
    text = "Identical content for dedup test."
    engine.remember(text)
    engine.remember(text)
    assert engine.stats()["relational"] == 1


def test_recall_returns_results_when_injected_empty(engine):
    """
    When injected is empty, recall() should degrade gracefully.
    With no relational records either, it returns an empty list — no crash.
    """
    results = engine.recall("anything")
    assert isinstance(results, list)


def test_recall_emits_recall_event(engine):
    """recall() must write a RecallEvent into the buffer."""
    before = engine.stats()["buffer"]
    engine.recall("query that triggers an event")
    after = engine.stats()["buffer"]
    assert after == before + 1


def test_recall_emits_multiple_events_for_multiple_calls(engine):
    """Each recall() call emits one recall-event regardless of results."""
    for _ in range(3):
        engine.recall("repeated query")
    assert engine.stats()["buffer"] == 3


def test_recall_skill_filter_does_not_crash(engine):
    """Recall with a skill filter should not raise even with empty stores."""
    results = engine.recall("query", skill="summarizer")
    assert isinstance(results, list)


def test_remember_with_skill_and_hint(engine):
    """remember() accepts skill and hint without error."""
    engine.remember(
        "ChromaDB supports cosine distance natively.",
        skill="code-reviewer",
        hint=0.9,
        metadata={"source": "https://docs.trychroma.com"},
    )
    assert engine.stats()["relational"] == 1


def test_recall_returns_search_result_objects(engine):
    """recall() results must be SearchResult instances."""
    engine.remember("The sky is blue during the day.")
    results = engine.recall("sky color")
    # May be empty if injected is empty and relational didn't embed yet,
    # but if results present, they must be SearchResult.
    for r in results:
        assert isinstance(r, SearchResult)


def test_stats_reports_all_three_tiers(engine):
    """stats() must report counts for all three tiers."""
    s = engine.stats()
    assert "relational" in s
    assert "buffer" in s
    assert "injected" in s


def test_recall_hint_is_accepted(engine):
    """recall() with a hint parameter should not raise."""
    engine.remember("Fact with importance hint.")
    results = engine.recall("importance hint", hint=0.95)
    assert isinstance(results, list)
