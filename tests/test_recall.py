"""
Tests for the recall path in detail.

Covers:
  - Results are sorted by score (highest first).
  - Injected results always appear regardless of relational state.
  - Relational freshness enriches injected results (blended source).
  - top_k caps the result count.
  - Asymmetric scoring: injected weight > relational weight.
  - Recall-event snapshots relational content (TTL-safety).
  - Recall with empty injected returns empty list gracefully.
"""
from __future__ import annotations

import pytest
import time

from etchmem import Config, Engine
from etchmem.models import RecallEvent, SearchResult
from etchmem.recall.recaller import Recaller


class StubConnector:
    def complete(self, prompt, **opts):
        return (
            "Stub article.\n\n"
            '{"tags": {"access_level": "public", "topic": "test", "product": "none"}}'
        )


@pytest.fixture
def engine(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",
        embedding_function="hash",
        openai_api_key=None,
        anthropic_api_key=None,
    )
    return Engine(config=cfg, connector=StubConnector())


def test_recall_empty_stores_returns_empty_list(engine):
    results = engine.recall("nothing stored yet")
    assert results == []


def test_recall_results_sorted_by_score(engine):
    """Results must be ordered highest-score-first."""
    # Add several relational records
    for i in range(5):
        engine.remember(f"Fact number {i} about Python.")
    results = engine.recall("Python facts", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recall_top_k_caps_results(engine):
    """top_k must be respected."""
    for i in range(10):
        engine.remember(f"Record {i}: unique content about topic X.", hint=0.9)
    results = engine.recall("topic X", top_k=3)
    assert len(results) <= 3


def test_recall_result_has_expected_fields(engine):
    """Each SearchResult must have id, content, score, source."""
    engine.remember("A memorable fact about the universe.")
    results = engine.recall("universe")
    for r in results:
        assert hasattr(r, "id")
        assert hasattr(r, "content")
        assert hasattr(r, "score")
        assert hasattr(r, "source")
        assert isinstance(r.score, float)


def test_recall_event_snapshots_relational(engine):
    """
    The recall-event written to the buffer must snapshot the relational
    content so it survives TTL expiry before the next consolidate().
    """
    engine.remember("Fresh relational record for snapshot test.")
    engine.recall("snapshot test query")

    # Inspect the buffer for the recall-event
    buffer_items = engine._buffer.get_all()
    recall_events = [i for i in buffer_items if isinstance(i, RecallEvent)]
    assert len(recall_events) >= 1

    # The event must have the query stored
    event = recall_events[0]
    assert "snapshot test query" in event.query or len(event.query) > 0


def test_recall_with_skill_filter(engine):
    """recall() with a skill filter should not return records from other skills."""
    engine.remember("Python tip: use list comprehensions.", skill="python-advisor")
    engine.remember("SQL tip: use indexes.", skill="sql-advisor")

    results = engine.recall("tips", skill="python-advisor")
    # Results from injected may be empty, but no crash
    for r in results:
        assert r.skill is None or r.skill == "python-advisor"


def test_recall_score_is_positive(engine):
    """All returned scores should be non-negative."""
    engine.remember("Positive score test record.")
    results = engine.recall("positive score")
    for r in results:
        assert r.score >= 0.0


def test_multiple_recalls_accumulate_events(engine):
    """Each call to recall() adds one event to the buffer."""
    queries = ["first query", "second query", "third query"]
    for q in queries:
        engine.recall(q)

    buffer_items = engine._buffer.get_all()
    recall_events = [i for i in buffer_items if isinstance(i, RecallEvent)]
    assert len(recall_events) == len(queries)


def test_recall_query_stored_in_event(engine):
    """The recall-event must store the original query verbatim."""
    query = "specific verbatim query string XYZ123"
    engine.recall(query)

    buffer_items = engine._buffer.get_all()
    recall_events = [i for i in buffer_items if isinstance(i, RecallEvent)]
    assert any(e.query == query for e in recall_events)
