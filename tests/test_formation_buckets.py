"""
Tests for affectability-based formation routing.

Covers:
  - Records with hint < 0.3 are dropped (noise).
  - Records with 0.3 ≤ hint < 0.7 are kept in buffer (accumulate).
  - Records with hint ≥ 0.7 (and no cluster partner) are promoted.
  - Clustered records (≥ min_cluster_size) bypass the affectability gate.
  - accumulate() nudges score upward deterministically.
  - cluster_rise() returns the max across a cluster.
  - route() deterministic routing logic.
"""
from __future__ import annotations

import pytest

from etchmem.util.affectability import (
    DEFAULT_SCORE,
    accumulate,
    cluster_rise,
    route,
    seed,
)
from etchmem.models import Affectability


# ── Unit tests for affectability utilities ────────────────────────────────────

class TestSeed:
    def test_no_hint_gives_default(self):
        aff = seed("rec1", hint=None)
        assert aff.score == DEFAULT_SCORE
        assert not aff.hint_seeded

    def test_hint_is_used(self):
        aff = seed("rec1", hint=0.9)
        assert aff.score == 0.9
        assert aff.hint_seeded

    def test_hint_clamped_above_1(self):
        aff = seed("rec1", hint=1.5)
        assert aff.score == 1.0

    def test_hint_clamped_below_0(self):
        aff = seed("rec1", hint=-0.5)
        assert aff.score == 0.0


class TestAccumulate:
    def test_accumulate_nudges_upward(self):
        aff = seed("rec1", hint=0.5)
        aff2 = accumulate(aff)
        assert aff2.score > aff.score

    def test_accumulate_caps_at_1(self):
        aff = Affectability(record_id="rec1", score=0.99)
        aff2 = accumulate(aff)
        assert aff2.score <= 1.0

    def test_accumulate_is_immutable(self):
        aff = seed("rec1", hint=0.5)
        original_score = aff.score
        accumulate(aff)
        assert aff.score == original_score  # original unchanged


class TestClusterRise:
    def test_max_score_wins(self):
        affs = [
            Affectability("r1", score=0.2),
            Affectability("r2", score=0.8),
            Affectability("r3", score=0.5),
        ]
        assert cluster_rise(affs) == 0.8

    def test_empty_list_returns_default(self):
        assert cluster_rise([]) == DEFAULT_SCORE

    def test_single_member(self):
        affs = [Affectability("r1", score=0.65)]
        assert cluster_rise(affs) == 0.65


class TestRoute:
    def test_below_drop_threshold(self):
        assert route(0.1, has_recall_events=False, drop_below=0.3, keep_below=0.7) == "drop"

    def test_mid_band_no_recall_events(self):
        assert route(0.5, has_recall_events=False, drop_below=0.3, keep_below=0.7) == "keep"

    def test_above_promote_no_recall_events(self):
        assert route(0.8, has_recall_events=False, drop_below=0.3, keep_below=0.7) == "form"

    def test_above_promote_with_recall_events(self):
        assert route(0.8, has_recall_events=True, drop_below=0.3, keep_below=0.7) == "reconsolidate"

    def test_exactly_at_drop_boundary(self):
        # score == drop_below → keep (not drop)
        assert route(0.3, has_recall_events=False, drop_below=0.3, keep_below=0.7) == "keep"

    def test_exactly_at_keep_boundary(self):
        # score == keep_below → form (not keep)
        assert route(0.7, has_recall_events=False, drop_below=0.3, keep_below=0.7) == "form"


# ── Integration: formation routing via Engine ─────────────────────────────────

class StubConnector:
    def __init__(self):
        self.call_count = 0

    def complete(self, prompt, **opts):
        self.call_count += 1
        return (
            "Synthesized article content.\n\n"
            '{"tags": {"access_level": "public", "topic": "test", "product": "none"}}'
        )


def test_high_hint_singleton_promotes(tmp_path):
    """A singleton deposit with hint=0.9 should be promoted to injected."""
    from etchmem import Config, Engine

    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",
        embedding_function="hash",
        openai_api_key=None,
        anthropic_api_key=None,
        min_cluster_size=1,
    )
    connector = StubConnector()
    engine = Engine(config=cfg, connector=connector)

    engine.remember("High importance fact.", hint=0.9)
    engine.consolidate()

    assert engine.stats()["injected"] >= 1


def test_drop_hint_singleton_drops(tmp_path):
    """A singleton deposit with hint=0.1 should be dropped (not promoted)."""
    from etchmem import Config, Engine

    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",
        embedding_function="hash",
        openai_api_key=None,
        anthropic_api_key=None,
        min_cluster_size=2,  # force singleton path
    )
    connector = StubConnector()
    engine = Engine(config=cfg, connector=connector)

    engine.remember("Low importance noise.", hint=0.1)
    engine.consolidate()

    # Dropped — connector should NOT have been called
    assert connector.call_count == 0
    assert engine.stats()["injected"] == 0


def test_clustered_records_bypass_affectability(tmp_path):
    """
    Two records with low hints that form a cluster together should still
    be promoted (cluster bypasses individual affectability gate).
    """
    from etchmem import Config, Engine

    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",
        embedding_function="hash",
        openai_api_key=None,
        anthropic_api_key=None,
        min_cluster_size=2,
        branch_threshold=0.9,  # very wide — most docs cluster together
    )
    connector = StubConnector()
    engine = Engine(config=cfg, connector=connector)

    # Both low hint — would be dropped as singletons, but cluster together
    engine.remember("Fact alpha: chromadb embeddings are local.", hint=0.1)
    engine.remember("Fact beta: chromadb uses HNSW index.", hint=0.1)
    engine.consolidate()

    # Clustered → bypass gate → form → injected
    assert engine.stats()["injected"] >= 1
