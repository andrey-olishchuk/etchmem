"""
Affectability scoring — formation gate for orphaned buffer records.

Affectability is an attribute of a single memory record that governs
whether it promotes to injected knowledge when it has not (yet) clustered
with other records.

Rules (from the spec):
  - Default score: 0.5 (when no hint provided)
  - Hint from agent seeds the score (treated as prior, not truth)
  - Cluster membership BYPASSES the affectability gate entirely —
    clustered records are always evaluated for formation/reconsolidation
    regardless of their individual affectability scores.
  - The gate only applies to ORPHANED (singleton-cluster) records.

Thresholds (configurable in Config):
  score < drop_below  → drop from buffer (noise)
  drop_below ≤ score < keep_below → keep in buffer (accumulate)
  score ≥ keep_below  → promote to injected

Accumulation:
  When a singleton record survives one consolidation run (kept in buffer),
  its affectability may rise — the idea is that persistence itself is a
  weak signal of relevance. In v0 this is a simple additive nudge
  (accumulate_delta per survival). This keeps routing deterministic and
  avoids LLM calls.
"""
from __future__ import annotations

from etchmem.models import Affectability


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SCORE: float = 0.5
ACCUMULATE_DELTA: float = 0.05   # nudge per consolidation survival


# ── Functions ─────────────────────────────────────────────────────────────────

def seed(record_id: str, hint: float | None) -> Affectability:
    """
    Create an Affectability seeded from an agent hint (or default 0.5).

    The hint is a prior, not truth — it is clamped to [0, 1].
    """
    if hint is not None:
        score = max(0.0, min(1.0, float(hint)))
        return Affectability(record_id=record_id, score=score, hint_seeded=True)
    return Affectability(record_id=record_id, score=DEFAULT_SCORE, hint_seeded=False)


def accumulate(aff: Affectability) -> Affectability:
    """
    Nudge score upward by ACCUMULATE_DELTA (capped at 1.0).

    Called when a mid-band orphan survives a consolidation run without
    being promoted.  Returns a new Affectability instance (immutable update).
    """
    new_score = min(1.0, aff.score + ACCUMULATE_DELTA)
    return Affectability(
        record_id=aff.record_id,
        score=new_score,
        hint_seeded=aff.hint_seeded,
    )


def cluster_rise(affs: list[Affectability]) -> float:
    """
    Compute the effective affectability for a cluster of records.

    The cluster score is the MAXIMUM of its members' scores, reflecting
    that one strong signal can pull the whole cluster toward promotion.
    Returns a scalar score (not an Affectability object — the cluster
    itself is not a tracked entity).
    """
    if not affs:
        return DEFAULT_SCORE
    return max(a.score for a in affs)


def route(
    score: float,
    has_recall_events: bool,
    drop_below: float,
    keep_below: float,
) -> str:
    """
    Deterministically route a cluster based on its effective affectability.

    Returns one of: "drop" | "keep" | "form" | "reconsolidate"

    Note: clustered (non-singleton) records always receive score=1.0 from
    the caller so they always land in "form" or "reconsolidate".  The gate
    only meaningfully filters orphaned singletons.
    """
    if score < drop_below:
        return "drop"
    if score < keep_below:
        return "keep"
    # score >= keep_below
    if has_recall_events:
        return "reconsolidate"
    return "form"
