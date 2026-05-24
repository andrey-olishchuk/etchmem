"""
Recaller — the recall() implementation.

Two things happen on every recall() call:
  1. Returns results (injected primary + relational freshness enrichment).
  2. Emits a recall-event into the buffer (cheap append, no LLM).

Design rules (from spec):
  - Injected is ALWAYS searched directly with the query (never gated).
  - Relational enriches; it never gates injected results.
  - Asymmetric scoring: injected = high-trust base; relational = freshness-
    boosted, confidence-discounted.
  - Degrades gracefully to injected-only when relational is empty.
  - The recall-event stores the QUERY (pointer), not an injected doc ref.
  - The recall-event SNAPSHOTS relational hits (TTL-perishable content).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from etchmem.config import Config
from etchmem.models import RecallEvent, SearchResult
from etchmem.stores.chroma_store import ChromaStore


class Recaller:
    """
    Encapsulates all recall() logic.
    Injected by Engine; tests can instantiate directly with mock stores.
    """

    def __init__(
        self,
        relational_store: ChromaStore,
        buffer_store: ChromaStore,
        injected_store: ChromaStore,
        config: Config,
    ) -> None:
        self._relational = relational_store
        self._buffer = buffer_store
        self._injected = injected_store
        self._cfg = config

    def recall(
        self,
        query: str,
        skill: str | None = None,
        top_k: int | None = None,
        hint: float | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve blended results and emit a recall-event into the buffer.

        Args:
            query:  Natural language query.
            skill:  Optional skill scope filter.
            top_k:  Max results to return. Defaults to config.recall_top_k_injected.
            hint:   Optional importance prior seeded into the recall-event.

        Returns:
            List of SearchResult, highest score first.
        """
        top_k = top_k or self._cfg.recall_top_k_injected

        # ── 1. Search injected (primary, never gated) ─────────────────────
        injected_filter = {"skill": skill} if skill else None
        injected_results = self._injected.query(
            query_text=query,
            top_k=top_k,
            filters=injected_filter,
            weight=self._cfg.recall_injected_weight,
            source="injected",
        )

        # ── 2. Search relational (freshness signal) ───────────────────────
        relational_results = self._relational.search(
            query_text=query,
            top_k=self._cfg.recall_top_k_relational,
            skill=skill,
            weight=self._cfg.recall_relational_weight,
            source="relational",
        )

        # ── 3. Asymmetric merge ───────────────────────────────────────────
        merged = self._merge(injected_results, relational_results, top_k)

        # ── 4. Emit recall-event into buffer ──────────────────────────────
        relational_snapshot = "\n\n".join(r.content for r in relational_results)
        self._emit_recall_event(query, relational_snapshot, skill, hint)

        return merged

    # ── Helpers ───────────────────────────────────────────────────────────

    def _merge(
        self,
        injected: list[SearchResult],
        relational: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Merge injected and relational results.

        Injected results are the trusted base. Relational results are
        freshness-boosted but confidence-discounted (already weighted by
        recall_relational_weight in the store query).

        De-duplicate by id; injected wins on collision.
        Sort by score descending; return top_k.
        """
        seen: dict[str, SearchResult] = {}

        for r in injected:
            seen[r.id] = r

        for r in relational:
            if r.id not in seen:
                seen[r.id] = r
            else:
                # Injected already there — boost injected score slightly
                # by the relational overlap (corroborating signal)
                existing = seen[r.id]
                seen[r.id] = SearchResult(
                    id=existing.id,
                    content=existing.content,
                    score=existing.score + r.score * 0.1,
                    source="blended",
                    skill=existing.skill,
                    metadata=existing.metadata,
                )

        results = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _emit_recall_event(
        self,
        query: str,
        relational_snapshot: str,
        skill: str | None,
        hint: float | None,
    ) -> None:
        """Write a RecallEvent into the buffer collection."""
        event_id = str(uuid.uuid4())
        event = RecallEvent(
            id=event_id,
            query=query,
            relational_snapshot=relational_snapshot,
            skill=skill,
            hint=hint,
            created_at=time.time(),
        )
        self._buffer.upsert([event])
