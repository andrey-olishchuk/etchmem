"""
Engine — the public entry point.

Wires together:
  - three ChromaStore instances (relational, buffer, injected)
  - one RemoteConnector for LLM synthesis
  - Recaller (recall logic + recall-event emission)
  - Worker (consolidation pipeline)

Exposes exactly three public methods:
  remember(data, hint, skill, metadata)   → None
  recall(query, skill, top_k, hint)       → list[SearchResult]
  consolidate(num_records, method)        → dict (summary)
"""
from __future__ import annotations

import time
from typing import Any

from etchmem.config import Config
from etchmem.models import Record, SearchResult
from etchmem.stores.chroma_store import build_stores
from etchmem.util.hashing import content_hash


class Engine:
    """
    The skillmem engine.

    Usage::

        from etchmem import Engine

        engine = Engine()                      # uses .skillmem/ in cwd
        engine.remember("Python 3.14 ships with a new JIT by default")
        results = engine.recall("what changed in Python 3.14?")
        summary = engine.consolidate()
    """

    def __init__(
        self,
        config: Config | None = None,
        connector=None,   # LLMConnector | None; lazy-built if None
    ) -> None:
        from etchmem.config import DEFAULT_CONFIG
        self._cfg = config or DEFAULT_CONFIG

        # Build stores
        self._relational, self._buffer, self._injected = build_stores(
            data_dir=self._cfg.data_dir,
            collection_relational=self._cfg.collection_relational,
            collection_buffer=self._cfg.collection_buffer,
            collection_injected=self._cfg.collection_injected,
            ttl_seconds=self._cfg.relational_ttl_seconds,
            embedding_function=self._cfg.embedding_function,
        )

        # Connector — built lazily on first consolidate() if not provided
        self._connector = connector
        self._connector_built = connector is not None

        # Wires
        from etchmem.recall.recaller import Recaller
        self._recaller = Recaller(
            relational_store=self._relational,
            buffer_store=self._buffer,
            injected_store=self._injected,
            config=self._cfg,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def remember(
        self,
        data: str,
        hint: float | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Deposit a raw record into the relational collection.

        No LLM call, no clustering. Cheap — embedding computed by Chroma
        on insert.

        Args:
            data:     The text content to remember.
            hint:     Optional importance prior (float 0–1). Seed, not truth.
            skill:    Optional skill scope name (e.g. "summarizer").
            metadata: Arbitrary dict (source URL, tags, etc.).
        """
        record = Record(
            id=content_hash(data),
            content=data,
            skill=skill,
            hint=hint,
            metadata=metadata or {},
            created_at=time.time(),
        )
        self._relational.add(record)

    def recall(
        self,
        query: str,
        skill: str | None = None,
        top_k: int | None = None,
        hint: float | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve knowledge AND emit a recall-event for future reconsolidation.

        Two things happen:
          1. Returns merged results from injected (primary) + relational (fresh).
          2. Writes a RecallEvent into the buffer so the worker can rewrite
             used knowledge against fresh context at the next consolidate().

        Args:
            query:  Natural language query.
            skill:  Optional skill scope filter.
            top_k:  Max results. Default from config.
            hint:   Importance prior seeded into the recall-event.

        Returns:
            List of SearchResult, highest score first.
        """
        return self._recaller.recall(
            query=query,
            skill=skill,
            top_k=top_k,
            hint=hint,
        )

    def consolidate(
        self,
        num_records: int | str | None = None,
        method: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the consolidation worker.

        Processes the buffer: forms new knowledge, reconsolidates recalled
        knowledge, hard-deletes superseded articles. Explicitly called in v0
        (predictable, demoable, debuggable).

        Args:
            num_records: How many relational deposits to pull ("all" or int N).
            method:      Ordering: "LIFO" (newest first, default) or "FIFO".

        Returns:
            Summary dict with counts of: formed, reconsolidated, dropped, kept,
            flushed, superseded.
        """
        self._ensure_connector()

        from etchmem.consolidate.worker import Worker
        worker = Worker(
            relational_store=self._relational,
            buffer_store=self._buffer,
            injected_store=self._injected,
            connector=self._connector,
            config=self._cfg,
        )

        num_records = num_records if num_records is not None else self._cfg.consolidate_default_num_records
        method = method or self._cfg.consolidate_default_method

        return worker.run(num_records=num_records, method=method)

    # ── Internals ─────────────────────────────────────────────────────────

    def _ensure_connector(self) -> None:
        """Lazy-build the LLM connector on first consolidate() call."""
        if self._connector_built:
            return
        from etchmem.connector.remote import build_connector
        self._connector = build_connector(self._cfg)
        self._connector_built = True

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def config(self) -> Config:
        return self._cfg

    def stats(self) -> dict[str, int]:
        """Return collection sizes for monitoring."""
        return {
            "relational": self._relational.count(),
            "buffer": self._buffer.count(),
            "injected": self._injected.count(),
        }
