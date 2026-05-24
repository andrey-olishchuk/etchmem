"""
Storage abstract base classes — the pluggability seam.

Everything in the engine talks to these ABCs, never to ChromaDB directly.
Two ports:
  - RelationalStore: append-only + TTL expiry + ANN search
  - InjectedStore:  upsert + ANN query + clustering + hard delete
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from etchmem.models import Cluster, Deposit, InjectedArticle, Record, RecallEvent, SearchResult


class RelationalStore(ABC):
    """
    Port for the *relational* collection.

    Semantics:
    - Pure append — records are NEVER mutated after insertion.
    - TTL expiry is the ONLY deletion path (swept via metadata filter).
    - No reinforcement counter. Usage signal is carried by recall-events
      written to the buffer, not by mutating relational rows.
    """

    @abstractmethod
    def add(self, record: Record) -> str:
        """
        Insert a raw Record into the relational collection.
        Returns the record id (same as record.id, passed through for
        convenience).
        The store sets expires_at = now + TTL before writing.
        """

    @abstractmethod
    def search(
        self,
        query_text: str,
        top_k: int,
        skill: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        ANN search of the relational collection.
        Optionally filter by skill (exact match on stored metadata).
        Returns up to top_k SearchResult objects.
        """

    @abstractmethod
    def expire(self) -> int:
        """
        Delete all records whose expires_at < now (TTL sweep).
        Returns the number of records deleted.
        Implemented as: collection.delete(where={"expires_at": {"$lt": now}})
        """

    @abstractmethod
    def iter_for_consolidation(
        self,
        num_records: int | str,
        method: str,
    ) -> Iterable[Record]:
        """
        Yield records for the consolidation worker to process.

        num_records: "all" or an integer N.
        method: "LIFO" (newest-first, by created_at) or "FIFO".

        Records are NOT deleted here — relational is append-only.
        Deletion happens only via TTL expiry.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of live (non-expired) records."""

    @abstractmethod
    def get_by_id(self, record_id: str) -> Record | None:
        """Fetch a single record by its id, or None if not found."""


class InjectedStore(ABC):
    """
    Port for both the *buffer* and *injected* collections.

    Buffer semantics:
    - Working set for the consolidation worker.
    - Both Deposit and RecallEvent documents are written here.
    - Flushed (hard-deleted) at the end of each successful consolidation run.

    Injected semantics:
    - Current-only knowledge store. No versions, no tombstones.
    - Superseded articles are hard-deleted on write of the new article.
    - Identity = hash of the source-doc-hash set (content-addressed).
    """

    @abstractmethod
    def upsert(self, items: list[InjectedArticle | Deposit | RecallEvent]) -> None:
        """
        Write or overwrite items. Uses item.id as the Chroma document id.
        doc_type metadata discriminator is set automatically.
        """

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        ANN search this collection.
        Returns up to top_k SearchResult objects.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Hard-delete documents by id.
        Deleting a non-existent id is a no-op (idempotent).
        Used for:
        - Buffer flush at end of run.
        - Hard-delete of superseded injected articles.
        """

    @abstractmethod
    def exists(self, content_hash: str) -> bool:
        """
        Return True if a document with this id already exists.
        Used as an idempotency guard before writing injected articles.
        """

    @abstractmethod
    def cluster(
        self,
        documents: list[Deposit | RecallEvent],
        branch_threshold: float,
    ) -> list[Cluster]:
        """
        Group documents into Cluster objects using their embeddings.
        Documents within branch_threshold cosine distance of each other
        are placed in the same cluster.
        Orphaned documents (no cluster partner) are returned as
        singleton Clusters.
        """

    @abstractmethod
    def get_all(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Deposit | RecallEvent | InjectedArticle]:
        """Return all documents, optionally filtered by metadata."""

    @abstractmethod
    def count(self) -> int:
        """Return the total document count."""

    @abstractmethod
    def get_neighbors(
        self,
        query_text: str,
        top_k: int,
        distance_threshold: float,
    ) -> list[SearchResult]:
        """
        ANN search with a distance filter — only return results whose
        cosine distance is below distance_threshold.
        Used in the reconciliation pass to find near-duplicate injected articles.
        """
