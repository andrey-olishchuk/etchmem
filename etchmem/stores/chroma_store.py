"""
ChromaStore — the bundled default storage implementation.

One PersistentClient, one data directory, three collections.
All three tiers share one client and one embedding function
(Chroma's bundled default embedding model — no external API required).

The same ChromaStore class is instantiated three times:
  - relational_store = ChromaStore(client, "relational") → used as RelationalStore
  - buffer_store     = ChromaStore(client, "buffer")     → used as InjectedStore
  - injected_store   = ChromaStore(client, "injected")   → used as InjectedStore

Stores receive TEXT, not pre-computed vectors. ChromaDB handles vectorization.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings

from etchmem.models import (
    Cluster,
    Deposit,
    InjectedArticle,
    Record,
    RecallEvent,
    SearchResult,
)
from etchmem.stores.base import InjectedStore, RelationalStore


def _build_client(data_dir: str) -> chromadb.PersistentClient:
    """Create or open the shared PersistentClient."""
    return chromadb.PersistentClient(
        path=data_dir,
        settings=Settings(anonymized_telemetry=False),
    )


# ── Optional lightweight embedding function for offline/test use ──────────────

def _hash_embed(texts) -> list[list[float]]:
    """Compute deterministic 64-dim hash embeddings for a list of texts."""
    import hashlib
    results = []
    for text in texts:
        digest = hashlib.sha256(str(text).encode()).digest()  # 32 bytes
        # Expand to 64 floats in [0, 1] by repeating the 32-byte digest
        vec = [b / 255.0 for b in digest] + [b / 255.0 for b in digest]
        results.append(vec)
    return results


class _HashEmbeddingFunction:
    """
    Deterministic hash-based embedding for offline environments (CI, tests,
    air-gapped deployments). NOT semantic — do not use in production.

    Produces a 64-dim float vector from SHA-256 bytes of each input text.
    Enabled by passing embedding_function="hash" to build_stores().

    Implements all methods that Chroma 1.5+ may call on an embedding function.
    Does NOT inherit from EmbeddingFunction (it is a Protocol, not a class).
    """

    # Called for document inserts
    def __call__(self, input) -> list[list[float]]:
        return _hash_embed(list(input))

    # Called for query vectors in Chroma 1.5+
    def embed_query(self, input) -> list[list[float]]:
        return _hash_embed(list(input))

    # Required by Chroma's embedding-function registry / config serialisation
    @classmethod
    def name(cls) -> str:
        return "skillmem-hash-offline"

    @classmethod
    def build_from_config(cls, config):
        return cls()

    def get_config(self) -> dict:
        return {}

    # Silences the "legacy embedding function" deprecation warning in Chroma 1.5
    def is_legacy(self) -> bool:
        return False

    # Silences the "supported_spaces" deprecation warning in Chroma 1.5
    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine", "l2", "ip"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_to_search_result(
    doc_id: str,
    document: str,
    distance: float,
    metadata: dict[str, Any],
    weight: float = 1.0,
    source: str = "unknown",
) -> SearchResult:
    skill = metadata.get("skill") or None
    return SearchResult(
        id=doc_id,
        content=document,
        score=weight * (1.0 - distance),   # convert distance → similarity score
        source=source,
        skill=skill,
        metadata={k: v for k, v in metadata.items() if k != "skill"},
    )


def _record_from_chroma(doc_id: str, document: str, metadata: dict) -> Record:
    return Record(
        id=doc_id,
        content=document,
        skill=metadata.get("skill") or None,
        hint=metadata.get("hint"),
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        created_at=float(metadata.get("created_at", 0)),
        expires_at=float(metadata.get("expires_at", 0)),
    )


def _deposit_from_chroma(doc_id: str, document: str, metadata: dict) -> Deposit:
    return Deposit(
        id=doc_id,
        content=document,
        skill=metadata.get("skill") or None,
        hint=metadata.get("hint"),
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        created_at=float(metadata.get("created_at", 0)),
        doc_type="deposit",
    )


def _recall_event_from_chroma(doc_id: str, document: str, metadata: dict) -> RecallEvent:
    return RecallEvent(
        id=doc_id,
        query=metadata.get("query", ""),
        relational_snapshot=metadata.get("relational_snapshot", ""),
        skill=metadata.get("skill") or None,
        hint=metadata.get("hint"),
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        created_at=float(metadata.get("created_at", 0)),
        doc_type="recall_event",
    )


def _injected_from_chroma(doc_id: str, document: str, metadata: dict) -> InjectedArticle:
    source_hashes_raw = metadata.get("source_hashes", "[]")
    try:
        source_hashes = frozenset(json.loads(source_hashes_raw))
    except (json.JSONDecodeError, TypeError):
        source_hashes = frozenset()

    tags_raw = metadata.get("tags", "{}")
    try:
        tags = json.loads(tags_raw)
    except (json.JSONDecodeError, TypeError):
        tags = {}

    return InjectedArticle(
        id=doc_id,
        content=document,
        source_hashes=source_hashes,
        skill=metadata.get("skill") or None,
        tags=tags,
        metadata=json.loads(metadata.get("extra_metadata", "{}")),
        created_at=float(metadata.get("created_at", 0)),
    )


# ── Core ChromaStore ──────────────────────────────────────────────────────────

class ChromaStore:
    """
    Implements both RelationalStore and InjectedStore using a single
    ChromaDB collection.

    Instantiate once per logical tier (relational / buffer / injected),
    all sharing the same `client`.
    """

    def __init__(
        self,
        client: chromadb.PersistentClient,
        collection_name: str,
        ttl_seconds: int = 7 * 24 * 3600,
        embedding_function=None,
    ) -> None:
        """
        Args:
            embedding_function: Optional Chroma-compatible embedding function.
                None (default) → Chroma's bundled ONNX model (requires
                network on first use to download weights).
                Pass embedding_function="hash" to use the offline
                _HashEmbeddingFunction (deterministic but NOT semantic —
                only suitable for tests / CI / air-gapped environments).
                Pass any chromadb EmbeddingFunction subclass for custom use.
        """
        self._client = client
        self._collection_name = collection_name
        self._ttl_seconds = ttl_seconds

        ef = None
        if embedding_function == "hash":
            ef = _HashEmbeddingFunction()
        elif embedding_function is not None:
            ef = embedding_function

        kwargs: dict = {"name": collection_name, "metadata": {"hnsw:space": "cosine"}}
        if ef is not None:
            kwargs["embedding_function"] = ef

        self._collection = client.get_or_create_collection(**kwargs)

    # ── RelationalStore interface ─────────────────────────────────────────

    def add(self, record: Record) -> str:
        """Insert a Record into this collection. Sets expires_at."""
        now = time.time()
        expires_at = now + self._ttl_seconds
        record.expires_at = expires_at
        record.created_at = record.created_at or now

        metadata: dict[str, Any] = {
            "skill": record.skill or "",
            "created_at": record.created_at,
            "expires_at": expires_at,
            "extra_metadata": json.dumps(record.metadata),
            "doc_type": "record",
        }
        if record.hint is not None:
            metadata["hint"] = record.hint

        self._collection.upsert(
            ids=[record.id],
            documents=[record.content],
            metadatas=[metadata],
        )
        return record.id

    def search(
        self,
        query_text: str,
        top_k: int,
        skill: str | None = None,
        filters: dict[str, Any] | None = None,
        weight: float = 1.0,
        source: str = "relational",
    ) -> list[SearchResult]:
        """ANN search; filters by skill metadata if provided."""
        now = time.time()
        where: dict[str, Any] = {"expires_at": {"$gt": now}}
        if skill:
            where = {"$and": [where, {"skill": {"$eq": skill}}]}
        if filters:
            for k, v in filters.items():
                where = {"$and": [where, {k: {"$eq": v}}]}

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        out: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            out.append(
                _result_to_search_result(doc_id, doc, dist, meta, weight=weight, source=source)
            )
        return out

    def expire(self) -> int:
        """Delete all records with expires_at < now."""
        now = time.time()
        try:
            # Fetch ids of expired records
            result = self._collection.get(
                where={"expires_at": {"$lt": now}},
                include=[],
            )
            ids = result.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def iter_for_consolidation(
        self,
        num_records: int | str,
        method: str,
    ) -> Iterable[Record]:
        """Yield records for the consolidation worker."""
        now = time.time()
        try:
            result = self._collection.get(
                where={"expires_at": {"$gt": now}},
                include=["documents", "metadatas"],
            )
        except Exception:
            return

        records = [
            _record_from_chroma(rid, doc, meta)
            for rid, doc, meta in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ]

        # Sort by created_at
        reverse = method.upper() == "LIFO"
        records.sort(key=lambda r: r.created_at, reverse=reverse)

        if num_records != "all":
            records = records[: int(num_records)]

        yield from records

    def count(self) -> int:
        return self._collection.count()

    def get_by_id(self, record_id: str) -> Record | None:
        try:
            result = self._collection.get(
                ids=[record_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None
            return _record_from_chroma(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
            )
        except Exception:
            return None

    # ── InjectedStore interface ───────────────────────────────────────────

    def upsert(self, items: list[InjectedArticle | Deposit | RecallEvent]) -> None:
        """Write or overwrite items into this collection."""
        ids, documents, metadatas = [], [], []
        for item in items:
            meta: dict[str, Any] = {
                "created_at": getattr(item, "created_at", time.time()),
                "skill": item.skill or "",
                "extra_metadata": json.dumps(getattr(item, "metadata", {})),
            }
            if isinstance(item, InjectedArticle):
                meta["doc_type"] = "injected"
                meta["source_hashes"] = json.dumps(sorted(item.source_hashes))
                meta["tags"] = json.dumps(item.tags)
                document = item.content
            elif isinstance(item, RecallEvent):
                meta["doc_type"] = "recall_event"
                meta["query"] = item.query
                meta["relational_snapshot"] = item.relational_snapshot
                if item.hint is not None:
                    meta["hint"] = item.hint
                # For embedding purposes, use the query + snapshot as the document
                document = f"{item.query}\n{item.relational_snapshot}"
            elif isinstance(item, Deposit):
                meta["doc_type"] = "deposit"
                if item.hint is not None:
                    meta["hint"] = item.hint
                document = item.content
            else:
                continue

            ids.append(item.id)
            documents.append(document)
            metadatas.append(meta)

        if ids:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def query(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        weight: float = 1.0,
        source: str = "injected",
    ) -> list[SearchResult]:
        """ANN search this collection."""
        count = self._collection.count()
        if count == 0:
            return []

        where = None
        if filters:
            clauses = [{"$eq": {k: v}} for k, v in filters.items()]  # type: ignore[dict-item]
            # Use simple single-key filter directly
            if len(filters) == 1:
                k, v = next(iter(filters.items()))
                where = {k: {"$eq": v}}
            else:
                where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        out: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            out.append(
                _result_to_search_result(doc_id, doc, dist, meta, weight=weight, source=source)
            )
        return out

    def delete(self, ids: list[str]) -> None:
        """Hard-delete by id. Deleting non-existent ids is a no-op."""
        if not ids:
            return
        # Chroma silently ignores missing ids
        try:
            self._collection.delete(ids=ids)
        except Exception:
            pass

    def exists(self, content_hash: str) -> bool:
        """Return True if a document with this id exists."""
        try:
            result = self._collection.get(ids=[content_hash], include=[])
            return bool(result.get("ids"))
        except Exception:
            return False

    def cluster(
        self,
        documents: list[Deposit | RecallEvent],
        branch_threshold: float,
    ) -> list[Cluster]:
        """
        Group documents into Clusters via embedding-based similarity.

        Implementation: single-link agglomerative clustering using Chroma
        embeddings obtained via query. Each un-clustered document queries
        all others and merges those within the branch_threshold.

        This is a best-effort, in-process clustering; for v0 correctness
        is prioritised over performance.
        """
        if not documents:
            return []

        # Obtain embeddings by doing pairwise queries
        # For simplicity in v0, use a greedy single-pass nearest-neighbour
        # approach: sort by created_at, then greedily assign to clusters.
        # A document joins an existing cluster if it is within branch_threshold
        # distance from ANY member of that cluster (single-link).

        # We'll use Chroma's query to get distances between documents.
        # First, upsert all documents into a temporary scratch space if they're
        # not already in this collection, then query.

        clusters: list[list[Deposit | RecallEvent]] = []
        assigned: set[str] = set()

        # Build a lookup
        doc_map = {d.id: d for d in documents}

        for doc in documents:
            if doc.id in assigned:
                continue

            # Query this collection for the closest documents
            doc_text = doc.content if isinstance(doc, Deposit) else f"{doc.query}\n{doc.relational_snapshot}"
            count = self._collection.count()
            if count == 0:
                clusters.append([doc])
                assigned.add(doc.id)
                continue

            try:
                results = self._collection.query(
                    query_texts=[doc_text],
                    n_results=min(len(documents), count),
                    include=["metadatas", "distances"],
                )
            except Exception:
                clusters.append([doc])
                assigned.add(doc.id)
                continue

            similar_ids: list[str] = []
            if results["ids"] and results["ids"][0]:
                for rid, dist in zip(results["ids"][0], results["distances"][0]):
                    if rid in doc_map and rid not in assigned and dist <= branch_threshold:
                        similar_ids.append(rid)

            cluster_members: list[Deposit | RecallEvent] = [doc]
            assigned.add(doc.id)
            for sid in similar_ids:
                if sid not in assigned:
                    cluster_members.append(doc_map[sid])
                    assigned.add(sid)

            clusters.append(cluster_members)

        # Convert to Cluster objects
        result_clusters: list[Cluster] = []
        for i, members in enumerate(clusters):
            centroid_text = members[0].content if isinstance(members[0], Deposit) else members[0].query
            result_clusters.append(
                Cluster(
                    cluster_id=str(uuid.uuid4()),
                    documents=members,
                    centroid_text=centroid_text,
                )
            )
        return result_clusters

    def get_all(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Deposit | RecallEvent | InjectedArticle]:
        """Return all documents, optionally filtered by metadata."""
        try:
            where = None
            if filters:
                if len(filters) == 1:
                    k, v = next(iter(filters.items()))
                    where = {k: {"$eq": v}}
                else:
                    where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

            result = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        out: list[Deposit | RecallEvent | InjectedArticle] = []
        for rid, doc, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            doc_type = meta.get("doc_type", "deposit")
            if doc_type == "recall_event":
                out.append(_recall_event_from_chroma(rid, doc, meta))
            elif doc_type == "injected":
                out.append(_injected_from_chroma(rid, doc, meta))
            else:
                out.append(_deposit_from_chroma(rid, doc, meta))
        return out

    def get_neighbors(
        self,
        query_text: str,
        top_k: int,
        distance_threshold: float,
    ) -> list[SearchResult]:
        """ANN search with a distance cut-off."""
        count = self._collection.count()
        if count == 0:
            return []
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        out: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            if dist <= distance_threshold:
                out.append(
                    _result_to_search_result(doc_id, doc, dist, meta, source="injected")
                )
        return out


# ── Factory ───────────────────────────────────────────────────────────────────

def build_stores(
    data_dir: str,
    collection_relational: str,
    collection_buffer: str,
    collection_injected: str,
    ttl_seconds: int,
    embedding_function=None,
) -> tuple[ChromaStore, ChromaStore, ChromaStore]:
    """
    Build the shared PersistentClient and return three ChromaStore instances:
    (relational_store, buffer_store, injected_store).

    All three collections share the same embedding_function.
    Pass embedding_function="hash" for offline/test use.
    """
    client = _build_client(data_dir)
    relational = ChromaStore(client, collection_relational, ttl_seconds=ttl_seconds, embedding_function=embedding_function)
    buffer = ChromaStore(client, collection_buffer, ttl_seconds=ttl_seconds, embedding_function=embedding_function)
    injected = ChromaStore(client, collection_injected, ttl_seconds=ttl_seconds, embedding_function=embedding_function)
    return relational, buffer, injected
