"""
skillmem data models.

All structs are plain dataclasses — no ORM, no Pydantic dependency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ── Raw deposit written by remember() ────────────────────────────────────────

@dataclass
class Record:
    """
    A raw signal record deposited via remember().
    Stored in the *relational* collection.
    Immutable after creation — relational is append-only.
    """
    id: str                          # content hash — computed before insert
    content: str
    skill: str | None = None         # optional skill scope
    hint: float | None = None        # agent-supplied importance prior 0–1
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0          # unix timestamp; set by store on insert


# ── Buffer documents ──────────────────────────────────────────────────────────

@dataclass
class Deposit:
    """
    A relational record copied into the *buffer* collection at intake.
    Carries the original content hash for idempotency.
    """
    id: str                          # same content hash as the source Record
    content: str
    skill: str | None = None
    hint: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    doc_type: str = "deposit"        # discriminator stored as metadata


@dataclass
class RecallEvent:
    """
    Emitted by recall() into the *buffer* collection.
    Carries the query (pointer into injected) and a snapshot of relational
    hits (perishable, so snapshotted here).
    """
    id: str                          # uuid / hash assigned at emit time
    query: str                       # the original recall query
    relational_snapshot: str         # concatenated text of relational hits
    skill: str | None = None
    hint: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    doc_type: str = "recall_event"   # discriminator stored as metadata


# ── Clustering output ─────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """
    A group of buffer documents (Deposits + RecallEvents) that cluster
    around a shared topic. Output of Stage B clustering.
    """
    cluster_id: str
    documents: list[Deposit | RecallEvent]
    centroid_text: str | None = None   # representative text for the cluster

    @property
    def has_recall_events(self) -> bool:
        return any(isinstance(d, RecallEvent) for d in self.documents)

    @property
    def deposits(self) -> list[Deposit]:
        return [d for d in self.documents if isinstance(d, Deposit)]

    @property
    def recall_events(self) -> list[RecallEvent]:
        return [d for d in self.documents if isinstance(d, RecallEvent)]

    @property
    def queries(self) -> list[str]:
        return [re.query for re in self.recall_events]

    @property
    def skill(self) -> str | None:
        """Dominant skill across all docs in the cluster (most common)."""
        skills = [d.skill for d in self.documents if d.skill]
        if not skills:
            return None
        return max(set(skills), key=skills.count)


# ── Search / recall results ───────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    A single result returned by recall().
    Carries the injected article content blended with freshness signal.
    """
    id: str
    content: str
    score: float                     # asymmetric blended score
    source: str                      # "injected" | "relational" | "blended"
    skill: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Injected article ──────────────────────────────────────────────────────────

@dataclass
class InjectedArticle:
    """
    A synthesized knowledge article stored in the *injected* collection.
    Identity is keyed on the frozenset of source-doc hashes (content-addressed),
    NOT on synthesized text (LLM is non-deterministic).
    """
    id: str                          # hash of sorted source-doc-hash set
    content: str
    source_hashes: frozenset[str]    # set of Deposit.id values used in synthesis
    skill: str | None = None
    tags: dict[str, str] = field(default_factory=dict)   # derived by synthesis prompt
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ── Affectability ─────────────────────────────────────────────────────────────

@dataclass
class Affectability:
    """
    Per-record affectability score.
    Seeded from the agent hint (or default 0.5), not yet accumulated.
    Accumulation logic lives in util/affectability.py.
    """
    record_id: str
    score: float                     # 0.0–1.0
    hint_seeded: bool = False        # True if score originated from an agent hint
