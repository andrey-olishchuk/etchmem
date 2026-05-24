"""
Content and source-set hashing for dedup and crash-safe identity.

Two hash functions:
  content_hash(text)          — stable id for a single piece of text.
                                Used as Record.id (inserted into relational)
                                and as Deposit.id (copied into buffer).
  source_set_hash(hash_set)   — stable id for a synthesized article keyed
                                on the FROZENSET of its source-doc hashes.
                                Non-deterministic LLM output does NOT affect
                                identity; the source set does.

Both use SHA-256 (hex digest) for stability and collision resistance.
No external dependencies.
"""
from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """
    Compute a deterministic SHA-256 hex digest of `text`.

    Used as the canonical id for raw records and buffer deposits.
    Two records with identical text will have the same hash — this is
    the desired behaviour for the idempotency guard at intake.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_set_hash(source_hashes: frozenset[str] | set[str]) -> str:
    """
    Compute a deterministic SHA-256 hex digest of a SET of content hashes.

    The set is sorted before hashing so that {A, B} and {B, A} produce
    the same article identity.  This is the InjectedArticle.id.

    A re-run of consolidate() over the same source documents will
    produce the same id and therefore skip re-synthesis (idempotency).
    """
    sorted_hashes = sorted(source_hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def short_id(text: str, length: int = 16) -> str:
    """Return the first `length` characters of content_hash(text)."""
    return content_hash(text)[:length]
