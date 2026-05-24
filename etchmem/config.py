"""
skillmem configuration.

All settings can be overridden by passing a Config instance to Engine().
Environment variables are read at import time as defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Storage ───────────────────────────────────────────────────────────
    data_dir: str = field(
        default_factory=lambda: os.environ.get("SKILLMEM_DATA_DIR", ".skillmem")
    )

    # ── Collections ───────────────────────────────────────────────────────
    collection_relational: str = "relational"
    collection_buffer: str = "buffer"
    collection_injected: str = "injected"

    # ── TTL (seconds) for relational records ─────────────────────────────
    # Default: 7 days
    relational_ttl_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("SKILLMEM_RELATIONAL_TTL", str(7 * 24 * 3600))
        )
    )

    # ── Recall ────────────────────────────────────────────────────────────
    recall_top_k_injected: int = 5
    recall_top_k_relational: int = 5

    # Asymmetric scoring weights
    recall_injected_weight: float = 1.0   # high-trust base
    recall_relational_weight: float = 0.6  # freshness-boosted, confidence-discounted

    # ── Consolidation thresholds ──────────────────────────────────────────
    # branch_threshold: cosine distance above which a cluster splits into
    # multiple injected articles (distinct experiences → distinct articles).
    # Higher = harder to branch.
    branch_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("SKILLMEM_BRANCH_THRESHOLD", "0.4")
        )
    )

    # dedup_delete_threshold: cosine distance below which two injected
    # articles are "the same" → older is superseded.
    # MUST be strictly tighter (lower) than branch_threshold.
    dedup_delete_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("SKILLMEM_DEDUP_THRESHOLD", "0.15")
        )
    )

    # ── Affectability buckets ─────────────────────────────────────────────
    affectability_drop_below: float = 0.3   # < this → drop (noise)
    affectability_keep_below: float = 0.7   # 0.3–0.7 → keep in buffer
    affectability_default: float = 0.5      # default when no hint given

    # ── LLM connector ─────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("SKILLMEM_LLM_PROVIDER", "auto")
    )  # "openai" | "anthropic" | "auto"

    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("SKILLMEM_OPENAI_MODEL", "gpt-4o-mini")
    )

    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get(
            "SKILLMEM_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
        )
    )

    # ── Consolidation ─────────────────────────────────────────────────────
    consolidate_default_num_records: int | str = "all"
    consolidate_default_method: str = "LIFO"

    # ── Embedding function ─────────────────────────────────────────────────
    # None (default) → Chroma's bundled ONNX model (downloads on first use).
    # "hash"         → offline deterministic hash embedding (tests/CI only).
    # Any chromadb EmbeddingFunction subclass → custom.
    embedding_function: object = None

    # Minimum cluster size before promoting (orphaned records use
    # affectability; clustered records bypass this gate).
    min_cluster_size: int = 2

    def __post_init__(self) -> None:
        if self.dedup_delete_threshold >= self.branch_threshold:
            raise ValueError(
                f"dedup_delete_threshold ({self.dedup_delete_threshold}) must be "
                f"strictly less than branch_threshold ({self.branch_threshold})"
            )


# Module-level default — override by passing Config() to Engine.
DEFAULT_CONFIG = Config()
