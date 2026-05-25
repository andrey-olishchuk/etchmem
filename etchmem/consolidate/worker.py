"""
Consolidation Worker — the engine of etchmem.

Called by Engine.consolidate(). Runs in-process, single-pass, crash-safe.

Pipeline:
  Stage A — Intake:        pull relational deposits into buffer (hash-dedup).
  Stage B — Cluster:       group buffer docs (deposits + recall-events) by topic.
  Stage C — Evaluate:      route each cluster: drop | keep | form | reconsolidate.
  Reconciliation pass:     over all candidate new articles + their injected
                           neighbours, decide the final set, write, confirm,
                           hard-delete superseded articles.
  Flush:                   remove promoted/dropped/recall-event buffer docs.
                           Mid-band deposits (kept) stay.

Crash-safety anchor: flush is the LAST operation. A crash before flush
means the next run sees the full buffer and re-runs cleanly. Source-set
hashing prevents double-writes; deleting already-deleted ids is a no-op.

All LLM inference happens only at formation-promotion and
reconsolidation-with-new-signal.  Everything else is free arithmetic.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from etchmem.config import Config
from etchmem.connector.base import LLMConnector
from etchmem.models import (
    Cluster,
    Deposit,
    InjectedArticle,
    RecallEvent,
    SearchResult,
)
from etchmem.stores.chroma_store import ChromaStore
from etchmem.util import affectability as aff_util
from etchmem.util.hashing import content_hash, source_set_hash


# ── Synthesis prompts ──────────────────────────────────────────────────────────

_FORMATION_PROMPT = """\
You are a knowledge distillation engine. Your task is to synthesize a coherent,
accurate knowledge article from a set of raw signal fragments.

## Raw signal fragments
{fragments}

## Instructions
- Summarize all stored information properly: compress each point to its main
  idea while keeping the same words and phrasing from the fragments wherever
  possible.
- Write a single, cohesive knowledge article that captures all key facts.
- Do NOT invent facts not present in the fragments.
- Opinions, preferences, and subjective claims must NOT be stated as facts.
  Attribute them explicitly, e.g. "There is an opinion that grapefruits are
  the best" — never "Grapefruits are the best."
- Each fragment may be prefixed with a timestamp like `[21 Jun 2025, 15:25 UTC]`.
  Preserve these timestamps verbatim inline, anchoring the facts they precede,
  e.g. "As of 21 Jun 2025, 15:25 UTC, the deployment was rolled back."
- At the end of the article, output a JSON block on its own line:
  {{"tags": {{"access_level": "...", "topic": "...", "product": "..."}}}}

Output the article first, then the JSON tags block.
"""

_RECONSOLIDATION_PROMPT = """\
You are a knowledge reconsolidation engine. An existing knowledge article
is being updated with fresh context. Rewrite the article so it remains
accurate and incorporates the new information.

## Existing article
{existing}

## Fresh context (new deposits and recall context)
{fresh}

## Instructions
- Write ONLY what appears in the existing article and fresh context. Do NOT
  add new facts, inferences, or knowledge from your own training.
- Summarize properly: compress to the main idea of each point while keeping
  the same words and phrasing from the source material wherever possible.
- Preserve ALL load-bearing facts from the existing article.
- Incorporate relevant new information from the fresh context.
- Remove outdated claims only when the fresh context explicitly supersedes them.
- Opinions, preferences, and subjective claims must NOT be stated as facts.
  Attribute them explicitly, e.g. "There is an opinion that grapefruits are
  the best" — never "Grapefruits are the best."
- Fresh context fragments may be prefixed with a timestamp like `[21 Jun 2025, 15:25 UTC]`.
  Preserve these timestamps verbatim inline, anchoring the facts they precede.
  Also carry forward any timestamps already present in the existing article.
- At the end, output a JSON block on its own line:
  {{"tags": {{"access_level": "...", "topic": "...", "product": "..."}}}}

Output the rewritten article first, then the JSON tags block.
"""

_FACT_RETENTION_CHECK_PROMPT = """\
Does the NEW article preserve the key facts of the OLD article?

## OLD article
{old}

## NEW article
{new}

Reply with exactly one word: YES or NO.
"""


def _stamp(ts: float) -> str:
    """Format a Unix timestamp as a human-readable UTC string, e.g. '21 Jun 2025, 15:25 UTC'."""
    if not ts:
        return "unknown time"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%-d %b %Y, %H:%M UTC")


def _parse_synthesis_output(raw: str) -> tuple[str, dict]:
    """Split LLM output into (article_text, tags_dict)."""
    lines = raw.strip().split("\n")
    tags: dict = {}
    article_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{") and "tags" in stripped:
            try:
                parsed = json.loads(stripped)
                tags = parsed.get("tags", {})
                continue
            except json.JSONDecodeError:
                pass
        article_lines.append(line)
    article = "\n".join(article_lines).strip()
    return article, tags


# ── Run summary ───────────────────────────────────────────────────────────────

@dataclass
class RunSummary:
    intake_new: int = 0        # relational records newly copied to buffer
    intake_skipped: int = 0    # relational records skipped (already in buffer)
    clusters_total: int = 0
    formed: int = 0            # new injected articles created
    reconsolidated: int = 0    # existing articles rewritten
    dropped: int = 0           # buffer records dropped (noise)
    kept: int = 0              # buffer records kept for next run
    superseded: int = 0        # old injected articles hard-deleted
    flushed: int = 0           # buffer records flushed at end

    def to_dict(self) -> dict[str, int]:
        return {
            "intake_new": self.intake_new,
            "intake_skipped": self.intake_skipped,
            "clusters_total": self.clusters_total,
            "formed": self.formed,
            "reconsolidated": self.reconsolidated,
            "dropped": self.dropped,
            "kept": self.kept,
            "superseded": self.superseded,
            "flushed": self.flushed,
        }


# ── Worker ────────────────────────────────────────────────────────────────────

class Worker:
    """
    Consolidation worker. Instantiated fresh per Engine.consolidate() call.
    """

    def __init__(
        self,
        relational_store: ChromaStore,
        buffer_store: ChromaStore,
        injected_store: ChromaStore,
        connector: LLMConnector,
        config: Config,
    ) -> None:
        self._relational = relational_store
        self._buffer = buffer_store
        self._injected = injected_store
        self._connector = connector
        self._cfg = config

    def run(self, num_records: int | str, method: str) -> dict[str, Any]:
        summary = RunSummary()

        # ── Stage A: Intake ───────────────────────────────────────────────
        self._stage_a(num_records, method, summary)

        # ── Stage B: Cluster ──────────────────────────────────────────────
        all_buffer_docs = self._get_all_buffer_docs()
        if not all_buffer_docs:
            return summary.to_dict()

        clusters = self._stage_b(all_buffer_docs)
        summary.clusters_total = len(clusters)

        # ── Stage C: Evaluate + route ─────────────────────────────────────
        # Collect candidate new articles keyed by article_id → InjectedArticle
        # and which old article ids they supersede.
        candidate_articles: dict[str, InjectedArticle] = {}
        # same-run siblings: article_ids that were deliberately branched
        # from the same cluster (excluded from mutual dedup-deletion)
        same_run_sibling_sets: list[frozenset[str]] = []

        # ids to flush after the reconciliation pass
        to_flush_ids: list[str] = []    # promoted + dropped + recall-events
        to_keep_ids: list[str] = []     # mid-band deposits — NOT flushed

        for cluster in clusters:
            action, result = self._stage_c(cluster)

            doc_ids = [d.id for d in cluster.documents]

            if action == "drop":
                summary.dropped += len(cluster.documents)
                to_flush_ids.extend(doc_ids)

            elif action == "keep":
                summary.kept += len(cluster.deposits)
                # Recall-events are always flushed even from kept clusters
                to_flush_ids.extend([d.id for d in cluster.recall_events])
                to_keep_ids.extend([d.id for d in cluster.deposits])

            elif action == "form":
                articles: list[InjectedArticle] = result  # type: ignore[assignment]
                if len(articles) > 1:
                    same_run_sibling_sets.append(frozenset(a.id for a in articles))
                for article in articles:
                    candidate_articles[article.id] = article
                summary.formed += len(articles)
                to_flush_ids.extend(doc_ids)

            elif action == "reconsolidate":
                articles = result  # type: ignore[assignment]
                if len(articles) > 1:
                    same_run_sibling_sets.append(frozenset(a.id for a in articles))
                for article in articles:
                    candidate_articles[article.id] = article
                summary.reconsolidated += len(articles)
                to_flush_ids.extend(doc_ids)

        # ── Reconciliation pass ───────────────────────────────────────────
        superseded_ids = self._reconcile(
            candidate_articles=candidate_articles,
            same_run_sibling_sets=same_run_sibling_sets,
        )
        summary.superseded = len(superseded_ids)

        # Write new articles first, then delete superseded
        articles_to_write = [
            a for aid, a in candidate_articles.items()
            if not self._injected.exists(aid)
        ]
        if articles_to_write:
            self._injected.upsert(articles_to_write)

        if superseded_ids:
            self._injected.delete(superseded_ids)

        # ── Flush ─────────────────────────────────────────────────────────
        # Flush only after all writes and deletes are confirmed.
        # Mid-band kept deposits are NOT flushed.
        flush_ids = [fid for fid in to_flush_ids if fid not in to_keep_ids]
        if flush_ids:
            self._buffer.delete(flush_ids)
            summary.flushed = len(flush_ids)

        # Sweep expired relational records
        self._relational.expire()

        return summary.to_dict()

    # ── Stage A — Intake ──────────────────────────────────────────────────

    def _stage_a(
        self,
        num_records: int | str,
        method: str,
        summary: RunSummary,
    ) -> None:
        """
        Pull relational deposits into buffer.
        Hash-dedup: skip if the content hash already exists in the buffer.
        Relational originals are NOT deleted; they expire via TTL.
        """
        for record in self._relational.iter_for_consolidation(num_records, method):
            if self._buffer.exists(record.id):
                summary.intake_skipped += 1
                continue

            deposit = Deposit(
                id=record.id,
                content=record.content,
                skill=record.skill,
                hint=record.hint,
                metadata=record.metadata,
                created_at=record.created_at,
            )
            self._buffer.upsert([deposit])
            summary.intake_new += 1

    # ── Stage B — Cluster ─────────────────────────────────────────────────

    def _stage_b(
        self,
        documents: list[Deposit | RecallEvent],
    ) -> list[Cluster]:
        """Cluster all buffer documents together."""
        return self._buffer.cluster(
            documents=documents,
            branch_threshold=self._cfg.branch_threshold,
        )

    # ── Stage C — Evaluate + route ────────────────────────────────────────

    def _stage_c(
        self,
        cluster: Cluster,
    ) -> tuple[str, list[InjectedArticle]]:
        """
        Evaluate a cluster and return (action, articles).
        action: "drop" | "keep" | "form" | "reconsolidate"
        articles: the synthesized InjectedArticle objects (may be empty for drop/keep)
        """
        deposits = cluster.deposits
        recall_events = cluster.recall_events

        # Determine effective affectability score
        # Non-singleton clusters (>1 deposit) bypass the gate → score = 1.0
        if len(deposits) >= self._cfg.min_cluster_size:
            effective_score = 1.0
        else:
            # Singleton or recall-event-only cluster — compute from hints
            affs = [aff_util.seed(d.id, d.hint) for d in cluster.documents]
            effective_score = aff_util.cluster_rise(affs)

        action = aff_util.route(
            score=effective_score,
            has_recall_events=cluster.has_recall_events,
            drop_below=self._cfg.affectability_drop_below,
            keep_below=self._cfg.affectability_keep_below,
        )

        if action in ("drop", "keep"):
            return action, []

        if action == "form":
            articles = self._form(cluster)
            return "form", articles

        # action == "reconsolidate"
        articles = self._reconsolidate(cluster)
        return "reconsolidate", articles

    # ── Formation ─────────────────────────────────────────────────────────

    def _form(self, cluster: Cluster) -> list[InjectedArticle]:
        """
        Synthesize one (or more, if branching) injected articles from a cluster.
        """
        # Check idempotency: has this source-set already been synthesized?
        source_hashes = frozenset(d.id for d in cluster.deposits)
        article_id = source_set_hash(source_hashes)

        if self._injected.exists(article_id):
            # Already formed in a previous run — skip synthesis, return as-is
            results = self._injected.get_all(filters={"doc_type": "injected"})
            for item in results:
                if isinstance(item, InjectedArticle) and item.id == article_id:
                    return [item]
            return []

        # Synthesize — prefix each fragment with its absolute UTC timestamp
        fragments = "\n\n---\n\n".join(
            f"[{_stamp(d.created_at)}]\n{d.content}" for d in cluster.deposits
        )
        prompt = _FORMATION_PROMPT.format(fragments=fragments)
        raw = self._connector.complete(prompt)
        article_text, tags = _parse_synthesis_output(raw)

        article = InjectedArticle(
            id=article_id,
            content=article_text,
            source_hashes=source_hashes,
            skill=cluster.skill,
            tags=tags,
            created_at=time.time(),
        )
        return [article]

    # ── Reconsolidation ───────────────────────────────────────────────────

    def _reconsolidate(self, cluster: Cluster) -> list[InjectedArticle]:
        """
        Rewrite existing injected knowledge blended with fresh context.

        Steps:
          1. Re-query injected with each recall-event's query → current articles.
          2. Cheap gate (no LLM): does the cluster add anything new?
          3. If yes, synthesize. Fact-retention check before deleting old.
          4. Branch if distinct experiences warrant multiple articles.
        """
        # Re-query injected with the recall queries (pointer-based, always current)
        current_articles: dict[str, InjectedArticle] = {}
        for query in cluster.queries:
            results = self._injected.query(
                query_text=query,
                top_k=self._cfg.recall_top_k_injected,
                filters={"doc_type": "injected"},
                source="injected",
            )
            for r in results:
                if r.id not in current_articles:
                    items = self._injected.get_all(filters={"doc_type": "injected"})
                    for item in items:
                        if isinstance(item, InjectedArticle) and item.id == r.id:
                            current_articles[r.id] = item
                            break

        # Fresh deposit content for the cluster — prefix with absolute UTC timestamps
        fresh_texts = [
            f"[{_stamp(d.created_at)}]\n{d.content}" for d in cluster.deposits
        ]
        snapshots = [re.relational_snapshot for re in cluster.recall_events]
        all_fresh = fresh_texts + snapshots
        fresh_combined = "\n\n---\n\n".join(t for t in all_fresh if t.strip())

        new_articles: list[InjectedArticle] = []

        if not current_articles:
            # No existing injected knowledge — treat as formation
            return self._form(cluster)

        for old_article in current_articles.values():
            # ── Cheap gate (no LLM) ───────────────────────────────────────
            # Does the cluster add anything the current article doesn't cover?
            # Heuristic: check if any fresh fragment token is not in old article.
            if not self._has_new_signal(old_article.content, fresh_combined):
                # Nothing new — leave injected untouched; recall-events will be flushed
                continue

            # ── Source-set identity ───────────────────────────────────────
            source_hashes = old_article.source_hashes | frozenset(
                d.id for d in cluster.deposits
            )
            article_id = source_set_hash(source_hashes)

            if self._injected.exists(article_id):
                # Already synthesized this exact combination — skip
                continue

            # ── Synthesize ────────────────────────────────────────────────
            prompt = _RECONSOLIDATION_PROMPT.format(
                existing=old_article.content,
                fresh=fresh_combined,
            )
            raw = self._connector.complete(prompt)
            article_text, tags = _parse_synthesis_output(raw)

            # ── Fact-retention check ──────────────────────────────────────
            if not self._fact_retention_ok(old_article.content, article_text):
                # New article doesn't preserve load-bearing facts — fallback
                # to a safe merge (prepend old, append new)
                article_text = (
                    old_article.content + "\n\n[Updated context:]\n" + article_text
                )

            new_article = InjectedArticle(
                id=article_id,
                content=article_text,
                source_hashes=source_hashes,
                skill=cluster.skill or old_article.skill,
                tags=tags or old_article.tags,
                created_at=time.time(),
                metadata={"supersedes": old_article.id},
            )
            new_articles.append(new_article)

        return new_articles

    # ── Reconciliation pass ────────────────────────────────────────────────

    def _reconcile(
        self,
        candidate_articles: dict[str, InjectedArticle],
        same_run_sibling_sets: list[frozenset[str]],
    ) -> list[str]:
        """
        Over (new articles ∪ their current injected neighbours), decide the
        final set using dedup_delete_threshold.

        Returns list of old injected article ids to hard-delete.

        Crash-safety: write new → confirm → then delete old.
        Same-run siblings are excluded from mutual dedup-deletion.
        """
        superseded_ids: list[str] = []
        seen_superseded: set[str] = set()

        # Build sibling lookup for fast exclusion
        sibling_of: dict[str, frozenset[str]] = {}
        for sib_set in same_run_sibling_sets:
            for aid in sib_set:
                sibling_of[aid] = sib_set

        for new_article in candidate_articles.values():
            # Find existing injected neighbours within dedup threshold
            neighbors = self._injected.get_neighbors(
                query_text=new_article.content,
                top_k=10,
                distance_threshold=self._cfg.dedup_delete_threshold,
            )
            for neighbor in neighbors:
                # Skip if already marked superseded this run
                if neighbor.id in seen_superseded:
                    continue
                # Skip if it's a sibling of the new article (deliberate branch)
                if neighbor.id in candidate_articles:
                    continue
                # Skip if it IS the new article
                if neighbor.id == new_article.id:
                    continue
                superseded_ids.append(neighbor.id)
                seen_superseded.add(neighbor.id)

            # Also check metadata-recorded supersession (from reconsolidation)
            old_id = new_article.metadata.get("supersedes")
            if old_id and old_id not in seen_superseded and old_id not in candidate_articles:
                superseded_ids.append(old_id)
                seen_superseded.add(old_id)

        return superseded_ids

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_all_buffer_docs(self) -> list[Deposit | RecallEvent]:
        """Return all current buffer documents (deposits + recall-events)."""
        all_items = self._buffer.get_all()
        return [
            item for item in all_items
            if isinstance(item, (Deposit, RecallEvent))
        ]

    def _has_new_signal(self, existing: str, fresh: str) -> bool:
        """
        Cheap (no-LLM) gate: does `fresh` contain information not in `existing`?

        Heuristic: tokenise both by whitespace, check whether fresh introduces
        any words not already in the existing article (after lowercasing).
        This is intentionally conservative — false positives (sending to LLM
        when unnecessary) are cheaper than false negatives (missing an update).
        """
        if not fresh.strip():
            return False
        existing_tokens = set(existing.lower().split())
        fresh_tokens = set(fresh.lower().split())
        # Stop-words we don't want to count as "new signal"
        _STOP = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "is", "are", "was", "were", "be",
            "been", "has", "have", "had", "it", "its", "this", "that",
            "these", "those", "as", "from", "not", "no", "so", "if",
        }
        new_tokens = (fresh_tokens - existing_tokens) - _STOP
        return len(new_tokens) > 0

    def _fact_retention_ok(self, old_article: str, new_article: str) -> bool:
        """
        Check that new_article preserves the load-bearing facts of old_article.

        Uses an LLM call (cheap, bounded prompt).
        Falls back to True on any error (conservative — prefer keeping data).
        """
        try:
            prompt = _FACT_RETENTION_CHECK_PROMPT.format(
                old=old_article[:2000],   # cap for token budget
                new=new_article[:2000],
            )
            response = self._connector.complete(prompt, max_tokens=10)
            return response.strip().upper().startswith("Y")
        except Exception:
            return True  # conservative fallback
