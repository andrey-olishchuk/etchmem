"""
Tests for engine.export().

Covers:
  - Empty injected store produces export_dir, count=0, documents=[].
  - Export directory is created under .etchmem/export/<timestamp>/.
  - Each injected article is written as a separate JSON file.
  - JSON file content matches the InjectedArticle fields.
  - Multiple articles all appear in the export.
  - Return dict has the expected keys and shapes.
  - Repeated export calls create distinct timestamped directories.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from etchmem import Config, Engine
from etchmem.models import InjectedArticle


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path):
    """Engine backed by a fresh temporary Chroma directory, using offline
    hash embeddings (no model download required)."""
    cfg = Config(
        data_dir=str(tmp_path / "skillmem"),
        llm_provider="auto",
        openai_api_key=None,
        anthropic_api_key=None,
        embedding_function="hash",
    )
    return Engine(config=cfg)


def _inject_article(engine: Engine, article_id: str, content: str, **kwargs) -> InjectedArticle:
    """Helper: directly upsert an InjectedArticle into the injected store."""
    article = InjectedArticle(
        id=article_id,
        content=content,
        source_hashes=frozenset(kwargs.get("source_hashes", [article_id + "_src"])),
        skill=kwargs.get("skill", None),
        tags=kwargs.get("tags", {}),
        metadata=kwargs.get("metadata", {}),
        created_at=kwargs.get("created_at", time.time()),
    )
    engine._injected.upsert([article])
    return article


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_export_empty_store_returns_zero_count(engine):
    """export() on an empty injected store should return count=0."""
    result = engine.export()
    assert result["count"] == 0
    assert result["documents"] == []


def test_export_creates_directory(engine, tmp_path):
    """export() must create the .etchmem/export/<timestamp> directory."""
    result = engine.export()
    assert os.path.isdir(result["export_dir"])


def test_export_dir_is_under_etchmem(engine, tmp_path):
    """The export directory must live under .etchmem/export/ next to the data_dir."""
    result = engine.export()
    # data_dir is tmp_path/"skillmem"; its parent is tmp_path
    expected_root = os.path.join(str(tmp_path), ".etchmem", "export")
    assert result["export_dir"].startswith(expected_root)


def test_export_single_article_creates_one_file(engine):
    """One InjectedArticle → one .json file in the export directory."""
    _inject_article(engine, "abc123", "Python uses reference counting.")
    result = engine.export()

    assert result["count"] == 1
    export_dir = result["export_dir"]
    files = [f for f in os.listdir(export_dir) if f.endswith(".json")]
    assert len(files) == 1


def test_export_file_named_after_article_id(engine):
    """Each exported file must be named <article_id>.json."""
    art_id = "myarticleid"
    _inject_article(engine, art_id, "Some synthesized content.")
    result = engine.export()

    export_dir = result["export_dir"]
    assert os.path.isfile(os.path.join(export_dir, f"{art_id}.json"))


def test_export_json_content_matches_article(engine):
    """The JSON file content must faithfully reproduce all article fields."""
    art = _inject_article(
        engine,
        "detailed_id",
        "ChromaDB stores embeddings on disk.",
        source_hashes=["hash_a", "hash_b"],
        skill="code-reviewer",
        tags={"topic": "databases", "access_level": "public"},
        metadata={"source": "https://docs.trychroma.com"},
    )
    result = engine.export()

    file_path = os.path.join(result["export_dir"], f"{art.id}.json")
    with open(file_path, encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["id"] == art.id
    assert data["content"] == art.content
    assert set(data["source_hashes"]) == art.source_hashes
    assert data["skill"] == art.skill
    assert data["tags"] == art.tags
    assert data["metadata"] == art.metadata
    assert isinstance(data["created_at"], float)


def test_export_multiple_articles(engine):
    """All InjectedArticles in the store must be exported."""
    ids = ["art_one", "art_two", "art_three"]
    for aid in ids:
        _inject_article(engine, aid, f"Content for {aid}.")

    result = engine.export()

    assert result["count"] == len(ids)
    assert len(result["documents"]) == len(ids)
    exported_ids = {doc["id"] for doc in result["documents"]}
    assert exported_ids == set(ids)


def test_export_return_dict_has_required_keys(engine):
    """export() must return a dict with export_dir, count, and documents."""
    result = engine.export()
    assert "export_dir" in result
    assert "count" in result
    assert "documents" in result
    assert isinstance(result["export_dir"], str)
    assert isinstance(result["count"], int)
    assert isinstance(result["documents"], list)


def test_export_document_dict_has_required_fields(engine):
    """Each document dict in the return value must have all expected fields."""
    _inject_article(engine, "field_check", "Testing field presence.")
    result = engine.export()

    doc = result["documents"][0]
    for field in ("id", "content", "source_hashes", "skill", "tags", "metadata", "created_at"):
        assert field in doc, f"Missing field: {field}"


def test_repeated_exports_use_distinct_directories(engine):
    """Two export() calls must produce different timestamped directories."""
    _inject_article(engine, "repeat_art", "Some knowledge.")
    result_a = engine.export()
    time.sleep(1.1)  # ensure the UTC second ticks over
    result_b = engine.export()

    assert result_a["export_dir"] != result_b["export_dir"]
    assert os.path.isdir(result_a["export_dir"])
    assert os.path.isdir(result_b["export_dir"])


def test_export_source_hashes_is_sorted_list(engine):
    """source_hashes in the JSON output must be a sorted list (JSON-serialisable)."""
    _inject_article(
        engine, "hash_order", "Hashes sorted.",
        source_hashes=["zzz", "aaa", "mmm"],
    )
    result = engine.export()
    doc = result["documents"][0]
    assert doc["source_hashes"] == sorted(doc["source_hashes"])
    assert isinstance(doc["source_hashes"], list)
