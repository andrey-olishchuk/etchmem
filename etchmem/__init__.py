"""
skillmem — a reconsolidating three-tier memory engine for LLM agents.

Public surface::

    from etchmem import Engine

    engine = Engine()
    engine.remember("text")
    results = engine.recall("query")
    summary = engine.consolidate()
"""
from etchmem.engine import Engine
from etchmem.config import Config
from etchmem.models import SearchResult, Record, InjectedArticle

__all__ = [
    "Engine",
    "Config",
    "SearchResult",
    "Record",
    "InjectedArticle",
]

__version__ = "0.1.0"
