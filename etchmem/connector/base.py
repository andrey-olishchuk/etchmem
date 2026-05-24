"""
LLMConnector abstract base class.

The connector's ONLY job is article synthesis — generative completion.
Embeddings are always Chroma-native and never go through this connector.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMConnector(ABC):
    """
    Single-method interface for generative synthesis.

    Used in exactly one place in the worker:
    - Reconsolidation-with-new-signal: rewrite an existing injected article
      blended with fresh context.

    All other worker operations (clustering, affectability arithmetic,
    the reconsolidation gate) are LLM-free.
    """

    @abstractmethod
    def complete(self, prompt: str, **opts) -> str:
        """
        Run a completion against the configured LLM.

        Args:
            prompt: The full synthesis prompt (already assembled by the worker).
            **opts: Optional provider-specific overrides (max_tokens, temperature…).

        Returns:
            The generated text string.
        """
