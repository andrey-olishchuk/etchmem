"""
Remote LLM connector — default synthesis backend.

Supports OpenAI and Anthropic. Auto-detects available keys.
Provider resolution order when llm_provider == "auto":
  1. Use OpenAI if OPENAI_API_KEY is set.
  2. Fall back to Anthropic if ANTHROPIC_API_KEY is set.
  3. Raise RuntimeError if neither key is configured.

Models (from README):
  OpenAI:    gpt-4o-mini  (default; gpt-5.4-nano when available)
  Anthropic: claude-haiku-4-5-20251001
"""
from __future__ import annotations

import json
from typing import Any

from etchmem.connector.base import LLMConnector


class RemoteConnector(LLMConnector):
    """
    API-backed connector for OpenAI and Anthropic.

    The connector is intentionally thin — it wraps one method (complete)
    and delegates everything else to the provider SDK.
    """

    def __init__(
        self,
        provider: str = "auto",           # "openai" | "anthropic" | "auto"
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        anthropic_api_key: str | None = None,
        anthropic_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._openai_key = openai_api_key
        self._openai_model = openai_model
        self._anthropic_key = anthropic_api_key
        self._anthropic_model = anthropic_model
        self._provider = self._resolve_provider(provider)

    def _resolve_provider(self, requested: str) -> str:
        if requested == "auto":
            if self._openai_key:
                return "openai"
            if self._anthropic_key:
                return "anthropic"
            raise RuntimeError(
                "skillmem: no LLM API key configured. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY (or pass them to Config)."
            )
        if requested == "openai" and not self._openai_key:
            raise RuntimeError(
                "skillmem: llm_provider='openai' but OPENAI_API_KEY is not set."
            )
        if requested == "anthropic" and not self._anthropic_key:
            raise RuntimeError(
                "skillmem: llm_provider='anthropic' but ANTHROPIC_API_KEY is not set."
            )
        return requested

    def complete(self, prompt: str, **opts: Any) -> str:
        """Run synthesis completion against the configured provider."""
        if self._provider == "openai":
            return self._complete_openai(prompt, **opts)
        return self._complete_anthropic(prompt, **opts)

    def _complete_openai(self, prompt: str, **opts: Any) -> str:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            raise ImportError("skillmem: 'openai' package is not installed.") from e

        client = OpenAI(api_key=self._openai_key)
        max_tokens = opts.pop("max_tokens", 2048)
        temperature = opts.pop("temperature", 0.3)

        response = client.chat.completions.create(
            model=self._openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **opts,
        )
        return response.choices[0].message.content or ""

    def _complete_anthropic(self, prompt: str, **opts: Any) -> str:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as e:
            raise ImportError("skillmem: 'anthropic' package is not installed.") from e

        client = anthropic.Anthropic(api_key=self._anthropic_key)
        max_tokens = opts.pop("max_tokens", 2048)

        message = client.messages.create(
            model=self._anthropic_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **opts,
        )
        # Extract text from the first text block
        for block in message.content:
            if hasattr(block, "text"):
                return block.text
        return ""


# ── Convenience factory ───────────────────────────────────────────────────────

def build_connector(config: "Any") -> RemoteConnector:
    """Build a RemoteConnector from a Config object."""
    return RemoteConnector(
        provider=config.llm_provider,
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_model,
        anthropic_api_key=config.anthropic_api_key,
        anthropic_model=config.anthropic_model,
    )
