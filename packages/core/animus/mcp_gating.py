"""MCP Tool Gating — intent-based schema filtering at the MCP protocol boundary.

Provides ISO (Intent-Schema Overlap) scoring for MCP tools, session-scoped intent
tracking, and compact schema generation to reduce per-turn token overhead.

Usage:
    from animus.mcp_gating import MCPToolGater, set_mcp_intent, get_mcp_intent

    gater = MCPToolGater()
    gater.register_tool("animus_recall", "Search memory", ["memory", "search", "find"])
    set_mcp_intent("search for old memories")
    schemas = gater.get_gated_schemas(max_full=5)
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, field
from typing import Any

# Session-scoped intent tracking (per async context / MCP connection)
_mcp_session_intent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_session_intent", default=None
)

# Session-scoped cache for ISO scores (avoids recomputation on repeated list_tools)
_mcp_score_cache: contextvars.ContextVar[dict[str, list[tuple[float, str]]] | None] = (
    contextvars.ContextVar("mcp_score_cache", default=None)
)


def set_mcp_intent(intent: str) -> None:
    """Set the current MCP session intent."""
    _mcp_session_intent.set(intent)
    # Clear score cache when intent changes
    _mcp_score_cache.set(None)


def get_mcp_intent() -> str | None:
    """Get the current MCP session intent."""
    return _mcp_session_intent.get(None)


def _tokenize(text: str) -> set[str]:
    """Normalize and tokenize text for ISO scoring."""
    return set(re.sub(r"[^a-z0-9]", "", w.lower()) for w in text.split() if len(w) > 2)


@dataclass
class MCPToolMeta:
    """Metadata for an MCP tool used in gating decisions."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    category: str = "general"
    always_expose: bool = False

    def get_searchable_text(self) -> str:
        """Text used for ISO scoring."""
        parts = [self.name, self.description]
        parts.extend(self.keywords)
        parts.append(self.category)
        return " ".join(parts)


@dataclass
class GatedSchema:
    """Result of gating: either full or compact schema."""

    name: str
    description: str
    input_schema: dict[str, Any]
    is_compact: bool = False
    iso_score: float = 0.0


def _make_compact_schema(original: dict[str, Any]) -> dict[str, Any]:
    """Create a compact schema: type only, no properties.

    Preserves the top-level type and adds a note that full schema
    is available on demand.
    """
    return {
        "type": "object",
        "description": "Compact schema — full parameters available on request.",
    }


class MCPToolGater:
    """Intent-aware tool gater for MCP boundary.

    Scores tools by ISO (Intent-Schema Overlap) and returns
    full schemas for top-ranked tools, compact for the rest.
    """

    def __init__(self, max_full_schemas: int = 5):
        self._tools: dict[str, MCPToolMeta] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self.max_full_schemas = max_full_schemas

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        keywords: list[str] | None = None,
        category: str = "general",
        always_expose: bool = False,
    ) -> None:
        """Register an MCP tool for gating."""
        self._tools[name] = MCPToolMeta(
            name=name,
            description=description,
            keywords=keywords or [],
            category=category,
            always_expose=always_expose,
        )
        self._schemas[name] = dict(input_schema)  # shallow copy

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._schemas[name]
            return True
        return False

    def _iso_score(self, intent: str, tool_meta: MCPToolMeta) -> float:
        """Compute Intent-Schema Overlap score."""
        intent_tokens = _tokenize(intent)
        tool_tokens = _tokenize(tool_meta.get_searchable_text())
        if not intent_tokens or not tool_tokens:
            return 0.5  # Neutral
        overlap = len(intent_tokens & tool_tokens)
        union = len(intent_tokens | tool_tokens)
        return overlap / union if union > 0 else 0.0

    def _score_all(self, intent: str) -> list[tuple[float, str]]:
        """Score all tools and return (score, name) sorted descending."""
        scored: list[tuple[float, str]] = []
        for name, meta in self._tools.items():
            if meta.always_expose:
                scored.append((1.0, name))
            else:
                score = self._iso_score(intent, meta)
                scored.append((score, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def get_gated_schemas(
        self,
        intent: str | None = None,
        max_full: int | None = None,
    ) -> list[GatedSchema]:
        """Get gated schemas: full for top-N, compact for rest.

        Args:
            intent: User intent for ISO scoring. Uses session intent if None.
            max_full: Override for max full schemas. Uses instance default if None.

        Returns:
            List of GatedSchema objects.
        """
        effective_intent = intent or get_mcp_intent()
        max_full = max_full if max_full is not None else self.max_full_schemas

        if not effective_intent:
            # No intent — return all with full schemas (backward compatible)
            return [
                GatedSchema(
                    name=name,
                    description=meta.description,
                    input_schema=self._schemas[name],
                    is_compact=False,
                    iso_score=0.5,
                )
                for name, meta in self._tools.items()
            ]

        # Check cache
        cache = _mcp_score_cache.get(None)
        if cache is not None and effective_intent in cache:
            scored = cache[effective_intent]
        else:
            scored = self._score_all(effective_intent)
            if cache is None:
                cache = {}
            cache[effective_intent] = scored
            _mcp_score_cache.set(cache)

        results: list[GatedSchema] = []
        for i, (score, name) in enumerate(scored):
            meta = self._tools[name]
            is_compact = i >= max_full and not meta.always_expose
            schema = (
                _make_compact_schema(self._schemas[name]) if is_compact else self._schemas[name]
            )
            results.append(
                GatedSchema(
                    name=name,
                    description=meta.description,
                    input_schema=schema,
                    is_compact=is_compact,
                    iso_score=round(score, 3),
                )
            )
        return results

    def get_all_full_schemas(self) -> list[GatedSchema]:
        """Return all tools with full schemas (compatibility mode)."""
        return [
            GatedSchema(
                name=name,
                description=meta.description,
                input_schema=self._schemas[name],
                is_compact=False,
                iso_score=1.0,
            )
            for name, meta in self._tools.items()
        ]
