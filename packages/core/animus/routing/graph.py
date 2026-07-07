"""RoutingGraph: tracks provider performance history for history-aware routing.

Graph model:
- Nodes: ProviderNode (model/provider instances)
- Edges: RoutingEdge (provider → task → outcome history)
- TaskSignature: canonical task identifier derived from prompt
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from animus.logging import get_logger

logger = get_logger("routing.graph")


@dataclass
class ProviderNode:
    """A model/provider that can handle requests."""

    name: str  # e.g., "claude-sonnet-5", "qwen2.5-coder:14b"
    provider_type: str  # e.g., "anthropic", "ollama", "openai"
    capabilities: set[str] = field(default_factory=set)
    # Capability tags: "code", "reasoning", "long_context", "cheap", "fast"
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    def can_handle(self, task_type: str, estimated_tokens: int = 0) -> bool:
        """Check if this provider can handle a given task type."""
        if estimated_tokens > self.max_tokens:
            return False
        # Provider can handle if it has the capability or is general-purpose
        return task_type in self.capabilities or "general" in self.capabilities


@dataclass
class TaskSignature:
    """Canonical task identifier derived from a prompt.

    Uses keyword extraction + content hashing to create stable identifiers
    for similar prompts. This enables learning from prompt clusters.
    """

    task_type: str  # e.g., "summarization", "code_generation", "debugging"
    keywords: tuple[str, ...]  # Sorted keyword tuple for clustering
    hash: str  # Content hash for exact matching

    @classmethod
    def from_prompt(cls, prompt: str, task_type: str | None = None) -> "TaskSignature":
        """Extract signature from a user prompt."""
        # Normalize and extract keywords
        text = prompt.lower()
        # Remove punctuation, keep alphanumeric
        words = re.findall(r"[a-z0-9]+", text)
        # Filter to meaningful keywords (stopwords could be filtered here)
        keywords = sorted(set(w for w in words if len(w) > 2))
        keyword_tuple = tuple(keywords[:20])  # Limit to top 20

        # Derive task type from keywords if not provided
        if task_type is None:
            task_type = cls._infer_task_type(text, keywords)

        # Hash the full prompt for exact matching
        content_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        return cls(task_type=task_type, keywords=keyword_tuple, hash=content_hash)

    @staticmethod
    def _infer_task_type(text: str, keywords: list[str]) -> str:
        """Infer task type from prompt text."""
        text = text.lower()

        # Code-related tasks
        if any(kw in text for kw in ("code", "implement", "function", "class", "debug", "fix bug", "refactor")):
            return "code"
        # Reasoning/planning tasks
        if any(kw in text for kw in ("plan", "design", "architect", "strategy", "analyze")):
            return "reasoning"
        # Summarization tasks
        if any(kw in text for kw in ("summarize", "summary", "tl;dr", "brief", "condense")):
            return "summarization"
        # Search/research tasks
        if any(kw in text for kw in ("search", "find", "research", "look up", "what is")):
            return "research"
        # Formatting tasks
        if any(kw in text for kw in ("format", "reformat", "convert", "translate", "json", "xml")):
            return "formatting"
        # Writing tasks
        if any(kw in text for kw in ("write", "draft", "compose", "email", "message")):
            return "writing"

        return "general"

    def similarity(self, other: "TaskSignature") -> float:
        """Compute Jaccard similarity between keyword sets."""
        set1 = set(self.keywords)
        set2 = set(other.keywords)
        if not set1 or not set2:
            return 0.0
        overlap = len(set1 & set2)
        union = len(set1 | set2)
        return overlap / union if union > 0 else 0.0


@dataclass
class RoutingOutcome:
    """Single outcome of routing a task to a provider."""

    success: bool
    latency_ms: float
    tokens_used: int
    quality_score: float  # 0.0–1.0, from rubric eval or manual rating
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_type: str | None = None  # e.g., "timeout", "rate_limit", "bad_response"


@dataclass
class RoutingEdge:
    """Edge between provider and task, with outcome history."""

    provider_name: str
    task_signature: TaskSignature
    outcomes: list[RoutingOutcome] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.outcomes)

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            return 0.5  # Neutral prior
        successes = sum(1 for o in self.outcomes if o.success)
        return successes / len(self.outcomes)

    @property
    def avg_latency_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.latency_ms for o in self.outcomes) / len(self.outcomes)

    @property
    def avg_quality(self) -> float:
        if not self.outcomes:
            return 0.5  # Neutral prior
        return sum(o.quality_score for o in self.outcomes) / len(self.outcomes)

    @property
    def recent_success_rate(self, window: int = 5) -> float:
        """Success rate over last N outcomes."""
        recent = self.outcomes[-window:]
        if not recent:
            return 0.5
        successes = sum(1 for o in recent if o.success)
        return successes / len(recent)

    @property
    def error_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        errors = sum(1 for o in self.outcomes if o.error_type)
        return errors / len(self.outcomes)


class RoutingGraph:
    """Graph tracking provider-task performance history.

    Enables history-aware routing by maintaining outcome statistics
    for each provider-task combination.
    """

    def __init__(self):
        self.providers: dict[str, ProviderNode] = {}
        self.edges: dict[str, RoutingEdge] = {}  # key = "provider_name:task_hash"
        self._task_clusters: dict[str, list[str]] = defaultdict(list)  # task_type -> hashes

    def register_provider(self, provider: ProviderNode) -> None:
        """Register a new provider node."""
        self.providers[provider.name] = provider
        logger.debug(f"Registered provider: {provider.name}")

    def record_outcome(
        self,
        provider_name: str,
        task_signature: TaskSignature,
        outcome: RoutingOutcome,
    ) -> None:
        """Record an outcome for a provider-task edge."""
        edge_key = f"{provider_name}:{task_signature.hash}"
        if edge_key not in self.edges:
            self.edges[edge_key] = RoutingEdge(
                provider_name=provider_name,
                task_signature=task_signature,
            )
        self.edges[edge_key].outcomes.append(outcome)
        self._task_clusters[task_signature.task_type].append(task_signature.hash)

        # Prune old outcomes to prevent unbounded growth
        edge = self.edges[edge_key]
        if len(edge.outcomes) > 100:
            edge.outcomes = edge.outcomes[-100:]

    def get_edge(
        self, provider_name: str, task_signature: TaskSignature
    ) -> RoutingEdge | None:
        """Get edge for exact provider-task match."""
        edge_key = f"{provider_name}:{task_signature.hash}"
        return self.edges.get(edge_key)

    def get_similar_edges(
        self, provider_name: str, task_signature: TaskSignature, min_similarity: float = 0.3
    ) -> list[tuple[RoutingEdge, float]]:
        """Get edges with similar task signatures for a provider."""
        results = []
        for edge in self.edges.values():
            if edge.provider_name != provider_name:
                continue
            sim = task_signature.similarity(edge.task_signature)
            if sim >= min_similarity:
                results.append((edge, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_provider_stats(self, provider_name: str) -> dict[str, Any]:
        """Get aggregate statistics for a provider across all tasks."""
        provider_edges = [e for e in self.edges.values() if e.provider_name == provider_name]
        if not provider_edges:
            return {"total_attempts": 0, "overall_success_rate": 0.5}

        total_attempts = sum(e.total_attempts for e in provider_edges)
        total_successes = sum(
            sum(1 for o in e.outcomes if o.success) for e in provider_edges
        )
        avg_latency = sum(e.avg_latency_ms for e in provider_edges) / len(provider_edges)
        avg_quality = sum(e.avg_quality for e in provider_edges) / len(provider_edges)

        return {
            "total_attempts": total_attempts,
            "overall_success_rate": total_successes / total_attempts if total_attempts > 0 else 0.5,
            "avg_latency_ms": avg_latency,
            "avg_quality": avg_quality,
            "task_coverage": len(provider_edges),
        }

    def get_task_type_stats(self, task_type: str) -> dict[str, Any]:
        """Get statistics for a task type across all providers."""
        hashes = self._task_clusters.get(task_type, [])
        if not hashes:
            return {}

        provider_stats: dict[str, list[float]] = defaultdict(list)
        for edge in self.edges.values():
            if edge.task_signature.hash in hashes:
                provider_stats[edge.provider_name].append(edge.success_rate)

        return {
            name: {"avg_success_rate": sum(rates) / len(rates) if rates else 0.5}
            for name, rates in provider_stats.items()
        }

    def to_dict(self) -> dict:
        """Serialize graph to dict."""
        return {
            "providers": {
                name: {
                    "name": p.name,
                    "provider_type": p.provider_type,
                    "capabilities": list(p.capabilities),
                    "max_tokens": p.max_tokens,
                }
                for name, p in self.providers.items()
            },
            "edges": [
                {
                    "provider_name": e.provider_name,
                    "task_type": e.task_signature.task_type,
                    "keywords": list(e.task_signature.keywords),
                    "hash": e.task_signature.hash,
                    "outcomes": [
                        {
                            "success": o.success,
                            "latency_ms": o.latency_ms,
                            "tokens_used": o.tokens_used,
                            "quality_score": o.quality_score,
                            "timestamp": o.timestamp,
                            "error_type": o.error_type,
                        }
                        for o in e.outcomes
                    ],
                }
                for e in self.edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingGraph":
        """Deserialize graph from dict."""
        graph = cls()
        for p_data in data.get("providers", {}).values():
            graph.register_provider(
                ProviderNode(
                    name=p_data["name"],
                    provider_type=p_data["provider_type"],
                    capabilities=set(p_data.get("capabilities", [])),
                    max_tokens=p_data.get("max_tokens", 4096),
                )
            )
        for e_data in data.get("edges", []):
            sig = TaskSignature(
                task_type=e_data["task_type"],
                keywords=tuple(e_data.get("keywords", [])),
                hash=e_data["hash"],
            )
            for o_data in e_data.get("outcomes", []):
                graph.record_outcome(
                    provider_name=e_data["provider_name"],
                    task_signature=sig,
                    outcome=RoutingOutcome(
                        success=o_data["success"],
                        latency_ms=o_data["latency_ms"],
                        tokens_used=o_data["tokens_used"],
                        quality_score=o_data["quality_score"],
                        timestamp=o_data.get("timestamp", datetime.now().isoformat()),
                        error_type=o_data.get("error_type"),
                    ),
                )
        return graph