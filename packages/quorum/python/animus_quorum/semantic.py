"""LLM-powered semantic matching for intent overlap detection.

Provides semantic understanding beyond structural matching — catches
overlaps like "AccountManager" ↔ "UserHandler" that name normalization
misses. Optional enhancement layer: when no SemanticMatcher is provided,
the resolver falls back to pure structural matching (Phase 1 behavior).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticMatch:
    """Result of checking whether two specs semantically overlap."""

    overlap: bool
    confidence: float
    reasoning: str
    source: str = "llm"


@dataclass
class ConstraintApplicability:
    """Result of checking whether a constraint applies to an intent."""

    applies: bool
    confidence: float
    reasoning: str


@dataclass
class TrajectoryPrediction:
    """Predicted next moves for an agent based on its history."""

    agent_id: str
    predicted_provisions: list[str] = field(default_factory=list)
    predicted_requirements: list[str] = field(default_factory=list)
    predicted_constraints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SemanticMatcher(Protocol):
    """Protocol for LLM-powered semantic matching."""

    def check_overlap(self, spec_a: dict[str, Any], spec_b: dict[str, Any]) -> SemanticMatch: ...

    def check_overlap_batch(
        self, pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[SemanticMatch]: ...

    def check_constraint_applies(
        self, constraint: dict[str, Any], intent: dict[str, Any]
    ) -> ConstraintApplicability: ...

    def predict_trajectory(self, agent_history: list[dict[str, Any]]) -> TrajectoryPrediction: ...


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _SemanticCache:
    """In-memory LRU-style cache keyed by SHA256 content hash.

    Max 1000 entries. Evicts oldest quarter when full.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _hash(data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key_data: Any) -> Any | None:
        h = self._hash(key_data)
        if h in self._store:
            self._store.move_to_end(h)
            return self._store[h]
        return None

    def set(self, key_data: Any, value: Any) -> None:
        h = self._hash(key_data)
        if len(self._store) >= self._max_size:
            # Evict oldest quarter
            evict_count = self._max_size // 4
            for _ in range(evict_count):
                self._store.popitem(last=False)
        self._store[h] = value
        self._store.move_to_end(h)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

_OVERLAP_SYSTEM = (
    "You are a semantic code analysis assistant. You determine whether two "
    "interface specifications refer to the same concept, even if they use "
    "different names or conventions. Respond ONLY with valid JSON."
)

_OVERLAP_BATCH_TEMPLATE = (
    "For each pair of interface specs below, determine if they semantically "
    "overlap (refer to the same concept). Consider name synonyms, functional "
    "equivalence, and domain context.\n\n"
    "Pairs:\n{pairs_json}\n\n"
    "Respond with a JSON array of objects, one per pair, each with:\n"
    '  "overlap": boolean,\n'
    '  "confidence": float 0.0-1.0,\n'
    '  "reasoning": string (brief explanation)\n'
)

_CONSTRAINT_SYSTEM = (
    "You are a semantic code analysis assistant. You determine whether a "
    "constraint applies to a given intent based on semantic understanding, "
    "not just tag matching. Respond ONLY with valid JSON."
)

_CONSTRAINT_TEMPLATE = (
    "Does this constraint apply to this intent?\n\n"
    "Constraint:\n{constraint_json}\n\n"
    "Intent:\n{intent_json}\n\n"
    'Respond with JSON: {{"applies": boolean, "confidence": float 0.0-1.0, '
    '"reasoning": string}}'
)

_TRAJECTORY_SYSTEM = (
    "You are a predictive code analysis assistant. Given an agent's history "
    "of intents, predict what the agent will likely build next. Respond ONLY "
    "with valid JSON."
)

_TRAJECTORY_TEMPLATE = (
    "Based on this agent's intent history, predict their next moves:\n\n"
    "History:\n{history_json}\n\n"
    "Respond with JSON:\n"
    '{{"predicted_provisions": [string], "predicted_requirements": [string], '
    '"predicted_constraints": [string], "confidence": float 0.0-1.0, '
    '"reasoning": string}}'
)

_BATCH_SIZE = 10


class AnthropicSemanticMatcher:
    """SemanticMatcher implementation using Anthropic's Claude API.

    Uses Haiku for fast classification tasks (overlap, constraint checks)
    and Sonnet for reasoning tasks (trajectory prediction).
    """

    def __init__(
        self,
        api_key: str | None = None,
        haiku_model: str | None = None,
        sonnet_model: str | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError as err:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicSemanticMatcher. "
                "Install it with: pip install anthropic"
            ) from err

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._haiku = haiku_model or "claude-haiku-4-5-20251001"
        self._sonnet = sonnet_model or "claude-sonnet-4-5-20250929"
        self._cache = _SemanticCache()

    def _call_llm(self, model: str, system: str, prompt: str) -> str:
        """Make a single LLM call and return the text response."""
        response = self._client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_json(self, text: str) -> Any:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            # Strip markdown code block
            lines = text.split("\n")
            lines = lines[1:]  # remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    def check_overlap(self, spec_a: dict[str, Any], spec_b: dict[str, Any]) -> SemanticMatch:
        """Check if two specs semantically overlap."""
        return self.check_overlap_batch([(spec_a, spec_b)])[0]

    def check_overlap_batch(
        self, pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[SemanticMatch]:
        """Check multiple pairs for semantic overlap, batched in chunks of 10."""
        results: list[SemanticMatch] = []

        for chunk_start in range(0, len(pairs), _BATCH_SIZE):
            chunk = pairs[chunk_start : chunk_start + _BATCH_SIZE]
            chunk_results: list[SemanticMatch | None] = [None] * len(chunk)
            uncached_indices: list[int] = []
            uncached_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

            # Check cache first
            for i, (a, b) in enumerate(chunk):
                cache_key = ("overlap", a, b)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    chunk_results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_pairs.append((a, b))

            # Call LLM for uncached pairs
            if uncached_pairs:
                try:
                    pairs_json = json.dumps(
                        [{"spec_a": a, "spec_b": b} for a, b in uncached_pairs],
                        indent=2,
                    )
                    prompt = _OVERLAP_BATCH_TEMPLATE.format(pairs_json=pairs_json)
                    response_text = self._call_llm(self._haiku, _OVERLAP_SYSTEM, prompt)
                    parsed = self._parse_json(response_text)

                    for j, item in enumerate(parsed):
                        match = SemanticMatch(
                            overlap=item.get("overlap", False),
                            confidence=float(item.get("confidence", 0.0)),
                            reasoning=item.get("reasoning", ""),
                        )
                        idx = uncached_indices[j]
                        chunk_results[idx] = match
                        self._cache.set(("overlap", *uncached_pairs[j]), match)
                except Exception:
                    logger.warning("LLM overlap batch call failed, using defaults")
                    for j in range(len(uncached_pairs)):
                        idx = uncached_indices[j]
                        if chunk_results[idx] is None:
                            chunk_results[idx] = SemanticMatch(
                                overlap=False,
                                confidence=0.0,
                                reasoning="LLM call failed",
                            )

            # Fill any remaining None slots (shouldn't happen, but safety)
            for i in range(len(chunk_results)):
                if chunk_results[i] is None:
                    chunk_results[i] = SemanticMatch(
                        overlap=False, confidence=0.0, reasoning="fallback"
                    )

            results.extend(chunk_results)  # type: ignore[arg-type]

        return results

    def check_constraint_applies(
        self, constraint: dict[str, Any], intent: dict[str, Any]
    ) -> ConstraintApplicability:
        """Check if a constraint semantically applies to an intent."""
        cache_key = ("constraint", constraint, intent)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            prompt = _CONSTRAINT_TEMPLATE.format(
                constraint_json=json.dumps(constraint, indent=2),
                intent_json=json.dumps(intent, indent=2),
            )
            response_text = self._call_llm(self._haiku, _CONSTRAINT_SYSTEM, prompt)
            parsed = self._parse_json(response_text)
            result = ConstraintApplicability(
                applies=parsed.get("applies", False),
                confidence=float(parsed.get("confidence", 0.0)),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception:
            logger.warning("LLM constraint check failed, using default")
            result = ConstraintApplicability(
                applies=False, confidence=0.0, reasoning="LLM call failed"
            )

        self._cache.set(cache_key, result)
        return result

    def predict_trajectory(self, agent_history: list[dict[str, Any]]) -> TrajectoryPrediction:
        """Predict an agent's next moves from their intent history."""
        if not agent_history:
            return TrajectoryPrediction(agent_id="", confidence=0.0)

        agent_id = agent_history[0].get("agent_id", "unknown")
        cache_key = ("trajectory", tuple(json.dumps(h, sort_keys=True) for h in agent_history))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            prompt = _TRAJECTORY_TEMPLATE.format(history_json=json.dumps(agent_history, indent=2))
            response_text = self._call_llm(self._sonnet, _TRAJECTORY_SYSTEM, prompt)
            parsed = self._parse_json(response_text)
            result = TrajectoryPrediction(
                agent_id=agent_id,
                predicted_provisions=parsed.get("predicted_provisions", []),
                predicted_requirements=parsed.get("predicted_requirements", []),
                predicted_constraints=parsed.get("predicted_constraints", []),
                confidence=float(parsed.get("confidence", 0.0)),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception:
            logger.warning("LLM trajectory prediction failed, using default")
            result = TrajectoryPrediction(agent_id=agent_id, confidence=0.0)

        self._cache.set(cache_key, result)
        return result
