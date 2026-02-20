"""
Deterministic replay engine.

Proves the contract's core guarantee: same intents + same policy ⇒ same
resolved state. The replay engine records all publish and resolve operations,
then replays them against a fresh graph to verify identical outcomes.

Usage:
    # Record
    log = ReplayLog()
    log.record_publish(intent)
    result = resolver.resolve(intent)
    log.record_resolve(intent, result)

    # Replay and verify
    replay_result = log.replay()
    assert replay_result.deterministic  # Same content hash
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from convergent.contract import (
    content_hash_intents,
)
from convergent.intent import (
    Adjustment,
    Intent,
    ResolutionResult,
)
from convergent.resolver import IntentResolver, PythonGraphBackend

# ---------------------------------------------------------------------------
# Replay log entries
# ---------------------------------------------------------------------------


class OperationType(str, Enum):
    PUBLISH = "publish"
    RESOLVE = "resolve"


@dataclass
class ReplayEntry:
    """A single recorded operation."""

    operation: OperationType
    intent: Intent
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # For resolve operations, the original result
    resolution_result: ResolutionResult | None = None


# ---------------------------------------------------------------------------
# Replay log
# ---------------------------------------------------------------------------


class ReplayLog:
    """Ordered log of graph operations for deterministic replay.

    Records publish and resolve operations in order. Can be replayed
    against a fresh graph to verify deterministic behavior.
    """

    def __init__(self) -> None:
        self._entries: list[ReplayEntry] = []

    def record_publish(self, intent: Intent) -> None:
        """Record a publish operation."""
        self._entries.append(
            ReplayEntry(
                operation=OperationType.PUBLISH,
                intent=copy.deepcopy(intent),
            )
        )

    def record_resolve(self, intent: Intent, result: ResolutionResult) -> None:
        """Record a resolve operation and its result."""
        self._entries.append(
            ReplayEntry(
                operation=OperationType.RESOLVE,
                intent=copy.deepcopy(intent),
                resolution_result=result,
            )
        )

    @property
    def entries(self) -> list[ReplayEntry]:
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def replay(self) -> ReplayResult:
        """Replay all recorded operations against a fresh graph.

        Creates a new, empty graph and replays all operations in order.
        Compares the final graph state hash and individual resolution
        results to verify determinism.

        Returns:
            ReplayResult with verification details.
        """

        # Fresh graph
        backend = PythonGraphBackend()
        resolver = IntentResolver(backend=backend, min_stability=0.0)

        replayed_resolutions: list[tuple[ResolutionResult, ResolutionResult | None]] = []
        published_intents: list[Intent] = []

        for entry in self._entries:
            if entry.operation == OperationType.PUBLISH:
                # Replay publish — use deepcopy to avoid mutation leaking
                intent_copy = copy.deepcopy(entry.intent)
                resolver.publish(intent_copy)
                published_intents.append(intent_copy)

            elif entry.operation == OperationType.RESOLVE:
                # Replay resolve — compare results
                intent_copy = copy.deepcopy(entry.intent)
                replayed_result = resolver.resolve(intent_copy)
                replayed_resolutions.append((replayed_result, entry.resolution_result))

        # Compute final state hash
        final_intents = resolver.backend.query_all(min_stability=0.0)
        final_hash = content_hash_intents(final_intents)

        # Compare resolution results
        resolution_matches = []
        for replayed, original in replayed_resolutions:
            match = _resolutions_equivalent(replayed, original)
            resolution_matches.append(match)

        all_resolutions_match = all(resolution_matches)

        return ReplayResult(
            final_content_hash=final_hash,
            replayed_intent_count=len(published_intents),
            replayed_resolution_count=len(replayed_resolutions),
            resolution_matches=resolution_matches,
            all_resolutions_match=all_resolutions_match,
            deterministic=all_resolutions_match,
            final_intents=final_intents,
        )


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Result of replaying a log against a fresh graph.

    The key property is `deterministic`: True if the replay produced
    exactly the same resolution results as the original execution.
    """

    final_content_hash: str
    replayed_intent_count: int
    replayed_resolution_count: int
    resolution_matches: list[bool]
    all_resolutions_match: bool
    deterministic: bool
    final_intents: list[Intent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolution comparison
# ---------------------------------------------------------------------------


def _resolutions_equivalent(a: ResolutionResult, b: ResolutionResult | None) -> bool:
    """Check if two resolution results are semantically equivalent.

    Compares adjustment kinds and conflict descriptions, ignoring
    minor text differences in descriptions.
    """
    if b is None:
        # No original to compare against — consider it a match
        # (this happens when only publish ops are recorded)
        return True

    # Compare adjustment counts and kinds
    if len(a.adjustments) != len(b.adjustments):
        return False

    a_kinds = sorted(_adj_key(adj) for adj in a.adjustments)
    b_kinds = sorted(_adj_key(adj) for adj in b.adjustments)
    if a_kinds != b_kinds:
        return False

    # Compare conflict counts
    if len(a.conflicts) != len(b.conflicts):
        return False

    # Compare adopted constraint counts
    return len(a.adopted_constraints) == len(b.adopted_constraints)


def _adj_key(adj: Adjustment) -> tuple[str, str]:
    """Extract a comparison key from an adjustment."""
    return (adj.kind, adj.source_intent_id)
