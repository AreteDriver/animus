"""
Graph versioning — snapshots, branching, and merging.

Supports point-in-time snapshots for deterministic replay, branching
for speculative work, and merging with conflict detection.

Versioning rules (part of the formal contract):
  - Snapshots are immutable captures of the full graph state.
  - Branches are independent copies that evolve separately.
  - Merging replays the branch's new intents into the target,
    running resolution on each to detect conflicts.
  - Content hashes enable verification of deterministic replay.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from convergent.contract import (
    ConflictClass,
    ResolutionPolicy,
    content_hash_intents,
    validate_publish,
)
from convergent.intent import (
    Intent,
    ResolutionResult,
)
from convergent.resolver import IntentResolver, PythonGraphBackend

# ---------------------------------------------------------------------------
# Graph snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable point-in-time capture of the graph state.

    Snapshots are the basis for branching and deterministic replay.
    Two snapshots with the same content_hash contain semantically
    identical intent sets (order-independent).
    """

    snapshot_id: str
    timestamp: datetime
    intents: tuple[Intent, ...]
    version: int
    source_branch: str

    @staticmethod
    def capture(
        resolver: IntentResolver,
        branch_name: str = "main",
        version: int = 0,
    ) -> GraphSnapshot:
        """Capture the current graph state as a snapshot."""
        all_intents = resolver.backend.query_all(min_stability=0.0)
        return GraphSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            intents=tuple(all_intents),
            version=version,
            source_branch=branch_name,
        )

    @property
    def content_hash(self) -> str:
        """Deterministic content hash of all intents in this snapshot."""
        return content_hash_intents(list(self.intents))

    @property
    def intent_count(self) -> int:
        return len(self.intents)

    def intent_ids(self) -> frozenset[str]:
        """Return the set of intent IDs in this snapshot."""
        return frozenset(i.id for i in self.intents)


# ---------------------------------------------------------------------------
# Merge result
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    """Result of merging one branch into another.

    Attributes:
        merged_intents: Intents successfully merged into the target.
        conflicts: Conflicts detected during merge resolution.
        hard_failures: Intents that caused HARD_FAIL conflicts.
        escalations: Intents that require HUMAN_ESCALATION.
        success: True if no hard failures and no escalations.
        resulting_snapshot: Snapshot of the merged state (if successful).
    """

    merged_intents: list[Intent] = field(default_factory=list)
    conflicts: list[ResolutionResult] = field(default_factory=list)
    hard_failures: list[tuple[Intent, str]] = field(default_factory=list)
    escalations: list[tuple[Intent, str]] = field(default_factory=list)
    success: bool = True
    resulting_snapshot: GraphSnapshot | None = None


# ---------------------------------------------------------------------------
# Versioned graph
# ---------------------------------------------------------------------------


class VersionedGraph:
    """Intent graph with versioning support.

    Wraps an IntentResolver with snapshot/branch/merge capabilities.
    Each VersionedGraph tracks its branch name and version history.

    Usage:
        vg = VersionedGraph("main")
        vg.publish(intent)

        # Take a snapshot
        snap = vg.snapshot()

        # Create a branch
        branch = vg.branch("feature-x")
        branch.publish(new_intent)

        # Merge back
        result = vg.merge(branch)
    """

    def __init__(
        self,
        branch_name: str = "main",
        resolver: IntentResolver | None = None,
        policy: ResolutionPolicy | None = None,
        backend_factory: type | None = None,
    ) -> None:
        self.branch_name = branch_name
        self._backend_factory = backend_factory
        if resolver is not None:
            self.resolver = resolver
        else:
            backend = backend_factory() if backend_factory else PythonGraphBackend()
            self.resolver = IntentResolver(backend=backend, min_stability=0.0)
        self.policy = policy or ResolutionPolicy()
        self._version = 0
        self._snapshots: list[GraphSnapshot] = []

    def publish(self, intent: Intent) -> float:
        """Publish an intent with contract validation.

        Raises ContractViolation if the publish would violate invariants.
        Returns computed stability.
        """
        existing_ids = {i.id for i in self.resolver.backend.query_all(min_stability=0.0)}
        violations = validate_publish(intent, existing_ids)
        if violations:
            raise violations[0]

        return self.resolver.publish(intent)

    def resolve(self, intent: Intent) -> ResolutionResult:
        """Resolve an intent against the current graph state."""
        return self.resolver.resolve(intent)

    def snapshot(self) -> GraphSnapshot:
        """Capture the current state as an immutable snapshot."""
        self._version += 1
        snap = GraphSnapshot.capture(
            self.resolver,
            branch_name=self.branch_name,
            version=self._version,
        )
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> list[GraphSnapshot]:
        """All snapshots taken on this branch."""
        return list(self._snapshots)

    @property
    def version(self) -> int:
        return self._version

    def branch(self, name: str) -> VersionedGraph:
        """Create a new branch from the current state.

        The branch gets an independent copy of all current intents.
        Changes on the branch do not affect this graph until merged.
        """
        current_intents = self.resolver.backend.query_all(min_stability=0.0)
        new_backend = PythonGraphBackend()
        for intent in current_intents:
            new_backend.publish(copy.deepcopy(intent))

        new_resolver = IntentResolver(
            backend=new_backend,
            min_stability=self.resolver.min_stability,
            semantic_matcher=self.resolver.semantic_matcher,
            semantic_confidence_threshold=self.resolver.semantic_confidence_threshold,
        )
        branch = VersionedGraph(
            branch_name=name,
            resolver=new_resolver,
            policy=self.policy,
        )
        branch._version = self._version
        return branch

    def merge(self, other: VersionedGraph) -> MergeResult:
        """Merge another branch's new intents into this graph.

        Identifies intents in `other` that don't exist in this graph,
        then replays them in timestamp order, running resolution on each.
        Conflicts are classified using the resolution policy.

        Args:
            other: The branch to merge from.

        Returns:
            MergeResult with details of what was merged and any conflicts.
        """
        my_ids = {i.id for i in self.resolver.backend.query_all(min_stability=0.0)}
        their_intents = other.resolver.backend.query_all(min_stability=0.0)

        # Find new intents (in other but not in self)
        new_intents = [i for i in their_intents if i.id not in my_ids]
        # Sort by timestamp for causal replay
        new_intents.sort(key=lambda i: i.timestamp)

        result = MergeResult()

        for intent in new_intents:
            # Resolve against current state
            resolution = self.resolver.resolve(intent)

            if resolution.conflicts:
                result.conflicts.append(resolution)

                # Classify each conflict
                for conflict in resolution.conflicts:
                    # Check if any conflict is a hard fail
                    # Hard fail if a critical constraint is involved
                    their_stability = conflict.their_stability
                    my_stability = intent.compute_stability()

                    conflict_class = self.policy.classify_provision_conflict(
                        my_stability, their_stability
                    )

                    if conflict_class == ConflictClass.HARD_FAIL:
                        result.hard_failures.append((intent, conflict.description))
                        result.success = False
                    elif conflict_class == ConflictClass.HUMAN_ESCALATION:
                        result.escalations.append((intent, conflict.description))
                        result.success = False

            # Only merge if no hard failures or escalations for this intent
            blocked = any(i is intent for i, _ in result.hard_failures) or any(
                i is intent for i, _ in result.escalations
            )
            if not blocked:
                self.resolver.publish(intent)
                result.merged_intents.append(intent)

        if result.success:
            self._version += 1
            result.resulting_snapshot = GraphSnapshot.capture(
                self.resolver,
                branch_name=self.branch_name,
                version=self._version,
            )

        return result
