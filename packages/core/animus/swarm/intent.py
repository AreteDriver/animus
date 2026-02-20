"""Intent graph and resolver for stigmergic agent coordination."""

import threading
from datetime import datetime, timezone

from animus.forge.models import AgentConfig
from animus.logging import get_logger
from animus.swarm.models import IntentEntry

logger = get_logger("swarm.intent")


class IntentGraph:
    """Thread-safe shared intent graph.

    Agents publish their intent before execution and update it after.
    Other agents read the graph to detect conflicts and adjust.
    All mutations are protected by a :class:`threading.Lock` since
    agents within a stage execute in parallel threads.
    """

    def __init__(self) -> None:
        self._entries: dict[str, IntentEntry] = {}
        self._lock = threading.Lock()
        self._history: list[IntentEntry] = []

    def publish(self, entry: IntentEntry) -> None:
        """Publish or update an agent's intent."""
        with self._lock:
            entry.timestamp = datetime.now(timezone.utc)
            self._entries[entry.agent] = entry
            self._history.append(entry)
            logger.debug(
                f"Intent published: {entry.agent} "
                f"provides={entry.provides} stability={entry.stability:.2f}"
            )

    def read_all(self) -> list[IntentEntry]:
        """Read all current intent entries (snapshot, safe to iterate)."""
        with self._lock:
            return list(self._entries.values())

    def get(self, agent_name: str) -> IntentEntry | None:
        """Get a specific agent's current intent."""
        with self._lock:
            return self._entries.get(agent_name)

    def remove(self, agent_name: str) -> None:
        """Remove an agent's intent entry."""
        with self._lock:
            self._entries.pop(agent_name, None)

    def find_conflicts(self, entry: IntentEntry) -> list[IntentEntry]:
        """Find entries that conflict with the given entry.

        Conflict is defined as overlapping ``provides`` fields from
        different agents.
        """
        with self._lock:
            my_provides = set(entry.provides)
            if not my_provides:
                return []
            conflicts = []
            for name, other in self._entries.items():
                if name == entry.agent:
                    continue
                if my_provides & set(other.provides):
                    conflicts.append(other)
            return conflicts

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    @property
    def history(self) -> list[IntentEntry]:
        """Return the full history log (snapshot)."""
        with self._lock:
            return list(self._history)


class IntentResolver:
    """Resolves intent conflicts using stability-based precedence.

    Implements the whitepaper Section 5.4 algorithm:
    higher stability wins, equal stability means both proceed.
    """

    @staticmethod
    def compute_stability(
        agent_config: AgentConfig,
        available_inputs: set[str],
    ) -> float:
        """Compute stability score based on input completeness.

        Returns 1.0 if the agent has no inputs (fully ready), otherwise
        the fraction of satisfied inputs.
        """
        if not agent_config.inputs:
            return 1.0
        satisfied = sum(1 for inp in agent_config.inputs if inp in available_inputs)
        return satisfied / len(agent_config.inputs)

    @staticmethod
    def resolve(entry: IntentEntry, conflicts: list[IntentEntry]) -> str:
        """Determine action for this agent given conflicts.

        Returns:
            ``"proceed"`` if no conflicts or higher stability,
            ``"defer"`` if a conflict has higher stability,
            ``"proceed_both"`` if equal stability (no deadlock).
        """
        if not conflicts:
            return "proceed"

        max_other = max(c.stability for c in conflicts)

        if entry.stability > max_other:
            return "proceed"
        if entry.stability < max_other:
            return "defer"
        return "proceed_both"
