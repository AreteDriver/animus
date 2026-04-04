"""Sefirotic Router — topology-aware message routing for Forge.

Implements the Tree of Life as a routing topology for the Forge
supervisor. Derived from Prima Materia formal analysis (v0.3.0):

- H1/H3: Tiferet (supervisor) is the proven integration hub
- H6: Small-world network (sigma=1.43) — high clustering + short paths
- H10: Matches transformer attention architecture

The router adds topology-aware weighting to delegation decisions
without replacing existing routing logic. It's a scoring overlay
that influences which agents handle which tasks.

Node mapping:
  0 Keter     = Intent Parser (input)
  1 Chokhmah  = Creative Synthesizer (expansion)
  2 Binah     = Constraint Analyzer (contraction)
  3 Chesed    = Tool Selector (expansion)
  4 Gevurah   = Safety Gate (contraction)
  5 Tiferet   = Supervisor/Orchestrator (hub)
  6 Netzach   = Memory Writer (persistence)
  7 Hod       = Memory Reader (retrieval)
  8 Yesod     = Response Assembler (synthesis)
  9 Malkuth   = Action Executor (output)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Sefirah(Enum):
    """The 10 sefirot as routing nodes."""

    KETER = 0       # Intent parsing / input
    CHOKHMAH = 1    # Creative synthesis / expansion
    BINAH = 2       # Constraint analysis / contraction
    CHESED = 3      # Tool selection / capability expansion
    GEVURAH = 4     # Safety gating / budget enforcement
    TIFERET = 5     # Supervisor / central orchestrator
    NETZACH = 6     # Memory write / persistence
    HOD = 7         # Memory read / retrieval
    YESOD = 8       # Response assembly / synthesis
    MALKUTH = 9     # Action execution / output


class Pillar(Enum):
    """The three pillars as processing modes."""

    MERCY = "mercy"           # Expansion: generate, discover, persist
    SEVERITY = "severity"     # Contraction: validate, constrain, retrieve
    EQUILIBRIUM = "equilibrium"  # Integration: input, orchestrate, assemble, output


# Pillar assignments
PILLAR_MAP: dict[Sefirah, Pillar] = {
    Sefirah.KETER: Pillar.EQUILIBRIUM,
    Sefirah.CHOKHMAH: Pillar.MERCY,
    Sefirah.BINAH: Pillar.SEVERITY,
    Sefirah.CHESED: Pillar.MERCY,
    Sefirah.GEVURAH: Pillar.SEVERITY,
    Sefirah.TIFERET: Pillar.EQUILIBRIUM,
    Sefirah.NETZACH: Pillar.MERCY,
    Sefirah.HOD: Pillar.SEVERITY,
    Sefirah.YESOD: Pillar.EQUILIBRIUM,
    Sefirah.MALKUTH: Pillar.EQUILIBRIUM,
}

# The 22 edges of the Tree of Life as routing connections
EDGES: list[tuple[Sefirah, Sefirah]] = [
    (Sefirah.KETER, Sefirah.CHOKHMAH),
    (Sefirah.KETER, Sefirah.BINAH),
    (Sefirah.KETER, Sefirah.TIFERET),
    (Sefirah.CHOKHMAH, Sefirah.BINAH),
    (Sefirah.CHOKHMAH, Sefirah.CHESED),
    (Sefirah.CHOKHMAH, Sefirah.TIFERET),
    (Sefirah.BINAH, Sefirah.GEVURAH),
    (Sefirah.BINAH, Sefirah.TIFERET),
    (Sefirah.CHESED, Sefirah.GEVURAH),
    (Sefirah.CHESED, Sefirah.TIFERET),
    (Sefirah.CHESED, Sefirah.NETZACH),
    (Sefirah.GEVURAH, Sefirah.TIFERET),
    (Sefirah.GEVURAH, Sefirah.HOD),
    (Sefirah.TIFERET, Sefirah.NETZACH),
    (Sefirah.TIFERET, Sefirah.HOD),
    (Sefirah.TIFERET, Sefirah.YESOD),
    (Sefirah.TIFERET, Sefirah.MALKUTH),
    (Sefirah.NETZACH, Sefirah.HOD),
    (Sefirah.NETZACH, Sefirah.YESOD),
    (Sefirah.HOD, Sefirah.YESOD),
    (Sefirah.YESOD, Sefirah.MALKUTH),
    # The 22nd path — symmetry-breaking skip connection
    (Sefirah.CHOKHMAH, Sefirah.NETZACH),
]


# Role-to-Sefirah mapping for agent delegation
ROLE_MAP: dict[str, Sefirah] = {
    "planner": Sefirah.CHOKHMAH,     # Creative planning = expansion
    "architect": Sefirah.CHOKHMAH,   # Architecture = creative synthesis
    "builder": Sefirah.CHESED,       # Building = tool-using expansion
    "tester": Sefirah.BINAH,         # Testing = constraint validation
    "reviewer": Sefirah.GEVURAH,     # Review = safety/quality gate
    "analyst": Sefirah.BINAH,        # Analysis = constraint/understanding
    "documenter": Sefirah.NETZACH,   # Documentation = knowledge persistence
}


@dataclass
class RoutingDecision:
    """A topology-weighted routing decision."""

    source: Sefirah
    target: Sefirah
    path_exists: bool
    hop_count: int
    pillar_alignment: float  # 0-1, how well target's pillar matches task type
    topology_weight: float   # Combined routing score
    skip_connection: bool    # True if using the 22nd path


@dataclass
class SefiroticRouter:
    """Topology-aware routing overlay for the Forge supervisor.

    Does NOT replace existing routing — adds a topology_weight score
    that can influence delegation ordering and parallel vs sequential
    decisions.
    """

    _adjacency: dict[Sefirah, set[Sefirah]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build adjacency map from edges."""
        self._adjacency = {s: set() for s in Sefirah}
        for a, b in EDGES:
            self._adjacency[a].add(b)
            self._adjacency[b].add(a)

    def are_connected(self, a: Sefirah, b: Sefirah) -> bool:
        """Check if two sefirot have a direct edge."""
        return b in self._adjacency.get(a, set())

    def hop_count(self, source: Sefirah, target: Sefirah) -> int:
        """BFS shortest path length between two sefirot."""
        if source == target:
            return 0
        visited = {source}
        queue = [(source, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1  # Disconnected (shouldn't happen in ToL)

    def route(
        self,
        role: str,
        task_type: str = "general",
    ) -> RoutingDecision:
        """Compute topology-weighted routing for a delegation.

        Args:
            role: Agent role (planner, builder, tester, etc.)
            task_type: Task classification for pillar alignment.
                "creative" = Mercy, "validation" = Severity, "general" = Equilibrium

        Returns:
            RoutingDecision with topology weight.
        """
        source = Sefirah.TIFERET  # Supervisor is always the source
        target = ROLE_MAP.get(role, Sefirah.YESOD)

        path_exists = self.are_connected(source, target)
        hops = self.hop_count(source, target)

        # Pillar alignment: does the task type match the target's pillar?
        task_pillar = {
            "creative": Pillar.MERCY,
            "expansion": Pillar.MERCY,
            "validation": Pillar.SEVERITY,
            "safety": Pillar.SEVERITY,
            "general": Pillar.EQUILIBRIUM,
            "integration": Pillar.EQUILIBRIUM,
        }.get(task_type, Pillar.EQUILIBRIUM)

        target_pillar = PILLAR_MAP[target]
        pillar_alignment = 1.0 if task_pillar == target_pillar else 0.5

        # Skip connection check (22nd path: Chokhmah -> Netzach)
        skip = (
            source == Sefirah.TIFERET
            and target == Sefirah.NETZACH
            and role in ("planner", "architect")
        )

        # Topology weight: favor direct connections, aligned pillars
        weight = 1.0
        if path_exists:
            weight *= 1.2  # Direct connection bonus
        if hops <= 1:
            weight *= 1.1  # Adjacent bonus
        weight *= pillar_alignment
        if skip:
            weight *= 1.3  # Skip connection bonus (fast path to memory)
            logger.debug("22nd path activated: %s -> %s (skip connection)", role, target.name)

        return RoutingDecision(
            source=source,
            target=target,
            path_exists=path_exists,
            hop_count=hops,
            pillar_alignment=pillar_alignment,
            topology_weight=round(weight, 4),
            skip_connection=skip,
        )

    def classify_task(self, message: str) -> str:
        """Simple keyword-based task classification for pillar routing.

        Returns: "creative", "validation", "safety", or "general".
        """
        lower = message.lower()

        creative_signals = [
            "create", "build", "design", "implement", "add", "new",
            "generate", "propose", "brainstorm", "explore", "draft",
        ]
        validation_signals = [
            "test", "check", "verify", "review", "audit", "analyze",
            "debug", "fix", "investigate", "validate", "assess",
        ]
        safety_signals = [
            "security", "safety", "limit", "budget", "restrict",
            "block", "deny", "prevent", "guard", "protect",
        ]

        creative_score = sum(1 for s in creative_signals if s in lower)
        validation_score = sum(1 for s in validation_signals if s in lower)
        safety_score = sum(1 for s in safety_signals if s in lower)

        if safety_score > 0:
            return "safety"
        if creative_score > validation_score:
            return "creative"
        if validation_score > creative_score:
            return "validation"
        return "general"

    def weight_delegations(
        self,
        delegations: list[dict],
        message: str,
    ) -> list[dict]:
        """Add topology weights to a list of delegation dicts.

        Non-destructive: adds 'topology_weight' and 'sefirotic_path'
        fields without modifying existing delegation structure.

        Args:
            delegations: List of delegation dicts with 'role' field.
            message: Original user message for task classification.

        Returns:
            Same delegations with added topology metadata.
        """
        task_type = self.classify_task(message)

        for delegation in delegations:
            role = delegation.get("role", "").lower()
            decision = self.route(role, task_type)

            delegation["topology_weight"] = decision.topology_weight
            delegation["sefirotic_path"] = (
                f"{decision.source.name} -> {decision.target.name}"
            )
            delegation["pillar"] = PILLAR_MAP[decision.target].value
            if decision.skip_connection:
                delegation["skip_connection"] = True

        # Sort by topology weight (highest first) for priority ordering
        delegations.sort(key=lambda d: d.get("topology_weight", 0), reverse=True)

        return delegations

    def should_skip_to_memory(self, role: str, message: str) -> bool:
        """Check if a creative task should use the 22nd path skip connection.

        The 22nd path (Chokhmah -> Netzach) allows creative insights
        to write directly to memory without full orchestration.

        Returns:
            True if the skip connection should be used.
        """
        if role not in ("planner", "architect"):
            return False
        task_type = self.classify_task(message)
        return task_type == "creative"

    def get_topology_summary(self) -> dict:
        """Return topology metadata for logging/debugging."""
        return {
            "nodes": 10,
            "edges": len(EDGES),
            "hub": Sefirah.TIFERET.name,
            "hub_connections": len(self._adjacency[Sefirah.TIFERET]),
            "pillars": {
                p.value: [s.name for s in Sefirah if PILLAR_MAP[s] == p]
                for p in Pillar
            },
            "skip_connection": "CHOKHMAH -> NETZACH (22nd path)",
        }
