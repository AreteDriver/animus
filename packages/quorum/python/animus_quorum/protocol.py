"""Coordination protocol data models for multi-agent consensus and signaling.

Defines the foundational types used by the Triumvirate voting engine,
Stigmergy trail markers, Signal Bus, and Flocking coordinator. These are
the "words" agents use to communicate during consensus, signaling, and
inter-agent learning.

All protocol messages are typed dataclasses with JSON serialization support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class QuorumLevel(str, Enum):
    """How many agents must agree for a decision to pass.

    Used by the Triumvirate voting engine to determine when consensus
    has been reached.
    """

    ANY = "any"  # 1 of N — low-risk reads
    MAJORITY = "majority"  # >50% — medium-risk, recoverable
    UNANIMOUS = "unanimous"  # all — high-risk, irreversible
    UNANIMOUS_HUMAN = "unanimous_human"  # all + human confirm — critical


class VoteChoice(str, Enum):
    """The choices available to a voting agent."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"  # Agent says "I'm not qualified to judge this"


class DecisionOutcome(str, Enum):
    """The possible outcomes of a consensus round."""

    APPROVED = "approved"
    REJECTED = "rejected"
    DEADLOCK = "deadlock"  # No quorum reached
    ESCALATED = "escalated"  # One or more agents escalated


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentIdentity:
    """Identifies an agent in the coordination system.

    Attributes:
        agent_id: Unique identifier for the agent.
        role: The agent's role (e.g. "planner", "builder", "tester", "reviewer").
        model: The model powering the agent (e.g. "claude:sonnet", "ollama:qwen2.5-coder").
        phi_score: Current trust score (0.0 - 1.0). Updated by PhiScorer.
    """

    agent_id: str
    role: str
    model: str
    phi_score: float = 0.5

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> AgentIdentity:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


@dataclass(frozen=True)
class Vote:
    """A single agent's vote on a consensus request.

    Attributes:
        agent: Identity of the voting agent.
        choice: The vote cast.
        confidence: How sure the agent is (0.0 - 1.0).
        reasoning: Why the agent voted this way.
        timestamp: When the vote was cast (ISO 8601 UTC).
        weighted_score: phi_score * confidence, calculated after creation.
    """

    agent: AgentIdentity
    choice: VoteChoice
    confidence: float
    reasoning: str
    timestamp: str = field(default_factory=_utc_now_iso)
    weighted_score: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        d["choice"] = self.choice.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> Vote:
        """Deserialize from JSON string."""
        d = json.loads(data)
        d["agent"] = AgentIdentity(**d["agent"])
        d["choice"] = VoteChoice(d["choice"])
        return cls(**d)


@dataclass(frozen=True)
class ConsensusRequest:
    """A request for agents to vote on.

    Attributes:
        request_id: Unique identifier for this request.
        task_id: Gorgon task this relates to.
        question: What are we deciding?
        context: Relevant information for voters.
        quorum: Required agreement level.
        artifacts: File paths, PR URLs, etc. relevant to the decision.
        timeout_seconds: How long to wait for votes.
        requested_at: When the request was created (ISO 8601 UTC).
    """

    request_id: str
    task_id: str
    question: str
    context: str
    quorum: QuorumLevel
    artifacts: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    requested_at: str = field(default_factory=_utc_now_iso)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        d["quorum"] = self.quorum.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> ConsensusRequest:
        """Deserialize from JSON string."""
        d = json.loads(data)
        d["quorum"] = QuorumLevel(d["quorum"])
        return cls(**d)


@dataclass
class Decision:
    """The outcome of a consensus round.

    Not frozen because weighted totals and reasoning summary are
    calculated after votes are collected.

    Attributes:
        request: The original consensus request.
        votes: All votes cast.
        outcome: The final decision.
        total_weighted_approve: Sum of weighted scores for approve votes.
        total_weighted_reject: Sum of weighted scores for reject votes.
        decided_at: When the decision was made (ISO 8601 UTC).
        human_override: If a human overrode the decision, their reasoning.
        reasoning_summary: Aggregated reasoning from votes.
    """

    request: ConsensusRequest
    votes: list[Vote]
    outcome: DecisionOutcome
    total_weighted_approve: float = 0.0
    total_weighted_reject: float = 0.0
    decided_at: str = field(default_factory=_utc_now_iso)
    human_override: str | None = None
    reasoning_summary: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["request"]["quorum"] = self.request.quorum.value
        for i, vote in enumerate(self.votes):
            d["votes"][i]["choice"] = vote.choice.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> Decision:
        """Deserialize from JSON string."""
        d = json.loads(data)
        d["request"]["quorum"] = QuorumLevel(d["request"]["quorum"])
        d["request"] = ConsensusRequest(**d["request"])
        for i, v in enumerate(d["votes"]):
            v["agent"] = AgentIdentity(**v["agent"])
            v["choice"] = VoteChoice(v["choice"])
            d["votes"][i] = Vote(**v)
        d["outcome"] = DecisionOutcome(d["outcome"])
        return cls(**d)


@dataclass(frozen=True)
class StigmergyMarker:
    """A trail marker left by an agent for future agents to find.

    Like ant pheromone trails — markers carry information that decays
    over time (evaporation) so stale data fades naturally.

    Attributes:
        marker_id: Unique identifier for this marker.
        agent_id: The agent that left the marker.
        marker_type: Category (e.g. "file_modified", "known_issue", "pattern_found").
        target: What this marker refers to (file path, module name, etc.).
        content: The actual information.
        strength: Marker strength, decays over time (1.0 = fresh).
        created_at: When the marker was created (ISO 8601 UTC).
        expires_at: Optional explicit expiry time (ISO 8601 UTC).
    """

    marker_id: str
    agent_id: str
    marker_type: str
    target: str
    content: str
    strength: float = 1.0
    created_at: str = field(default_factory=_utc_now_iso)
    expires_at: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> StigmergyMarker:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


@dataclass(frozen=True)
class Signal:
    """A message on the signal bus.

    Signals are how agents communicate events: "I'm blocked on X",
    "I finished Y, you can start Z", "I found a conflict in file W."

    Attributes:
        signal_type: Event type (e.g. "task_complete", "blocked", "conflict").
        source_agent: The agent that sent the signal.
        target_agent: Specific recipient, or None for broadcast.
        payload: JSON string with signal-specific data.
        timestamp: When the signal was sent (ISO 8601 UTC).
    """

    signal_type: str
    source_agent: str
    target_agent: str | None = None
    payload: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> Signal:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))
