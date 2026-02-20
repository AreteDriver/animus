"""Configuration for the Convergent coordination protocol (Phase 3).

Separate from the existing intent graph configuration. Controls behavior
of Triumvirate voting, phi-weighted scoring, stigmergy markers, and
the signal bus.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from convergent.protocol import QuorumLevel


@dataclass
class CoordinationConfig:
    """Configuration for the coordination protocol subsystems.

    Attributes:
        db_path: Path to the SQLite database for scores, votes, and markers.
        default_quorum: Default quorum level for consensus requests.
        phi_decay_rate: How fast old outcomes lose influence in phi scoring.
            Higher values = faster decay. Default 0.05.
        stigmergy_evaporation_rate: How fast marker strength decays per day.
            Higher values = faster evaporation. Default 0.1.
        signal_bus_type: Backend for the signal bus
            ("sqlite", "filesystem"). Default "sqlite" for cross-process.
        vote_timeout_seconds: How long to wait for votes before DEADLOCK.
    """

    db_path: str = "./convergent_coordination.db"
    default_quorum: QuorumLevel = QuorumLevel.MAJORITY
    phi_decay_rate: float = 0.05
    stigmergy_evaporation_rate: float = 0.1
    signal_bus_type: str = "sqlite"
    vote_timeout_seconds: int = 300

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        d["default_quorum"] = self.default_quorum.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> CoordinationConfig:
        """Deserialize from JSON string."""
        d = json.loads(data)
        d["default_quorum"] = QuorumLevel(d["default_quorum"])
        return cls(**d)
