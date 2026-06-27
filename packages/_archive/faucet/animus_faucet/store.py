"""Persistent storage for claim history and faucet health."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_faucet.models import ClaimResult, FaucetHealth

logger = logging.getLogger("animus_faucet.store")


class ClaimStore:
    """Append-only store for claim results."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "claims.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record(self, result: ClaimResult) -> None:
        """Append a claim result to the log."""
        with open(self._file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_all(self) -> list[ClaimResult]:
        """Load all recorded claims."""
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(ClaimResult.from_dict(json.loads(line)))
        return results

    def total_claims(self) -> int:
        """Count total claim attempts."""
        if not self._file.exists():
            return 0
        with open(self._file) as f:
            return sum(1 for line in f if line.strip())

    def total_earned_usd(self) -> float:
        """Sum all successful USD earnings."""
        return sum(
            r.amount_usd for r in self.load_all() if r.succeeded and r.amount_usd is not None
        )

    def claims_for_faucet(self, faucet_name: str) -> list[ClaimResult]:
        """Get all claims for a specific faucet."""
        return [r for r in self.load_all() if r.faucet_name == faucet_name]

    def success_rate(self) -> float:
        """Overall success rate across all claims."""
        claims = self.load_all()
        if not claims:
            return 0.0
        succeeded = sum(1 for r in claims if r.succeeded)
        return succeeded / len(claims)


class HealthStore:
    """Persists per-faucet health status."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "health.json"
        self._health: dict[str, FaucetHealth] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            with open(self._file) as f:
                data = json.load(f)
            for name, health_data in data.items():
                self._health[name] = FaucetHealth.from_dict(health_data)

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w") as f:
            json.dump(
                {name: h.to_dict() for name, h in self._health.items()},
                f,
                indent=2,
            )

    def get(self, name: str) -> FaucetHealth | None:
        return self._health.get(name)

    def save(self, health: FaucetHealth) -> None:
        self._health[health.name] = health
        self._save()

    def get_all(self) -> list[FaucetHealth]:
        return list(self._health.values())

    def remove(self, name: str) -> bool:
        if name in self._health:
            del self._health[name]
            self._save()
            return True
        return False
