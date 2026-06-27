"""Persistent storage for bounties, matches, and claims."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_bounty.models import (
    Bounty,
    BountyStatus,
    ClaimResult,
    SkillMatch,
)

logger = logging.getLogger("animus_bounty.store")


class BountyStore:
    """Append-only JSONL store for discovered bounties with dedup."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "bounties.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        """Load existing bounty IDs for dedup."""
        if not self._file.exists():
            return
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("bounty_id", ""))
                    except json.JSONDecodeError:
                        continue

    def is_new(self, bounty: Bounty) -> bool:
        """Check if a bounty hasn't been seen before."""
        return bounty.bounty_id not in self._seen_ids

    def record(self, bounty: Bounty) -> bool:
        """Record a bounty. Returns True if new, False if duplicate."""
        if not self.is_new(bounty):
            return False
        with open(self._file, "a") as f:
            f.write(json.dumps(bounty.to_dict()) + "\n")
        self._seen_ids.add(bounty.bounty_id)
        return True

    def record_batch(self, bounties: list[Bounty]) -> int:
        """Record multiple bounties. Returns count of new ones."""
        new_count = 0
        for bounty in bounties:
            if self.record(bounty):
                new_count += 1
        return new_count

    def load_all(self) -> list[Bounty]:
        """Load all recorded bounties."""
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(Bounty.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def total_count(self) -> int:
        return len(self._seen_ids)

    def by_status(self, status: BountyStatus) -> list[Bounty]:
        return [b for b in self.load_all() if b.status == status]

    def by_platform(self, platform: str) -> list[Bounty]:
        return [b for b in self.load_all() if b.platform.value == platform]

    def get_by_id(self, bounty_id: str) -> Bounty | None:
        """Look up a bounty by its ID."""
        for b in self.load_all():
            if b.bounty_id == bounty_id:
                return b
        return None


class MatchStore:
    """Stores skill match evaluations."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "matches.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record(self, match: SkillMatch) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(match.to_dict()) + "\n")

    def load_all(self) -> list[SkillMatch]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(SkillMatch.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def get_for_bounty(self, bounty_id: str) -> SkillMatch | None:
        for m in self.load_all():
            if m.bounty_id == bounty_id:
                return m
        return None


class ClaimStore:
    """Stores claim lifecycle records."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "claims.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record(self, claim: ClaimResult) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(claim.to_dict()) + "\n")

    def load_all(self) -> list[ClaimResult]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(ClaimResult.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def get_for_bounty(self, bounty_id: str) -> ClaimResult | None:
        for c in self.load_all():
            if c.bounty_id == bounty_id:
                return c
        return None

    def update(self, claim: ClaimResult) -> None:
        """Update a claim by rewriting the file (claim_id = bounty_id)."""
        all_claims = self.load_all()
        updated = False
        with open(self._file, "w") as f:
            for existing in all_claims:
                if existing.bounty_id == claim.bounty_id:
                    f.write(json.dumps(claim.to_dict()) + "\n")
                    updated = True
                else:
                    f.write(json.dumps(existing.to_dict()) + "\n")
        if not updated:
            with open(self._file, "a") as f:
                f.write(json.dumps(claim.to_dict()) + "\n")
