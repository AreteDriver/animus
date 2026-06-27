"""Persistent storage for opportunities and evaluations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_prospector.models import (
    EvaluationResult,
    Opportunity,
    OpportunityStatus,
)

logger = logging.getLogger("animus_prospector.store")


class OpportunityStore:
    """Append-only store for discovered opportunities with dedup."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "opportunities.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        """Load existing opportunity IDs for dedup."""
        if not self._file.exists():
            return
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("opportunity_id", ""))
                    except json.JSONDecodeError:
                        continue

    def is_new(self, opportunity: Opportunity) -> bool:
        """Check if an opportunity hasn't been seen before."""
        return opportunity.opportunity_id not in self._seen_ids

    def record(self, opportunity: Opportunity) -> bool:
        """Record an opportunity. Returns True if new, False if duplicate."""
        if not self.is_new(opportunity):
            return False
        with open(self._file, "a") as f:
            f.write(json.dumps(opportunity.to_dict()) + "\n")
        self._seen_ids.add(opportunity.opportunity_id)
        return True

    def record_batch(self, opportunities: list[Opportunity]) -> int:
        """Record multiple opportunities. Returns count of new ones."""
        new_count = 0
        for opp in opportunities:
            if self.record(opp):
                new_count += 1
        return new_count

    def load_all(self) -> list[Opportunity]:
        """Load all recorded opportunities."""
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(Opportunity.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def total_count(self) -> int:
        return len(self._seen_ids)

    def by_status(self, status: OpportunityStatus) -> list[Opportunity]:
        return [o for o in self.load_all() if o.status == status]

    def by_type(self, opp_type: str) -> list[Opportunity]:
        return [o for o in self.load_all() if o.opportunity_type.value == opp_type]


class EvaluationStore:
    """Stores evaluation results."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "evaluations.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record(self, result: EvaluationResult) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_all(self) -> list[EvaluationResult]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(EvaluationResult.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def get_for_opportunity(self, opportunity_id: str) -> EvaluationResult | None:
        for r in self.load_all():
            if r.opportunity_id == opportunity_id:
                return r
        return None
