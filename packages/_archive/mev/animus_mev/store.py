"""JSONL persistence for MEV data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_mev.models import ExtractionResult, MEVOpportunity

logger = logging.getLogger("animus_mev.store")


class MEVStore:
    """Append-only JSONL store for MEV opportunities and results."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._opportunities_file = data_dir / "opportunities.jsonl"
        self._results_file = data_dir / "results.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        """Load existing opportunity IDs for dedup."""
        if not self._opportunities_file.exists():
            return
        with open(self._opportunities_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("opportunity_id", ""))
                    except json.JSONDecodeError:
                        continue

    def is_new(self, opportunity: MEVOpportunity) -> bool:
        """Check if an opportunity hasn't been seen before."""
        return opportunity.opportunity_id not in self._seen_ids

    def record_opportunity(self, opportunity: MEVOpportunity) -> bool:
        """Record an opportunity. Returns True if new, False if duplicate."""
        if not self.is_new(opportunity):
            return False
        with open(self._opportunities_file, "a") as f:
            f.write(json.dumps(opportunity.to_dict()) + "\n")
        self._seen_ids.add(opportunity.opportunity_id)
        return True

    def record_result(self, result: ExtractionResult) -> None:
        """Record an extraction result."""
        with open(self._results_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_opportunities(self) -> list[MEVOpportunity]:
        """Load all recorded opportunities."""
        if not self._opportunities_file.exists():
            return []
        results = []
        with open(self._opportunities_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(MEVOpportunity.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def load_results(self) -> list[ExtractionResult]:
        """Load all extraction results."""
        if not self._results_file.exists():
            return []
        results = []
        with open(self._results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(ExtractionResult.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def total_opportunities(self) -> int:
        return len(self._seen_ids)

    def recent_opportunities(self, limit: int = 20) -> list[MEVOpportunity]:
        """Get the most recent opportunities."""
        all_opps = self.load_opportunities()
        return sorted(all_opps, key=lambda o: o.detected_at, reverse=True)[:limit]

    def recent_results(self, limit: int = 20) -> list[ExtractionResult]:
        """Get the most recent extraction results."""
        all_results = self.load_results()
        return sorted(all_results, key=lambda r: r.timestamp, reverse=True)[:limit]
