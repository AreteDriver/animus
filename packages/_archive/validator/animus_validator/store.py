"""Persistent storage for validator data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_validator.models import (
    CompoundAction,
    Network,
    Node,
    RewardRecord,
    StakePosition,
)

logger = logging.getLogger("animus_validator.store")


class PositionStore:
    """JSONL store for staking positions with dedup."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "positions" / "stakes.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        if not self._file.exists():
            return
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("position_id", ""))
                    except json.JSONDecodeError:
                        continue

    def is_new(self, position: StakePosition) -> bool:
        return position.position_id not in self._seen_ids

    def record(self, position: StakePosition) -> bool:
        """Record a position. Returns True if new, False if duplicate."""
        if not self.is_new(position):
            return False
        with open(self._file, "a") as f:
            f.write(json.dumps(position.to_dict()) + "\n")
        self._seen_ids.add(position.position_id)
        return True

    def record_batch(self, positions: list[StakePosition]) -> int:
        new_count = 0
        for pos in positions:
            if self.record(pos):
                new_count += 1
        return new_count

    def load_all(self) -> list[StakePosition]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(StakePosition.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def total_count(self) -> int:
        return len(self._seen_ids)

    def by_network(self, network: Network) -> list[StakePosition]:
        return [p for p in self.load_all() if p.network == network]


class RewardStore:
    """JSONL store for reward records."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "rewards" / "history.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, reward: RewardRecord) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(reward.to_dict()) + "\n")

    def record_batch(self, rewards: list[RewardRecord]) -> int:
        count = 0
        for r in rewards:
            self.record(r)
            count += 1
        return count

    def load_all(self) -> list[RewardRecord]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(RewardRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def by_network(self, network: Network) -> list[RewardRecord]:
        return [r for r in self.load_all() if r.network == network]

    def total_earned(self) -> float:
        return sum(r.amount for r in self.load_all())

    def total_earned_usd(self) -> float:
        return sum(r.amount_usd for r in self.load_all())


class CompoundStore:
    """JSONL store for compound actions."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "compounds" / "actions.jsonl"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: CompoundAction) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(action.to_dict()) + "\n")

    def load_all(self) -> list[CompoundAction]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(CompoundAction.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def total_compounded(self) -> float:
        return sum(a.amount_compounded for a in self.load_all())


class NodeStore:
    """JSONL store for node status snapshots."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._file = data_dir / "nodes.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        if not self._file.exists():
            return
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("node_id", ""))
                    except json.JSONDecodeError:
                        continue

    def record(self, node: Node) -> bool:
        """Record or update a node. Returns True if new."""
        is_new = node.node_id not in self._seen_ids
        with open(self._file, "a") as f:
            f.write(json.dumps(node.to_dict()) + "\n")
        self._seen_ids.add(node.node_id)
        return is_new

    def load_all(self) -> list[Node]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(Node.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def load_latest(self) -> dict[str, Node]:
        """Load latest snapshot per node_id (last write wins)."""
        latest: dict[str, Node] = {}
        for node in self.load_all():
            latest[node.node_id] = node
        return latest

    def total_count(self) -> int:
        return len(self._seen_ids)
