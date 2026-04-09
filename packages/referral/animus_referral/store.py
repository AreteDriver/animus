"""Persistent JSONL storage for referral data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_referral.models import Account, ReferralBonus, ReferralLink

logger = logging.getLogger("animus_referral.store")


class _JsonlStore:
    """Base class for append-only JSONL storage."""

    def __init__(self, file_path: Path):
        self._file = file_path
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, data: dict) -> None:
        with open(self._file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _load_all_raw(self) -> list[dict]:
        if not self._file.exists():
            return []
        results = []
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def _overwrite(self, items: list[dict]) -> None:
        """Rewrite the entire file (for updates)."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def count(self) -> int:
        if not self._file.exists():
            return 0
        with open(self._file) as f:
            return sum(1 for line in f if line.strip())


class AccountStore(_JsonlStore):
    """Persistent storage for accounts."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir / "accounts.jsonl")

    def save(self, account: Account) -> None:
        """Add or update an account."""
        existing = self._load_all_raw()
        updated = False
        for i, raw in enumerate(existing):
            if raw["alias"] == account.alias and raw["platform"] == account.platform:
                existing[i] = account.to_dict()
                updated = True
                break
        if updated:
            self._overwrite(existing)
        else:
            self._append(account.to_dict())

    def load_all(self) -> list[Account]:
        return [Account.from_dict(d) for d in self._load_all_raw()]

    def get(self, alias: str, platform: str) -> Account | None:
        for d in self._load_all_raw():
            if d["alias"] == alias and d["platform"] == platform:
                return Account.from_dict(d)
        return None

    def get_by_alias(self, alias: str) -> list[Account]:
        return [Account.from_dict(d) for d in self._load_all_raw() if d["alias"] == alias]

    def get_by_platform(self, platform: str) -> list[Account]:
        return [Account.from_dict(d) for d in self._load_all_raw() if d["platform"] == platform]

    def remove(self, alias: str, platform: str) -> bool:
        raw = self._load_all_raw()
        filtered = [d for d in raw if not (d["alias"] == alias and d["platform"] == platform)]
        if len(filtered) == len(raw):
            return False
        self._overwrite(filtered)
        return True


class LinkStore(_JsonlStore):
    """Persistent storage for referral links."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir / "links.jsonl")

    def save(self, link: ReferralLink) -> None:
        """Add or update a link."""
        existing = self._load_all_raw()
        updated = False
        for i, raw in enumerate(existing):
            if (
                raw["program_name"] == link.program_name
                and raw["referrer_account"] == link.referrer_account
            ):
                existing[i] = link.to_dict()
                updated = True
                break
        if updated:
            self._overwrite(existing)
        else:
            self._append(link.to_dict())

    def load_all(self) -> list[ReferralLink]:
        return [ReferralLink.from_dict(d) for d in self._load_all_raw()]

    def get(self, program_name: str, referrer_account: str) -> ReferralLink | None:
        for d in self._load_all_raw():
            if d["program_name"] == program_name and d["referrer_account"] == referrer_account:
                return ReferralLink.from_dict(d)
        return None

    def get_for_program(self, program_name: str) -> list[ReferralLink]:
        return [
            ReferralLink.from_dict(d)
            for d in self._load_all_raw()
            if d["program_name"] == program_name
        ]


class BonusStore(_JsonlStore):
    """Persistent storage for referral bonuses."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir / "bonuses.jsonl")

    def record(self, bonus: ReferralBonus) -> None:
        self._append(bonus.to_dict())

    def load_all(self) -> list[ReferralBonus]:
        return [ReferralBonus.from_dict(d) for d in self._load_all_raw()]

    def get_for_program(self, program_name: str) -> list[ReferralBonus]:
        return [
            ReferralBonus.from_dict(d)
            for d in self._load_all_raw()
            if d["program_name"] == program_name
        ]

    def get_for_account(self, account_alias: str) -> list[ReferralBonus]:
        return [
            ReferralBonus.from_dict(d)
            for d in self._load_all_raw()
            if d["account_alias"] == account_alias
        ]

    def total_earned_usd(self) -> float:
        return sum(
            b.amount_usd for b in self.load_all() if b.status == "paid" or b.status.value == "paid"
        )

    def update_status(
        self, program_name: str, account_alias: str, earned_at_iso: str, new_status: str
    ) -> bool:
        """Update the status of a specific bonus. Returns True if updated."""
        raw = self._load_all_raw()
        updated = False
        for item in raw:
            if (
                item["program_name"] == program_name
                and item["account_alias"] == account_alias
                and item["earned_at"] == earned_at_iso
            ):
                item["status"] = new_status
                updated = True
                break
        if updated:
            self._overwrite(raw)
        return updated
