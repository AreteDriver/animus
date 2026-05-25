"""Backfill migration: assign Sensitivity to existing memories (Stage 2.D).

One-shot. Reads all memories in the configured store, applies a heuristic
to assign a tier, queues CONFIDENTIAL / SECRET candidates to
``<data_dir>/audit/backfill-review.jsonl`` for ARETE's three-state review
(``confirm`` / ``downgrade`` / ``upgrade``), and updates each memory
record via ``store.update()``.

The data store remains append-heavy per non-negotiable #3 — this script
never deletes anything, only tier-classifies in place.

Run::

    python -m animus.scripts.migrate_sensitivity_tiers --dry-run
    python -m animus.scripts.migrate_sensitivity_tiers
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.config import AnimusConfig
from animus.memory.layer import MemoryLayer
from animus.memory.types import Memory, Sensitivity

logger = logging.getLogger("migrate_sensitivity")


# Heuristic — substring matches against tag set drive the tier assignment.
# These reflect ARETE's known PII surfaces. Override via env if needed in
# the future; for now the defaults are tuned for this specific account.
CONFIDENTIAL_TAG_NEEDLES: tuple[str, ...] = (
    "tiaid",
    "client",
    "legal",
    "litigation",
    "toyota",
    "ibm-internal",
    "navex",
)
SECRET_TAG_NEEDLES: tuple[str, ...] = (
    "secret",
    "credential",
    "password",
    "private-key",
    "api-key",
)
PUBLIC_SOURCES: tuple[str, ...] = ("harvest",)


def classify(memory: Memory) -> tuple[Sensitivity, str]:
    """Apply heuristic to assign a tier; return (assigned_tier, reason).

    Order matters: SECRET wins over CONFIDENTIAL wins over PUBLIC-source wins
    over PERSONAL fallback. Returns the most-protective match.
    """
    lower_tags = {t.lower() for t in memory.tags}

    for tag in lower_tags:
        for needle in SECRET_TAG_NEEDLES:
            if needle in tag:
                return Sensitivity.SECRET, f"tag '{tag}' matches secret pattern '{needle}'"

    for tag in lower_tags:
        for needle in CONFIDENTIAL_TAG_NEEDLES:
            if needle in tag:
                return (
                    Sensitivity.CONFIDENTIAL,
                    f"tag '{tag}' matches confidential pattern '{needle}'",
                )

    if memory.source in PUBLIC_SOURCES:
        return Sensitivity.PUBLIC, f"source='{memory.source}' is in PUBLIC_SOURCES"

    return Sensitivity.PERSONAL, "default — no other heuristic matched"


def queue_for_review(
    audit_path: Path,
    memory: Memory,
    tier: Sensitivity,
    reason: str,
) -> None:
    """Append a row to backfill-review.jsonl for ARETE's three-state review.

    Row schema:
        - id: memory ID
        - tier_assigned: heuristic's tier verdict
        - reason: human-readable explanation of the match
        - content_preview: first 200 chars of memory content
        - tags: list of original tags
        - source: original source string
        - created_at: when memory was first written
        - review_state: "pending" → user edits to "confirm" / "downgrade" / "upgrade"
    """
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": memory.id,
        "tier_assigned": tier.value,
        "reason": reason,
        "content_preview": memory.content[:200],
        "tags": memory.tags,
        "source": memory.source,
        "created_at": memory.created_at.isoformat(),
        "review_state": "pending",
    }
    with audit_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def migrate(dry_run: bool = False, data_dir: Path | None = None) -> dict[str, Any]:
    """Run the migration. Returns stats dict.

    Args:
        dry_run: If True, compute stats without writing changes.
        data_dir: Override the data directory (defaults to AnimusConfig load).

    Returns:
        Stats dict with: total, already_classified, classified_{tier},
        review_queued, dry_run.
    """
    if data_dir is None:
        config = AnimusConfig.load()
        config.ensure_dirs()
        data_dir = config.data_dir
        backend = config.memory.backend
    else:
        backend = "local"

    layer = MemoryLayer(data_dir, backend=backend)
    audit_path = data_dir / "audit" / "backfill-review.jsonl"

    if audit_path.exists() and not dry_run:
        with audit_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "_session_boundary": datetime.now().isoformat(),
                        "_note": "subsequent rows from a new migration run",
                    }
                )
                + "\n"
            )

    all_memories = layer.store.list_all()
    stats: dict[str, Any] = {
        "total": len(all_memories),
        "already_classified": 0,
        "classified_public": 0,
        "classified_personal": 0,
        "classified_confidential": 0,
        "classified_secret": 0,
        "review_queued": 0,
        "dry_run": dry_run,
    }

    for memory in all_memories:
        # Memories with non-PUBLIC sensitivity were classified at write time
        # (Stage 2.C writers). Trust those — don't second-guess explicit labels.
        if memory.sensitivity is not Sensitivity.PUBLIC:
            stats["already_classified"] += 1
            continue

        new_tier, reason = classify(memory)
        stats[f"classified_{new_tier.value}"] += 1

        if new_tier in (Sensitivity.CONFIDENTIAL, Sensitivity.SECRET):
            stats["review_queued"] += 1
            if not dry_run:
                queue_for_review(audit_path, memory, new_tier, reason)

        if new_tier is not memory.sensitivity and not dry_run:
            memory.sensitivity = new_tier
            layer.store.update(memory)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill sensitivity tiers on existing memories (Stage 2.D)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing changes.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = migrate(dry_run=args.dry_run)

    print(json.dumps(stats, indent=2))
    if args.dry_run:
        print("\nDRY RUN — no changes written. Re-run without --dry-run to apply.")
    else:
        config = AnimusConfig.load()
        audit_file = config.data_dir / "audit" / "backfill-review.jsonl"
        print(f"\nReview queued items at: {audit_file}")
        print(
            "Edit each row's 'review_state' to 'confirm' / 'downgrade' / 'upgrade',\n"
            "then a follow-up commit-backfill-review script will finalize."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
