"""Finalize the Stage 2.E backfill review (commit ARETE's three-state pass).

Reads ``<data_dir>/audit/backfill-review.jsonl`` produced by
``migrate_sensitivity_tiers.py`` and applies the decisions in the
``review_state`` field of each row.

**Accepted review_state values** (case-insensitive):

- ``pending`` — no decision yet; row is skipped with a warning.
- ``confirm`` — keep ``tier_assigned`` (the heuristic was right).
- ``downgrade`` — bump to the next-less-sensitive tier
  (SECRET → CONFIDENTIAL → PERSONAL → PUBLIC).
- ``upgrade`` — bump to the next-more-sensitive tier
  (PUBLIC → PERSONAL → CONFIDENTIAL → SECRET).
- ``public`` / ``personal`` / ``confidential`` / ``secret`` — set
  directly to that tier (explicit override).
- ``applied`` — already committed in a prior run; row is skipped.

After applying, each row's ``review_state`` is rewritten to ``applied``
along with a ``final_tier`` field so the file is idempotent — re-running
the script does nothing.

Run::

    python -m animus.scripts.commit_backfill_review
    python -m animus.scripts.commit_backfill_review --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from animus.config import AnimusConfig
from animus.memory.layer import MemoryLayer
from animus.memory.types import Sensitivity

logger = logging.getLogger("commit_backfill_review")


# Ordering for downgrade / upgrade verbs.
_TIER_LADDER = [
    Sensitivity.PUBLIC,
    Sensitivity.PERSONAL,
    Sensitivity.CONFIDENTIAL,
    Sensitivity.SECRET,
]


def _resolve_target(review_state: str, tier_assigned: Sensitivity) -> Sensitivity | None:
    """Map a review_state value to the final Sensitivity tier.

    Returns None when the row should be skipped (``pending`` / ``applied``).
    Raises ``ValueError`` for unrecognized values.
    """
    rs = review_state.strip().lower()

    if rs in ("pending", "applied"):
        return None

    if rs == "confirm":
        return tier_assigned

    idx = _TIER_LADDER.index(tier_assigned)
    if rs == "downgrade":
        return _TIER_LADDER[max(0, idx - 1)]
    if rs == "upgrade":
        return _TIER_LADDER[min(len(_TIER_LADDER) - 1, idx + 1)]

    # Explicit tier name
    try:
        return Sensitivity(rs)
    except ValueError as exc:
        raise ValueError(
            f"Unrecognized review_state {review_state!r} — must be one of: "
            "pending / confirm / downgrade / upgrade / public / personal / "
            "confidential / secret"
        ) from exc


def commit_review(
    dry_run: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply each row's review_state to the live store.

    Returns a stats dict with: total, skipped_pending, skipped_applied,
    applied, errors, dry_run.
    """
    if data_dir is None:
        config = AnimusConfig.load()
        config.ensure_dirs()
        data_dir = config.data_dir
        backend = config.memory.backend
    else:
        backend = "local"

    audit_path = data_dir / "audit" / "backfill-review.jsonl"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"No review file at {audit_path}. Run migrate_sensitivity_tiers first."
        )

    layer = MemoryLayer(data_dir, backend=backend)

    raw_lines = audit_path.read_text().splitlines()
    rows: list[dict[str, Any]] = []
    boundary_lines: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if "_session_boundary" in parsed:
            boundary_lines.append(line)
            continue
        rows.append(parsed)

    stats: dict[str, Any] = {
        "total": len(rows),
        "skipped_pending": 0,
        "skipped_applied": 0,
        "applied": 0,
        "errors": 0,
        "dry_run": dry_run,
    }

    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        review_state = row.get("review_state", "pending")
        try:
            tier_assigned = Sensitivity(row["tier_assigned"])
            target = _resolve_target(review_state, tier_assigned)
        except ValueError as e:
            logger.error("Row %s: %s", row.get("id", "?")[:8], e)
            stats["errors"] += 1
            updated_rows.append(row)
            continue

        if target is None:
            if review_state.lower() == "applied":
                stats["skipped_applied"] += 1
            else:
                stats["skipped_pending"] += 1
                logger.warning(
                    "Row %s left pending — edit review_state and re-run",
                    row["id"][:8],
                )
            updated_rows.append(row)
            continue

        if not dry_run:
            memory = layer.store.retrieve(row["id"])
            if memory is None:
                logger.error("Row %s: memory not found in store", row["id"][:8])
                stats["errors"] += 1
                updated_rows.append(row)
                continue
            memory.sensitivity = target
            layer.store.update(memory)

        stats["applied"] += 1
        updated_rows.append(
            {
                **row,
                "review_state": "applied",
                "final_tier": target.value,
                "committed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not dry_run:
        # Rewrite the file with applied rows + preserve session boundaries.
        new_content = "\n".join(boundary_lines + [json.dumps(r) for r in updated_rows]) + "\n"
        audit_path.write_text(new_content)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Commit the Stage 2.E backfill review against the live store."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing to the store or file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = commit_review(dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))

    if stats["skipped_pending"]:
        print(
            f"\n⚠ {stats['skipped_pending']} row(s) still have review_state='pending'. "
            "Edit those rows and re-run to finalize."
        )
    if stats["errors"]:
        print(f"\n✗ {stats['errors']} row(s) errored — see log output above.")
        return 1
    if args.dry_run:
        print("\nDRY RUN — no changes written.")
    elif stats["applied"]:
        print(
            f"\n✓ {stats['applied']} memory tier(s) committed. The review file is "
            "now idempotent — re-running this script is a no-op."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
