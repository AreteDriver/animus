"""Bridge between Arete Tools and Animus Core.

Syncs Verdict decisions into episodic memory and Calibrate stats
into the identity model. All imports are optional — graceful no-op
when tools are not installed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from verdict.store import DecisionStore

    HAS_VERDICT = True
except ImportError:
    HAS_VERDICT = False

try:
    import calibrate.stats  # noqa: F401

    HAS_CALIBRATE = True
except ImportError:
    HAS_CALIBRATE = False

__all__ = [
    "HAS_VERDICT",
    "HAS_CALIBRATE",
    "calibrate",
    "sync_verdict_to_memory",
    "sync_calibrate_to_identity",
    "auto_sync_verdicts",
]


def sync_verdict_to_memory(memory_layer, decision: dict) -> object | None:
    """Store a Verdict decision as episodic memory.

    Args:
        memory_layer: Animus MemoryLayer instance
        decision: Dict with keys: id, title, reasoning, alternatives, category, review_date

    Returns:
        Memory object if stored, None if skipped
    """
    from animus.memory import MemoryType

    title = decision.get("title", "Untitled Decision")
    reasoning = decision.get("reasoning", "")
    category = decision.get("category", "general")
    alternatives = decision.get("alternatives", [])
    decision_id = decision.get("id", "")
    review_date = decision.get("review_date", "")

    content = f"Decision: {title}\n\nReasoning: {reasoning}"
    if alternatives:
        content += f"\n\nAlternatives considered: {', '.join(str(a) for a in alternatives)}"
    if review_date:
        content += f"\n\nReview date: {review_date}"

    tags = ["verdict", "decision", category]
    metadata = {
        "decision_id": decision_id,
        "category": category,
    }
    if review_date:
        metadata["review_date"] = review_date

    return memory_layer.remember(
        content=content,
        memory_type=MemoryType.EPISODIC,
        tags=tags,
        source="learned",
        confidence=0.95,
        subtype="decision",
        metadata=metadata,
    )


def sync_calibrate_to_identity(identity, stats: dict) -> None:
    """Update identity model from Calibrate accuracy stats.

    Args:
        identity: AnimusIdentity instance
        stats: Dict with keys: accuracy, total_predictions, overconfidence_rate, domains
    """
    accuracy = stats.get("accuracy", 0.0)
    total = stats.get("total_predictions", 0)
    overconfidence = stats.get("overconfidence_rate", 0.0)
    domains = stats.get("domains", {})

    summary = (
        f"Calibration check: {accuracy:.1%} accuracy across {total} predictions. "
        f"Overconfidence rate: {overconfidence:.1%}."
    )

    improvements = []
    if overconfidence > 0.2:
        improvements.append("Reduce confidence on uncertain predictions")
    if accuracy < 0.7:
        improvements.append("Review prediction methodology for low-accuracy domains")

    for domain, domain_stats in domains.items():
        domain_acc = domain_stats.get("accuracy", 0.0)
        if domain_acc < 0.6:
            improvements.append(
                f"Focus on improving {domain} predictions ({domain_acc:.0%} accuracy)"
            )

    identity.record_reflection(summary=summary, improvements=improvements or None)


def auto_sync_verdicts(
    memory_layer,
    db_path: Path | None = None,
    since: datetime | None = None,
) -> int:
    """Batch sync Verdict decisions to episodic memory.

    Args:
        memory_layer: Animus MemoryLayer instance
        db_path: Path to Verdict SQLite DB (uses default if None)
        since: Only sync decisions after this datetime (UTC)

    Returns:
        Number of decisions synced
    """
    if not HAS_VERDICT:
        logger.debug("Verdict not installed, skipping auto_sync")
        return 0

    store = DecisionStore(db_path=db_path) if db_path else DecisionStore()
    decisions = store.list_decisions(since=since)

    count = 0
    for decision in decisions:
        try:
            sync_verdict_to_memory(memory_layer, decision)
            count += 1
        except Exception:
            logger.warning("Failed to sync verdict %s", decision.get("id", "?"))

    logger.info("Synced %d verdicts to episodic memory", count)
    return count
