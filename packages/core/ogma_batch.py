#!/usr/bin/env python3
"""Batch Ogma synthesis runner for AI-focused media backlog.

Usage:
    python3 ogma_batch.py

Expects to run from packages/core/ with PYTHONPATH set.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bootstrap pathing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from animus.lugh.sources.base import SourceCache
from animus.ogma.read import synthesize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ogma-batch")

TWO_WEEKS = timedelta(days=14)
AI_SOURCES = {
    "youtube:@AIDailyBrief",
    "youtube:@LatentSpacePod",
    "youtube:@NoPriorsPodcast",
    "youtube:@a16z",
    "hn:front_page",
}


def main() -> int:
    cache = SourceCache()
    since = datetime.now(timezone.utc) - TWO_WEEKS
    items = cache.recent(limit=500)

    targets = [
        i for i in items if i.source_id in AI_SOURCES and i.published and i.published >= since
    ]

    # Prioritize YouTube with captions first, then HN
    yt_items = [i for i in targets if i.source_id.startswith("youtube:")]
    [i for i in targets if i.source_id == "hn:front_page"]
    # Retry only the 3 items that failed with phi4:14b
    failed_titles = {
        "The Agent Cloud: Databricks' Bet on the Future of AI",
        "Why AI Users Are Raving About GLM 5.2",
        "AI Security After Codex and Claude Code",
    }
    targets_ordered = [i for i in yt_items if any(t in i.title for t in failed_titles)]

    logger.info("Starting batch: %d items", len(targets_ordered))
    for idx, item in enumerate(targets_ordered, 1):
        logger.info("[%d/%d] %s — %s", idx, len(targets_ordered), item.source_id, item.title[:60])
        from animus.cognitive import ModelConfig, create_model

        try:
            model = create_model(ModelConfig.ollama(model="qwen2.5:32b"))
            result = synthesize(item, model=model)
            if result:
                logger.info("  → written: %s", result.title[:60])
            else:
                logger.info("  → skipped (relevance gate)")
        except Exception as e:
            logger.error("  → FAILED: %s", e)

    logger.info("Batch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
