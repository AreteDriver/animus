#!/usr/bin/env python3
"""Autonomous improvement loop for Animus.

Entry point for cron/systemd timer. Runs Architect → Proposal Queue →
Forge Commissioner → Evaluation in one headless pass.

Usage::

    python -m animus.autonomous-loop [--focus codebase|conversation|evaluation|all]

Exit codes:
    0 — Loop completed, no action needed or proposal queued
    1 — Error during loop (check logs)
    2 — Proposal generated and queued successfully
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger("animus.autonomous-loop")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Animus autonomous improvement loop")
    parser.add_argument(
        "--focus",
        choices=["codebase", "conversation", "evaluation", "all"],
        default="all",
        help="Observation focus (default: all)",
    )
    args = parser.parse_args(argv)

    _setup_logging()
    logger.info("Starting autonomous loop — focus=%s", args.focus)

    try:
        from animus.citizens.architect import ArchitectCitizen

        citizen = ArchitectCitizen()
        logger.info("Architect citizen instantiated")

        # Observe
        if args.focus in ("codebase", "all"):
            findings = citizen.observe_codebase()
            logger.info("Codebase observation: %d findings", len(findings))
        else:
            findings = []

        # Generate proposal
        proposal = citizen.generate_proposal(findings or None)
        if proposal is None:
            logger.info("No actionable findings — nothing to queue")
            return 0

        logger.info("Proposal generated: %s", proposal.title)

        # Store to queue
        stored = citizen.store_proposal(proposal)
        if not stored:
            logger.warning("Proposal generation succeeded but storage failed")
            return 1

        logger.info("Proposal stored successfully")
        return 2

    except Exception as exc:
        logger.exception("Autonomous loop failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
