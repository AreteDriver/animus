#!/usr/bin/env python3
"""Autonomous improvement loop for Animus.

Entry point for cron/systemd timer. Runs a selected citizen's
observation → proposal → queue pipeline headlessly.

Usage::

    python -m animus.autonomous-loop \
        --citizen architect|conversation|knowledge|test \
        [--focus codebase|conversation|evaluation|all]

Exit codes:
    0 — Loop completed, no action needed or proposal queued
    1 — Error during loop
    2 — Proposal generated and queued successfully
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("animus.autonomous-loop")

CITIZEN_REGISTRY: dict[str, tuple[type, str]] = {
    "architect": (
        "animus.citizens.architect.ArchitectCitizen",
        "codebase",
    ),
    "conversation": (
        "animus.citizens.conversation_designer.ConversationDesignerCitizen",
        "conversation",
    ),
    "abstraction": (
        "animus.citizens.abstraction.AbstractionCitizen",
        "codebase",
    ),
    "harvester": (
        "animus.citizens.harvester.HarvesterCitizen",
        "codebase",
    ),
    "intelligence": (
        "animus.citizens.intelligence.IntelligenceCitizen",
        "codebase",
    ),
    "knowledge": (
        "animus.citizens.knowledge_curator.KnowledgeCuratorCitizen",
        "codebase",
    ),
    "pattern": (
        "animus.citizens.pattern.PatternCitizen",
        "codebase",
    ),
    "first_principles": (
        "animus.citizens.first_principles.FirstPrinciplesCitizen",
        "codebase",
    ),
    "architecture_citizen": (
        "animus.citizens.architecture_citizen.ArchitectureCitizen",
        "codebase",
    ),
    "research_guild": (
        "animus.citizens.research_guild.ResearchGuildOrchestrator",
        "codebase",
    ),
    "test": (
        "animus.citizens.test_oracle.TestOracleCitizen",
        "evaluation",
    ),
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _import_citizen(dotted: str):
    mod_path, _, cls_name = dotted.rpartition(".")
    mod = __import__(mod_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Animus autonomous improvement loop")
    parser.add_argument(
        "--citizen",
        choices=list(CITIZEN_REGISTRY.keys()),
        default="architect",
        help="Which citizen to run (default: architect)",
    )
    parser.add_argument(
        "--focus",
        choices=["codebase", "conversation", "evaluation", "all"],
        default=None,
        help="Observation focus (default: citizen-specific)",
    )
    args = parser.parse_args(argv)

    _setup_logging()

    citizen_dotted, default_focus = CITIZEN_REGISTRY[args.citizen]
    focus = args.focus or default_focus
    logger.info("Starting loop — citizen=%s focus=%s", args.citizen, focus)

    try:
        citizen_cls = _import_citizen(citizen_dotted)
        citizen = citizen_cls()
        logger.info("%s instantiated", citizen_cls.__name__)

        # Select observation method by focus
        obs_method = {
            "codebase": getattr(citizen, "observe_codebase", None),
            "conversation": getattr(citizen, "observe_conversations", None),
            "evaluation": getattr(citizen, "observe_evaluations", None),
            "all": None,
        }.get(focus)

        if obs_method is None and focus == "all":
            # Fall back: try all available observe_* methods and merge
            findings: list = []
            for name in dir(citizen):
                if name.startswith("observe_") and callable(getattr(citizen, name)):
                    try:
                        batch = getattr(citizen, name)()
                        if batch:
                            findings.extend(batch if isinstance(batch, list) else [batch])
                            logger.info(
                                "%s returned %d items",
                                name,
                                len(batch) if isinstance(batch, list) else 1,
                            )
                    except Exception as exc:
                        logger.warning("%s failed: %s", name, exc)
        elif obs_method:
            findings = obs_method()
            logger.info("Observation returned %d findings", len(findings))
        else:
            logger.error("No observation method for focus=%s", focus)
            return 1

        # Generate proposal (Architect accepts findings; others analyze internally)
        gen = citizen.generate_proposal
        import inspect

        try:
            sig = inspect.signature(gen)
            takes_arg = (
                len(
                    [
                        p
                        for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty or p.default is not None
                    ]
                )
                > 1
            )
        except Exception:
            takes_arg = False
        if takes_arg:
            proposal = gen(findings or None)
        else:
            proposal = gen()
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
