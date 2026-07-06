"""Animus CLI — argparse entry point for subcommands.

Wires ``animus ingest`` and future structured commands outside the
interactive REPL.
"""

from __future__ import annotations

import argparse
import sys

from animus.workflows.ingest import ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    result = ingest(
        args.url,
        synthesize=args.synthesize,
        tag=args.tag,
    )
    if result.item:
        print(f"item: {result.item.title}")
    if result.synthesis:
        print(f"synthesis: {result.synthesis.title}")
    if result.memory_tags:
        print(f"memory_tags: {len(result.memory_tags)}")
    for err in result.errors:
        print(f"warning: {err.stage} failed — {err.message}", file=sys.stderr)
    if not result.success:
        return 1
    return 0


def _cmd_architect(args: argparse.Namespace) -> int:
    from animus.citizens import ArchitectCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    cb_path = config.citizens.codebase_path or str(config.data_dir.parent)
    log_dir = config.citizens.conversation_log_dir or None
    evidence_dir = config.citizens.evidence_dir or None

    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if args.store else None

    architect = ArchitectCitizen(
        codebase_path=cb_path,
        memory_layer=memory,
        conversation_log_dir=log_dir,
        evidence_dir=evidence_dir,
    )

    print("# Running Architect Citizen scan...", file=sys.stderr)

    if args.focus in ("codebase", "all"):
        obs = architect.observe_codebase()
        if obs:
            print(f"\n## Codebase Observations ({len(obs)} found)", file=sys.stderr)
            for o in obs:
                print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    if args.focus in ("conversation", "all"):
        obs = architect.observe_conversations()
        if obs:
            print(f"\n## Conversation Observations ({len(obs)} found)", file=sys.stderr)
            for o in obs:
                print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    if args.focus in ("evaluation", "all"):
        obs = architect.observe_evaluations()
        if obs:
            print(f"\n## Evaluation Observations ({len(obs)} found)", file=sys.stderr)
            for o in obs:
                print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    report = architect.analyze()
    proposal = architect.generate_proposal(report)

    if proposal:
        print(f"\n## Proposal Generated: {proposal.title}")
        print(f"**ID:** {proposal.id}")
        print(f"**Problem:** {proposal.problem}")
        print(f"**Confidence:** {proposal.confidence.value} ({proposal.confidence_score:.0%})")
        print(f"**Recommendation:** {proposal.recommendation}")
        print(f"**Effort:** {proposal.estimated_effort_hours}h")
        print(f"**Components:** {', '.join(proposal.affected_components)}")
        if proposal.potential_risks:
            print("**Risks:**")
            for r in proposal.potential_risks:
                print(f"  - {r.description} ({r.severity})")
        if args.store and memory:
            stored = architect.store_proposal(proposal)
            if stored:
                print(f"\n✅ Proposal stored in memory.")
    else:
        print("\nNo actionable findings — no proposal generated.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animus")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a URL")
    ingest_parser.add_argument("url")
    ingest_parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Run Ogma synthesis after fetching",
    )
    ingest_parser.add_argument(
        "--tag",
        action="store_true",
        help="Push structured concepts to semantic memory",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    architect_parser = subparsers.add_parser(
        "architect",
        help="Run the Architect Citizen observation and analysis cycle",
    )
    architect_parser.add_argument(
        "--focus",
        choices=["codebase", "conversation", "evaluation", "all"],
        default="all",
        help="Observation focus area (default: all)",
    )
    architect_parser.add_argument(
        "--store",
        action="store_true",
        help="Store generated proposal in Animus memory",
    )
    architect_parser.set_defaults(func=_cmd_architect)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
