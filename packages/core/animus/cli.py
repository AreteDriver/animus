"""Animus CLI — argparse entry point for subcommands.

Wires ``animus ingest`` and future structured commands outside the
interactive REPL.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta

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


def _cmd_conversation_designer(args: argparse.Namespace) -> int:
    from animus.citizens import ConversationDesignerCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    log_dir = args.log_dir or config.citizens.conversation_log_dir or None
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if args.store else None

    designer = ConversationDesignerCitizen(
        conversation_log_dir=log_dir,
        memory_layer=memory,
    )

    print("# Running Conversation Designer scan...", file=sys.stderr)

    repeated = designer.observe_repeated_prompts()
    if repeated:
        print(f"\n## Repeated Prompts ({len(repeated)} found)", file=sys.stderr)
        for o in repeated:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    vague = designer.observe_vague_requests()
    if vague:
        print(f"\n## Vague Requests ({len(vague)} found)", file=sys.stderr)
        for o in vague:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    corrections = designer.observe_correction_loops()
    if corrections:
        print(f"\n## Correction Loops ({len(corrections)} found)", file=sys.stderr)
        for o in corrections:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    proposal = designer.generate_proposal()

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
            stored = designer.store_proposal(proposal)
            if stored:
                print(f"\n✅ Proposal stored in memory.")
    else:
        print("\nNo actionable conversation patterns — no proposal generated.")

    return 0


def _cmd_knowledge_curator(args: argparse.Namespace) -> int:
    from animus.citizens import KnowledgeCuratorCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    cb_path = args.codebase_path or config.citizens.codebase_path or str(config.data_dir.parent)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if args.store else None

    curator = KnowledgeCuratorCitizen(
        codebase_path=cb_path,
        memory_layer=memory,
    )

    print("# Running Knowledge Curator scan...", file=sys.stderr)

    stale = curator.observe_stale_references()
    if stale:
        print(f"\n## Stale References ({len(stale)} found)", file=sys.stderr)
        for o in stale:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    contradictions = curator.observe_contradictions()
    if contradictions:
        print(f"\n## Contradictions ({len(contradictions)} found)", file=sys.stderr)
        for o in contradictions:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    outdated = curator.observe_outdated_claims()
    if outdated:
        print(f"\n## Outdated Claims ({len(outdated)} found)", file=sys.stderr)
        for o in outdated:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    orphans = curator.observe_orphan_topics()
    if orphans:
        print(f"\n## Orphan Topics ({len(orphans)} found)", file=sys.stderr)
        for o in orphans:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    proposal = curator.generate_proposal()

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
            stored = curator.store_proposal(proposal)
            if stored:
                print(f"\n✅ Proposal stored in memory.")
    else:
        print("\nNo actionable knowledge drift — no proposal generated.")

    return 0


def _cmd_test_oracle(args: argparse.Namespace) -> int:
    from animus.citizens import TestOracleCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    cb_path = args.codebase_path or config.citizens.codebase_path or str(config.data_dir.parent)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if args.store else None

    oracle = TestOracleCitizen(
        codebase_path=cb_path,
        memory_layer=memory,
    )

    print("# Running Test Oracle scan...", file=sys.stderr)

    failures = oracle.observe_test_failures()
    if failures:
        print(f"\n## Test Failures ({len(failures)} found)", file=sys.stderr)
        for o in failures:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    gaps = oracle.observe_coverage_gaps()
    if gaps:
        print(f"\n## Coverage Gaps ({len(gaps)} found)", file=sys.stderr)
        for o in gaps:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    drift = oracle.observe_eval_drift()
    if drift:
        print(f"\n## Eval Drift ({len(drift)} found)", file=sys.stderr)
        for o in drift:
            print(f"- [{o.severity.upper()}] {o.description}", file=sys.stderr)

    proposal = oracle.generate_proposal()

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
            stored = oracle.store_proposal(proposal)
            if stored:
                print(f"\n✅ Proposal stored in memory.")
    else:
        print("\nNo actionable quality regressions — no proposal generated.")

    return 0


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------


def _cmd_session(args: argparse.Namespace) -> int:
    from animus_kernel.head.repl import HeadREPL

    timer = None
    if args.timer:
        value = args.timer.strip().lower()
        if value.endswith("h"):
            timer = timedelta(hours=int(value[:-1]))
        elif value.endswith("m"):
            timer = timedelta(minutes=int(value[:-1]))
        elif value.endswith("s"):
            timer = timedelta(seconds=int(value[:-1]))
        else:
            timer = timedelta(minutes=int(value))

    wrapup = args.wrapup_at if args.wrapup_at < 1.0 else 1.0

    try:
        repl = HeadREPL(
            model=args.model,
            project_root=args.project,
            session_timer=timer,
            wrapup_threshold=wrapup,
        )
        if args.no_restart and repl._session_controller:
            repl._session_controller.policy.auto_restart = False
        repl.start()
    except RuntimeError as exc:
        print(f"Failed to start session: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye.")
    return 0


def _cmd_session_steward(args: argparse.Namespace) -> int:
    from animus.citizens import SessionStewardCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    if not config.citizens.session_steward_enabled:
        print("Session Steward is disabled. Set citizens.session_steward_enabled=true to use it.", file=sys.stderr)
        return 1

    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if not args.no_store else None

    telemetry_data = None
    if args.telemetry_file:
        import json
        import os

        path = os.path.expanduser(args.telemetry_file)
        if not os.path.exists(path):
            print(f"Telemetry file not found: {path}", file=sys.stderr)
            return 1
        with open(path, "r", encoding="utf-8") as fh:
            telemetry_data = fh.read()
    else:
        print("No --telemetry-file provided. Session Steward requires telemetry data.", file=sys.stderr)
        return 1

    steward = SessionStewardCitizen(
        min_sessions=5,
        memory_layer=memory,
    )

    # Reconstruct a minimal SessionController from JSON data
    try:
        data = json.loads(telemetry_data)
        from animus_kernel.head.session_controller import SessionController, SessionPolicy

        policy = SessionPolicy(
            wrapup_threshold=data.get("wrapup_threshold", 0.96),
            session_timer=timedelta(minutes=data.get("session_timer_minutes", 30)),
            auto_restart=data.get("auto_restart", True),
        )
        controller = SessionController(policy=policy)
        for ev in data.get("events", []):
            from animus_kernel.head.session_controller import SessionLifecycleEvent

            controller.log_event(
                session_id=ev.get("session_id", "unknown"),
                event=SessionLifecycleEvent[ev.get("event", "RUNNING")],
                utilization_percent=ev.get("utilization_percent", 0.0),
                elapsed_seconds=ev.get("elapsed_seconds", 0.0),
                turns=ev.get("turns", 0),
                message=ev.get("message", ""),
            )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Failed to parse telemetry data: {exc}", file=sys.stderr)
        return 1

    print("# Session Steward Scan Report\n")

    patterns = steward.observe_telemetry(controller)
    if patterns:
        print(f"## Detected Patterns ({len(patterns)})\n")
        for p in patterns:
            print(f"- **[{p.heuristic}]** {p.description} ({p.severity})")
        print()
    else:
        print("## No Patterns Detected")
        print("Either insufficient telemetry (<5 sessions) or no inefficiencies found.\n")

    proposal = steward.generate_proposal(patterns)
    if proposal:
        print("## Improvement Proposal Generated")
        print(f"**ID:** `{proposal.id}`")
        print(f"**Title:** {proposal.title}")
        print(f"**Problem:** {proposal.problem}")
        print(f"**Confidence:** {proposal.confidence.value} ({proposal.confidence_score:.0%})")
        print(f"**Recommendation:**")
        print(proposal.recommendation)
        if proposal.potential_risks:
            print("**Risks:**")
            for r in proposal.potential_risks:
                print(f"  - {r.description} ({r.severity}) — {r.mitigation}")
        print()

        if not args.no_store:
            stored = steward.store_proposal(proposal)
            if stored:
                print("✅ Proposal stored in memory for review.")
            else:
                print("⚠️ Memory layer unavailable — proposal not persisted.")
    else:
        print("## No Proposal Generated")
        print("No actionable inefficiency patterns detected.")

    return 0


# ------------------------------------------------------------------
# Proposal Queue
# ------------------------------------------------------------------


def _cmd_proposal_queue_list(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    status = args.status
    if status == "all":
        items = list(queue._proposals.values())
    elif status == "pending":
        items = queue.list_pending()
    elif status == "approved":
        items = queue.list_approved()
    elif status == "commissioned":
        items = queue.list_commissioned()
    elif status == "complete":
        items = queue.list_completed()
    elif status == "rejected":
        items = queue.list_rejected()
    elif status == "backlog":
        items = queue.get_backlog()
    else:
        print(f"Unknown status: {status}", file=sys.stderr)
        return 1

    if not items:
        print(f"No proposals with status '{status}' found.")
        return 0

    print(f"# Proposal Queue ({status})\n")
    for qp in items:
        p = qp.proposal
        print(f"## {p.id}")
        print(f"  Title:      {p.title}")
        print(f"  Status:     {qp.current_status.value}")
        print(f"  Priority:   {qp.priority}")
        print(f"  Tags:       {', '.join(qp.tags) if qp.tags else 'none'}")
        print(f"  Confidence: {p.confidence.value} ({p.confidence_score:.0%})")
        print(f"  Effort:     {p.estimated_effort_hours}h")
        print(f"  Problem:    {p.problem[:120]}...")
        print(f"  Recommendation: {p.recommendation[:120]}...")
        if qp.transitions:
            last = qp.transitions[-1]
            print(f"  Last action: {last.from_status.value} → {last.to_status.value} by {last.actor}")
        print()

    return 0


def _cmd_proposal_queue_approve(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    result = queue.approve(args.proposal_id, actor="human", reason=args.reason)
    if result is None:
        print(f"Proposal '{args.proposal_id}' not found.", file=sys.stderr)
        return 1

    print(f"✅ Proposal {args.proposal_id} approved.")
    print(f"   Status: {result.current_status.value}")
    return 0


def _cmd_proposal_queue_reject(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    result = queue.reject(args.proposal_id, actor="human", reason=args.reason)
    if result is None:
        print(f"Proposal '{args.proposal_id}' not found.", file=sys.stderr)
        return 1

    print(f"❌ Proposal {args.proposal_id} rejected.")
    print(f"   Status: {result.current_status.value}")
    return 0


def _cmd_proposal_queue_stats(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    stats = queue.stats()
    print("# Proposal Queue Statistics\n")
    for key, value in stats.items():
        print(f"  {key:15s}: {value}")
    return 0


def _cmd_citizen_council_backlog(args: argparse.Namespace) -> int:
    from animus.citizens import CitizenCouncil
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    council = CitizenCouncil(memory_layer=memory)
    count = council.collect_from_memory()
    if count == 0:
        print("No proposals found in memory. Run citizen scans first.")
        return 0

    ranked = council.rank_backlog(deduplicate=args.deduplicate)
    if not ranked:
        print("Backlog is empty after ranking.")
        return 0

    print("# Citizen Council — Unified Ranked Backlog\n")
    print(f"Total proposals: {len(council._proposals)}")
    print(f"Displayed after deduplication: {len(ranked)}")
    print(f"Unique components: {council.summary()['unique_components']}\n")

    for rp in ranked:
        p = rp.proposal
        print(f"## #{rp.rank} — {p.id}")
        print(f"  Score:      {rp.priority_score:.2f}")
        print(f"  Title:      {p.title}")
        print(f"  Sources:    {', '.join(rp.source_citizens)}")
        print(f"  Confidence: {p.confidence.value} ({p.confidence_score:.0%})")
        print(f"  Effort:     {p.estimated_effort_hours}h")
        print(f"  Components: {', '.join(p.affected_components)}")
        print(f"  Problem:    {p.problem[:120]}...")
        if rp.duplicates:
            print(f"  Duplicates: {', '.join(rp.duplicates)}")
        print()

    return 0


def _cmd_citizen_council_summary(args: argparse.Namespace) -> int:
    from animus.citizens import CitizenCouncil
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    council = CitizenCouncil(memory_layer=memory)
    council.collect_from_memory()
    summary = council.summary()

    print("# Citizen Council Summary\n")
    for key, value in summary.items():
        print(f"  {key:30s}: {value}")
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

    designer_parser = subparsers.add_parser(
        "conversation-designer",
        help="Run the Conversation Designer Citizen observation and analysis cycle",
    )
    designer_parser.add_argument(
        "--log-dir",
        default="",
        help="Directory containing conversation JSONL logs",
    )
    designer_parser.add_argument(
        "--store",
        action="store_true",
        help="Store generated proposal in Animus memory",
    )
    designer_parser.set_defaults(func=_cmd_conversation_designer)

    curator_parser = subparsers.add_parser(
        "knowledge-curator",
        help="Run the Knowledge Curator Citizen observation and analysis cycle",
    )
    curator_parser.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase for cross-reference checks",
    )
    curator_parser.add_argument(
        "--store",
        action="store_true",
        help="Store generated proposal in Animus memory",
    )
    curator_parser.set_defaults(func=_cmd_knowledge_curator)

    oracle_parser = subparsers.add_parser(
        "test-oracle",
        help="Run the Test Oracle Citizen observation and analysis cycle",
    )
    oracle_parser.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase",
    )
    oracle_parser.add_argument(
        "--store",
        action="store_true",
        help="Store generated proposal in Animus memory",
    )
    oracle_parser.set_defaults(func=_cmd_test_oracle)

    # Session
    session_parser = subparsers.add_parser(
        "session",
        help="Start an interactive Head REPL session with lifecycle management",
    )
    session_parser.add_argument(
        "--model",
        default="qwen2.5:32b",
        help="Ollama model to use (default: qwen2.5:32b)",
    )
    session_parser.add_argument(
        "--project",
        default=".",
        help="Project root directory (default: current directory)",
    )
    session_parser.add_argument(
        "--timer",
        default=None,
        help="Session wall-clock limit, e.g. 30m, 1h, 90s (default: disabled)",
    )
    session_parser.add_argument(
        "--wrapup-at",
        type=float,
        default=0.96,
        help="Token utilization fraction (0.0–1.0) that triggers graceful finalize (default: 0.96)",
    )
    session_parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Disable automatic session restart after wrap-up",
    )
    session_parser.set_defaults(func=_cmd_session)

    # ------------------------------------------------------------------
    # Session Steward
    # ------------------------------------------------------------------

    session_steward_parser = subparsers.add_parser(
        "session-steward",
        help="Run Session Steward retrospective audit",
    )
    session_steward_parser.add_argument(
        "--telemetry-file",
        default=None,
        help="Path to JSON file with SessionController telemetry (from /session stats)",
    )
    session_steward_parser.add_argument(
        "--no-store",
        action="store_true",
        help="Skip storing the proposal in memory",
    )
    session_steward_parser.set_defaults(func=_cmd_session_steward)

    # Proposal Queue
    proposal_queue_parser = subparsers.add_parser(
        "proposal-queue",
        help="Manage the proposal approval queue",
    )
    proposal_queue_subparsers = proposal_queue_parser.add_subparsers(dest="pq_command")

    pq_list = proposal_queue_subparsers.add_parser("list", help="List proposals by status")
    pq_list.add_argument(
        "--status",
        default="pending",
        choices=["pending", "approved", "commissioned", "complete", "rejected", "backlog", "all"],
        help="Status filter (default: pending)",
    )
    pq_list.set_defaults(func=_cmd_proposal_queue_list)

    pq_approve = proposal_queue_subparsers.add_parser("approve", help="Approve a proposal")
    pq_approve.add_argument("proposal_id", help="ID of proposal to approve")
    pq_approve.add_argument("--reason", default="", help="Approval rationale")
    pq_approve.set_defaults(func=_cmd_proposal_queue_approve)

    pq_reject = proposal_queue_subparsers.add_parser("reject", help="Reject a proposal")
    pq_reject.add_argument("proposal_id", help="ID of proposal to reject")
    pq_reject.add_argument("--reason", default="", help="Rejection rationale")
    pq_reject.set_defaults(func=_cmd_proposal_queue_reject)

    pq_stats = proposal_queue_subparsers.add_parser("stats", help="Show queue statistics")
    pq_stats.set_defaults(func=_cmd_proposal_queue_stats)

    # Citizen Council
    council_parser = subparsers.add_parser(
        "citizen-council",
        help="Unified backlog from all citizens",
    )
    council_subparsers = council_parser.add_subparsers(dest="cc_command")

    cc_backlog = council_subparsers.add_parser("backlog", help="Show ranked backlog")
    cc_backlog.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        default=True,
        help="Disable component-based deduplication",
    )
    cc_backlog.set_defaults(func=_cmd_citizen_council_backlog)

    cc_summary = council_subparsers.add_parser("summary", help="Show summary statistics")
    cc_summary.set_defaults(func=_cmd_citizen_council_summary)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
