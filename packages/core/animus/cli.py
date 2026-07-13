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


def _cmd_proposal_queue_show(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    qp = queue.get(args.proposal_id)
    if qp is None:
        print(f"Proposal '{args.proposal_id}' not found.", file=sys.stderr)
        return 1

    p = qp.proposal
    print(f"# {p.id}\n")
    print(f"Title:       {p.title}")
    print(f"Status:      {qp.current_status.value}")
    print(f"Priority:    {qp.priority}")
    print(f"Confidence:  {p.confidence.value} ({p.confidence_score:.0%})")
    print(f"Effort:      {p.estimated_effort_hours}h")
    print(f"Problem:     {p.problem}")
    print(f"Recommendation: {p.recommendation}")
    print(f"Affected:    {', '.join(p.affected_files) if p.affected_files else 'none'}")
    print(f"Tags:        {', '.join(qp.tags) if qp.tags else 'none'}")
    if qp.transitions:
        print("\nTransitions:")
        for t in qp.transitions:
            print(f"  {t.from_status.value} → {t.to_status.value} by {t.actor} ({t.timestamp.isoformat()})")
    if p.evidence:
        print(f"\nEvidence ({len(p.evidence)} items):")
        for i, ev in enumerate(p.evidence, 1):
            print(f"  {i}. [{ev.source}] {ev.description[:100]}")
            if ev.data:
                print(f"     Data: {str(ev.data)[:120]}")
    else:
        print("\nNo evidence attached.")
    return 0


def _cmd_proposal_queue_clear_sources(args: argparse.Namespace) -> int:
    from animus.citizens import ProposalQueue
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    queue = ProposalQueue(memory_layer=memory)
    queue.load_from_memory()

    qp = queue.get(args.proposal_id)
    if qp is None:
        print(f"Proposal '{args.proposal_id}' not found.", file=sys.stderr)
        return 1

    count = len(qp.proposal.evidence)
    qp.proposal.evidence.clear()
    queue.save_to_memory()
    print(f"🧹 Cleared {count} evidence item(s) from {args.proposal_id}.")
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


def _cmd_harvester(args: argparse.Namespace) -> int:
    from animus.citizens import HarvesterCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    store = getattr(args, "store", False)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if store else None
    cb_path = getattr(args, "codebase_path", "") or config.citizens.codebase_path or str(config.data_dir.parent)
    harvester = HarvesterCitizen(
        memory_layer=memory,
        codebase_path=cb_path,
    )

    sub = args.harvester_command

    if sub == "harvest":
        if not args.target:
            print("Provide --target (GitHub repo URL or user/repo).", file=sys.stderr)
            return 1
        source = harvester.harvest_repository(args.target, depth=args.depth)
        if source is None:
            print(f"Harvest failed for {args.target}. Check logs for details.", file=sys.stderr)
            return 1
        print(f"# Harvested Source: {source.title}\n")
        print(f"**Type:** {source.source_type}")
        print(f"**Identifier:** {source.identifier}")
        print(f"**Confidence:** {source.confidence}")
        if source.tags:
            print(f"**Tags:** {', '.join(source.tags)}")
        if source.metadata:
            print(f"**Metadata:**")
            for k, v in source.metadata.items():
                print(f"  - {k}: {v}")
        if store and memory:
            stored = harvester.store_source(source)
            if stored:
                print("\n✅ Source stored in memory.")
        return 0

    if sub == "watchlist":
        report = harvester.harvest_watchlist(interval_hours=args.interval_hours)
        print(f"# Harvester Watchlist Report\n")
        print(f"**Sources collected:** {report.total_collected}")
        print(f"**Duplicates removed:** {report.duplicates_removed}")
        if report.errors:
            print(f"**Errors:** {len(report.errors)}")
            for err in report.errors:
                print(f"  - {err}")
        for source in report.sources:
            print(f"\n- [{source.source_type}] {source.title} ({source.identifier})")
        if store and memory:
            stored = harvester.store_report(report)
            if stored:
                print("\n✅ Report stored in memory.")
        return 0

    if sub == "sources":
        sources = harvester.list_stored_sources(limit=args.limit)
        print(f"# Stored Harvested Sources ({len(sources)} found)\n")
        if not sources:
            print("No harvested sources found in memory.")
            return 0
        for s in sources:
            meta = s.get("metadata", {})
            title = meta.get("title", "Untitled")
            source_type = meta.get("source_type", "unknown")
            print(f"- [{source_type}] {title}")
        return 0

    if sub == "analyze":
        # Run full observation sweep and generate proposal
        obs = harvester.observe_codebase()
        memory_sources = harvester.observe_memory()
        all_sources = []
        for o in obs:
            all_sources.append(
                harvester.harvest_text(
                    text=o["description"],
                    source_type="code_snippet",
                    identifier=o["context"].get("file", "unknown"),
                )
            )
        all_sources.extend(memory_sources)
        all_sources = harvester.deduplicate(all_sources)

        print(f"# Harvester Observation Sweep\n")
        print(f"**Codebase findings:** {len(obs)}")
        print(f"**Memory sources:** {len(memory_sources)}")
        print(f"**Unique sources:** {len(all_sources)}\n")

        proposal = harvester.generate_proposal(all_sources)
        if proposal:
            print(f"## Proposal Generated: {proposal.title}")
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
            if store and memory:
                stored = harvester.store_proposal(proposal)
                if stored:
                    print(f"\n✅ Proposal stored in memory.")
        else:
            print("## No Proposal Generated")
            print("No actionable findings from harvest sweep.")
        return 0

    print(f"Unknown harvester subcommand: {sub}", file=sys.stderr)
    return 1


def _cmd_abstraction(args: argparse.Namespace) -> int:
    from animus.citizens import AbstractionCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    store = getattr(args, "store", False)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if store else None
    abstraction = AbstractionCitizen(
        memory_layer=memory,
        codebase_path=args.codebase_path or config.citizens.codebase_path or str(config.data_dir.parent),
    )

    sub = args.abstraction_command

    if sub == "scan":
        print("# Running Abstraction Citizen scan...", file=sys.stderr)

        # Codebase observations
        obs = abstraction.observe_codebase()
        if obs:
            print(f"\n## Codebase Mechanisms ({len(obs)} found)", file=sys.stderr)
            for o in obs:
                print(f"- [{o['severity'].upper()}] {o['description']}", file=sys.stderr)

        # Harvested sources
        sources = abstraction.observe_harvested_sources()
        if sources:
            print(f"\n## Harvested Sources ({len(sources)} found)", file=sys.stderr)
            for s in sources:
                print(f"- [{s['severity'].upper()}] {s['description']}", file=sys.stderr)

        # Extract mechanisms from all sources
        mechanisms: list = []
        for s in sources:
            content = s["context"].get("content", "")
            sid = s["context"].get("identifier", "")
            if content:
                mechs = abstraction.extract_mechanisms(content, sid)
                mechanisms.extend(mechs)

        print(f"\n# Extracted Mechanisms ({len(mechanisms)} total)")
        for m in mechanisms:
            print(f"\n## {m.name} ({m.category})")
            print(f"**Description:** {m.description}")
            if m.source_provenance:
                print(f"**Sources:** {', '.join(m.source_provenance)}")
            print(f"**Confidence:** {m.confidence}")
            if store and memory:
                stored = abstraction.store_mechanism(m)
                if stored:
                    print(f"✅ Stored mechanism '{m.name}'")

        # Generate proposal
        proposal = abstraction.generate_proposal(mechanisms)
        if proposal:
            print(f"\n## Proposal Generated: {proposal.title}")
            print(f"**ID:** {proposal.id}")
            print(f"**Problem:** {proposal.problem}")
            print(f"**Confidence:** {proposal.confidence.value} ({proposal.confidence_score:.0%})")
            print(f"**Recommendation:** {proposal.recommendation}")
            print(f"**Effort:** {proposal.estimated_effort_hours}h")
            if store and memory:
                stored = abstraction.store_proposal(proposal)
                if stored:
                    print("\n✅ Proposal stored in memory.")
        else:
            print("\n## No Proposal Generated")
            print("No mechanisms extracted — nothing to propose.")
        return 0

    if sub == "mechanisms":
        mechs = abstraction.list_stored_mechanisms(limit=args.limit)
        print(f"# Stored Mechanisms ({len(mechs)} found)\n")
        if not mechs:
            print("No mechanisms found in memory.")
            return 0
        for m in mechs:
            meta = m.get("metadata", {})
            name = meta.get("name", "Untitled")
            category = meta.get("category", "unknown")
            print(f"- [{category}] {name}")
        return 0

    print(f"Unknown abstraction subcommand: {sub}", file=sys.stderr)
    return 1


def _cmd_pattern(args: argparse.Namespace) -> int:
    from animus.citizens import PatternCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    store = getattr(args, "store", False)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if store else None
    pattern = PatternCitizen(
        memory_layer=memory,
        codebase_path=getattr(args, "codebase_path", "") or config.citizens.codebase_path or str(config.data_dir.parent),
    )

    sub = args.pattern_command

    if sub == "scan":
        print("# Running Pattern Citizen scan...", file=sys.stderr)

        # Observe mechanisms from memory
        mechanisms = pattern.observe_mechanisms()
        if mechanisms:
            print(f"\n## Mechanisms Observed ({len(mechanisms)} found)", file=sys.stderr)
            for m in mechanisms:
                print(f"- [{m['severity'].upper()}] {m['description']}", file=sys.stderr)
        else:
            print("\n## No mechanisms found in memory.", file=sys.stderr)

        # Discover patterns
        mech_contexts = [m["context"] for m in mechanisms]
        patterns = pattern.discover_patterns(mech_contexts)
        print(f"\n# Discovered Patterns ({len(patterns)} total)")
        for p in patterns:
            print(f"\n## {p.name} ({p.category})")
            print(f"**Description:** {p.description}")
            print(f"**Mechanisms:** {', '.join(p.constituent_mechanisms)}")
            print(f"**Occurrences:** {p.occurrence_count}")
            print(f"**Confidence:** {p.confidence}")
            if store and memory:
                stored = pattern.store_pattern(p)
                if stored:
                    print(f"✅ Stored pattern '{p.name}'")

        # Generate proposal
        proposal = pattern.generate_proposal(patterns)
        if proposal:
            print(f"\n## Proposal Generated: {proposal.title}")
            print(f"**ID:** {proposal.id}")
            print(f"**Problem:** {proposal.problem}")
            print(f"**Confidence:** {proposal.confidence.value} ({proposal.confidence_score:.0%})")
            print(f"**Recommendation:** {proposal.recommendation}")
            print(f"**Effort:** {proposal.estimated_effort_hours}h")
            if store and memory:
                stored = pattern.store_proposal(proposal)
                if stored:
                    print("\n✅ Proposal stored in memory.")
        else:
            print("\n## No Proposal Generated")
            print("No patterns discovered — nothing to propose.")
        return 0

    if sub == "patterns":
        patterns = pattern.list_stored_patterns(limit=args.limit)
        print(f"# Stored Patterns ({len(patterns)} found)\n")
        if not patterns:
            print("No patterns found in memory.")
            return 0
        for p in patterns:
            meta = p.get("metadata", {})
            name = meta.get("name", "Untitled")
            category = meta.get("category", "unknown")
            mechanisms = meta.get("constituent_mechanisms", [])
            print(f"- [{category}] {name} ({len(mechanisms)} mechanisms)")
        return 0

    print(f"Unknown pattern subcommand: {sub}", file=sys.stderr)
    return 1


def _cmd_first_principles(args: argparse.Namespace) -> int:
    from animus.citizens import FirstPrinciplesCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    if not config.citizens.enabled:
        print("Citizens are disabled in configuration.", file=sys.stderr)
        return 1

    store = getattr(args, "store", False)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if store else None
    fp = FirstPrinciplesCitizen(
        memory_layer=memory,
        codebase_path=getattr(args, "codebase_path", "") or config.citizens.codebase_path or str(config.data_dir.parent),
    )

    sub = args.first_principles_command

    if sub == "scan":
        print("# Running First-Principles Citizen scan...", file=sys.stderr)

        # Observe patterns from memory
        patterns = fp.observe_patterns()
        if patterns:
            print(f"\n## Patterns Observed ({len(patterns)} found)", file=sys.stderr)
            for p in patterns:
                print(f"- [{p['severity'].upper()}] {p['description']}", file=sys.stderr)
        else:
            print("\n## No patterns found in memory.", file=sys.stderr)

        # Reduce to principles
        pattern_contexts = [p["context"] for p in patterns]
        principles = fp.reduce_to_principles(pattern_contexts)
        print(f"\n# Reduced Principles ({len(principles)} total)")
        for pr in principles:
            print(f"\n## Principle ({pr.category})")
            print(f"**Statement:** {pr.principle_statement}")
            print(f"**Supporting Patterns:** {', '.join(pr.supporting_patterns)}")
            print(f"**Confidence:** {pr.confidence}")
            if pr.contradictions:
                print(f"**Contradictions Flagged:** {len(pr.contradictions)}")
            if store and memory:
                stored = fp.store_principle(pr)
                if stored:
                    print(f"✅ Stored principle")

        # Generate proposal
        proposal = fp.generate_proposal(principles)
        if proposal:
            print(f"\n## Proposal Generated: {proposal.title}")
            print(f"**ID:** {proposal.id}")
            print(f"**Problem:** {proposal.problem}")
            print(f"**Confidence:** {proposal.confidence.value} ({proposal.confidence_score:.0%})")
            print(f"**Recommendation:** {proposal.recommendation}")
            print(f"**Effort:** {proposal.estimated_effort_hours}h")
            if store and memory:
                stored = fp.store_proposal(proposal)
                if stored:
                    print("\n✅ Proposal stored in memory.")
        else:
            print("\n## No Proposal Generated")
            print("No principles reduced — nothing to propose.")
        return 0

    if sub == "principles":
        principles = fp.list_stored_principles(limit=args.limit)
        print(f"# Stored Principles ({len(principles)} found)\n")
        if not principles:
            print("No principles found in memory.")
            return 0
        for p in principles:
            meta = p.get("metadata", {})
            statement = meta.get("principle_statement", "Untitled")
            category = meta.get("category", "unknown")
            print(f"- [{category}] {statement}")
        return 0

    print(f"Unknown first-principles subcommand: {sub}", file=sys.stderr)
    return 1


def _cmd_intelligence(args: argparse.Namespace) -> int:
    from animus.citizens import IntelligenceCitizen
    from animus.config import get_config
    from animus.memory import MemoryLayer

    config = get_config()
    store = getattr(args, "store", False)
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend) if store else None
    intel = IntelligenceCitizen(memory_layer=memory)

    sub = args.intel_command

    if sub == "extract":
        text = args.text or ""
        if not text and args.file:
            from pathlib import Path

            path = Path(args.file)
            text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        if not text:
            print("Provide --text or --file.", file=sys.stderr)
            return 1

        entities = intel.extract_entities(text)
        data = entities.to_dict()
        print(f"# Extracted Entities ({entities.total_count()} total)\n")
        for category, items in data.items():
            if items:
                print(f"## {category.replace('_', ' ').title()} ({len(items)})")
                for item in items:
                    print(f"- {item}")
                print()
        return 0

    if sub == "secrets":
        if args.file:
            from pathlib import Path

            path = Path(args.file)
            if not path.exists():
                print(f"File not found: {args.file}", file=sys.stderr)
                return 1
            findings = intel.scan_file_secrets(path)
        elif args.text:
            findings = intel.scan_secrets(args.text)
        else:
            print("Provide --text or --file.", file=sys.stderr)
            return 1

        if not findings:
            print("No secrets detected.")
            return 0

        critical = [f for f in findings if f.severity == "critical"]
        high = [f for f in findings if f.severity == "high"]
        print(f"# Secret Findings: {len(findings)} total ({len(critical)} critical, {len(high)} high)\n")
        for f in findings:
            loc = f" (line {f.line_number})" if f.line_number else ""
            print(f"[{f.severity.upper()}] {f.description}{loc}")
            print(f"  pattern={f.pattern_name} match={f.matched_text}")
        return 0

    if sub == "osint":
        if not args.username:
            print("Provide --username.", file=sys.stderr)
            return 1

        profiles = intel.generate_profile_urls(args.username)
        print(f"# OSINT Profiles for @{args.username}\n")
        if profiles:
            for p in profiles:
                print(f"- {p.platform}: {p.url} ({p.category})")
        else:
            print("No valid profile URLs generated.")
        return 0

    if sub == "analyze":
        text = args.text or ""
        if args.file:
            from pathlib import Path

            path = Path(args.file)
            if not path.exists():
                print(f"File not found: {args.file}", file=sys.stderr)
                return 1
            text = path.read_text(encoding="utf-8", errors="ignore")
        if not text:
            print("Provide --text or --file.", file=sys.stderr)
            return 1

        report = intel.analyze(text=text)
        print(f"# Intelligence Report: {report.source}\n")
        data = report.extracted.to_dict()
        total = report.extracted.total_count()
        print(f"## Entities ({total} total)")
        for category, items in data.items():
            if items:
                print(f"- {category}: {len(items)}")
        print()

        if report.secrets:
            crit = len([s for s in report.secrets if s.severity == "critical"])
            print(f"## Secrets ({len(report.secrets)} total, {crit} critical)")
            for s in report.secrets[:10]:
                print(f"- [{s.severity}] {s.description}")
            print()
        else:
            print("## Secrets")
            print("None found.\n")

        if report.profiles:
            print(f"## OSINT Profiles ({len(report.profiles)} generated)")
            for p in report.profiles[:10]:
                print(f"- {p.platform}: {p.url}")
            print()

        proposal = intel.generate_proposal(report)
        if proposal:
            print(f"## Proposal: {proposal.title}")
            print(f"ID: {proposal.id}")
            print(f"Problem: {proposal.problem}")
            print(f"Recommendation: {proposal.recommendation}")
            print(f"Effort: {proposal.estimated_effort_hours}h")
            if store and memory:
                stored = intel.store_report(report)
                if stored:
                    print("\n✅ Report stored in memory.")
        else:
            print("## No Proposal")
            print("No critical security findings — no proposal generated.")
        return 0

    print(f"Unknown intelligence subcommand: {sub}", file=sys.stderr)
    return 1


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

    # ------------------------------------------------------------------
    # Abstraction Citizen (Research Guild)
    # ------------------------------------------------------------------

    abstraction_parser = subparsers.add_parser(
        "abstraction",
        help="Run the Research Guild Abstraction Citizen — extract mechanisms",
    )
    abstraction_subparsers = abstraction_parser.add_subparsers(dest="abstraction_command")

    abs_scan = abstraction_subparsers.add_parser("scan", help="Scan codebase and memory for mechanisms")
    abs_scan.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase",
    )
    abs_scan.add_argument(
        "--store",
        action="store_true",
        help="Store extracted mechanisms and proposal in memory",
    )
    abs_scan.set_defaults(func=_cmd_abstraction)

    abs_mechanisms = abstraction_subparsers.add_parser("mechanisms", help="List stored mechanism cards")
    abs_mechanisms.add_argument("--limit", type=int, default=20, help="Max mechanisms to list")
    abs_mechanisms.set_defaults(func=_cmd_abstraction)

    # ------------------------------------------------------------------
    # Pattern Citizen (Research Guild)
    # ------------------------------------------------------------------

    pattern_parser = subparsers.add_parser(
        "pattern",
        help="Run the Research Guild Pattern Citizen — discover recurring patterns",
    )
    pattern_subparsers = pattern_parser.add_subparsers(dest="pattern_command")

    ptn_scan = pattern_subparsers.add_parser("scan", help="Scan memory for mechanisms and discover patterns")
    ptn_scan.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase",
    )
    ptn_scan.add_argument(
        "--store",
        action="store_true",
        help="Store discovered patterns and proposal in memory",
    )
    ptn_scan.set_defaults(func=_cmd_pattern)

    ptn_patterns = pattern_subparsers.add_parser("patterns", help="List stored pattern cards")
    ptn_patterns.add_argument("--limit", type=int, default=20, help="Max patterns to list")
    ptn_patterns.set_defaults(func=_cmd_pattern)

    # ------------------------------------------------------------------
    # First-Principles Citizen (Research Guild)
    # ------------------------------------------------------------------

    fp_parser = subparsers.add_parser(
        "first-principles",
        help="Run the Research Guild First-Principles Citizen — reduce patterns to truths",
    )
    fp_subparsers = fp_parser.add_subparsers(dest="first_principles_command")

    fp_scan = fp_subparsers.add_parser("scan", help="Scan memory for patterns and reduce to principles")
    fp_scan.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase",
    )
    fp_scan.add_argument(
        "--store",
        action="store_true",
        help="Store reduced principles and proposal in memory",
    )
    fp_scan.set_defaults(func=_cmd_first_principles)

    fp_principles = fp_subparsers.add_parser("principles", help="List stored principle cards")
    fp_principles.add_argument("--limit", type=int, default=20, help="Max principles to list")
    fp_principles.set_defaults(func=_cmd_first_principles)

    # ------------------------------------------------------------------
    # Harvester (Research Guild)
    # ------------------------------------------------------------------

    harvester_parser = subparsers.add_parser(
        "harvester",
        help="Run the Research Guild Harvester Citizen — collect raw sources",
    )
    harvester_subparsers = harvester_parser.add_subparsers(dest="harvester_command")

    hv_harvest = harvester_subparsers.add_parser("harvest", help="Harvest a GitHub repository")
    hv_harvest.add_argument("--target", required=True, help="GitHub repo URL or user/repo")
    hv_harvest.add_argument("--depth", default="quick", choices=["quick", "deep"], help="Scan depth")
    hv_harvest.add_argument(
        "--store",
        action="store_true",
        help="Store harvested source in Animus memory",
    )
    hv_harvest.set_defaults(func=_cmd_harvester)

    hv_watchlist = harvester_subparsers.add_parser("watchlist", help="Harvest all due watchlist repos")
    hv_watchlist.add_argument(
        "--interval-hours",
        type=int,
        default=0,
        help="Override scan interval (0 = default 168h)",
    )
    hv_watchlist.add_argument(
        "--store",
        action="store_true",
        help="Store report in Animus memory",
    )
    hv_watchlist.set_defaults(func=_cmd_harvester)

    hv_sources = harvester_subparsers.add_parser("sources", help="List stored harvested sources")
    hv_sources.add_argument("--limit", type=int, default=20, help="Max sources to list")
    hv_sources.set_defaults(func=_cmd_harvester)

    hv_analyze = harvester_subparsers.add_parser("analyze", help="Run observation sweep and generate proposal")
    hv_analyze.add_argument(
        "--codebase-path",
        default="",
        help="Path to the codebase",
    )
    hv_analyze.add_argument(
        "--store",
        action="store_true",
        help="Store generated proposal in Animus memory",
    )
    hv_analyze.set_defaults(func=_cmd_harvester)

    # ------------------------------------------------------------------
    # Intelligence Officer
    # ------------------------------------------------------------------

    intel_parser = subparsers.add_parser(
        "intelligence",
        help="Run the Intelligence Officer Citizen — extraction, secrets, OSINT",
    )
    intel_subparsers = intel_parser.add_subparsers(dest="intel_command")

    intel_extract = intel_subparsers.add_parser("extract", help="Extract entities from text")
    intel_extract.add_argument("--text", default="", help="Text to analyze")
    intel_extract.add_argument("--file", default="", help="File to read and analyze")
    intel_extract.set_defaults(func=_cmd_intelligence)

    intel_secrets = intel_subparsers.add_parser("secrets", help="Scan for secrets and credentials")
    intel_secrets.add_argument("--text", default="", help="Text to scan")
    intel_secrets.add_argument("--file", default="", help="File to scan")
    intel_secrets.set_defaults(func=_cmd_intelligence)

    intel_osint = intel_subparsers.add_parser("osint", help="Generate OSINT profile URLs")
    intel_osint.add_argument("--username", required=True, help="Username to look up")
    intel_osint.set_defaults(func=_cmd_intelligence)

    intel_analyze = intel_subparsers.add_parser("analyze", help="Comprehensive intelligence report")
    intel_analyze.add_argument("--text", default="", help="Text to analyze")
    intel_analyze.add_argument("--file", default="", help="File to analyze")
    intel_analyze.add_argument(
        "--store",
        action="store_true",
        help="Store report in Animus memory",
    )
    intel_analyze.set_defaults(func=_cmd_intelligence)

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

    pq_show = proposal_queue_subparsers.add_parser("show", help="Show proposal details with evidence")
    pq_show.add_argument("proposal_id", help="ID of proposal to show")
    pq_show.set_defaults(func=_cmd_proposal_queue_show)

    pq_clear = proposal_queue_subparsers.add_parser("clear-sources", help="Clear evidence from a proposal")
    pq_clear.add_argument("proposal_id", help="ID of proposal to clear")
    pq_clear.set_defaults(func=_cmd_proposal_queue_clear_sources)

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
