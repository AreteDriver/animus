"""CLI entry point for ``python -m convergent``.

Subcommands:
    inspect <db_path>  — Inspect a SQLite intent graph
    health <db_path>   — Show coordination health report
    events <db_path>   — Show coordination event timeline
    cycles <db_path>   — Detect dependency cycles in the intent graph
    demo               — Run the interactive demo
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convergent",
        description="Convergent — multi-agent coherence through ambient intent awareness",
    )
    sub = parser.add_subparsers(dest="command")

    # --- inspect ---
    inspect_p = sub.add_parser("inspect", help="Inspect a SQLite intent graph")
    inspect_p.add_argument("db_path", help="Path to SQLite database file")
    inspect_p.add_argument(
        "--format",
        choices=["table", "dot", "html", "matrix"],
        default="table",
        dest="fmt",
        help="Output format (default: table)",
    )
    inspect_p.add_argument(
        "--min-stability",
        type=float,
        default=0.0,
        help="Filter intents below this stability threshold",
    )
    inspect_p.add_argument("--agent", help="Filter to a single agent")
    inspect_p.add_argument(
        "--show-evidence",
        action="store_true",
        help="Include evidence in table output",
    )
    inspect_p.add_argument("--output", help="Write output to file instead of stdout")

    # --- health ---
    health_p = sub.add_parser("health", help="Show coordination health report")
    health_p.add_argument("db_path", help="Path to coordination database")

    # --- events ---
    events_p = sub.add_parser("events", help="Show coordination event timeline")
    events_p.add_argument("db_path", help="Path to event log database")
    events_p.add_argument("--type", dest="event_type", help="Filter by event type")
    events_p.add_argument("--agent", help="Filter by agent ID")
    events_p.add_argument("--limit", type=int, default=50, help="Max events (default: 50)")

    # --- cycles ---
    cycles_p = sub.add_parser("cycles", help="Detect dependency cycles in intent graph")
    cycles_p.add_argument("db_path", help="Path to SQLite intent graph database")

    # --- demo ---
    sub.add_parser("demo", help="Run the interactive demo")

    return parser


def _cmd_inspect(args: argparse.Namespace) -> None:
    import os

    from convergent.resolver import IntentResolver
    from convergent.sqlite_backend import SQLiteBackend
    from convergent.visualization import dot_graph, html_report, overlap_matrix, text_table

    if not os.path.exists(args.db_path):
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        backend = SQLiteBackend(args.db_path)
    except Exception as exc:
        print(f"Error: cannot open database: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        # Build a filtered backend view based on --agent and --min-stability
        from convergent.resolver import PythonGraphBackend

        if args.agent:
            source_intents = backend.query_by_agent(args.agent)
            if not source_intents:
                print(f"No intents found for agent: {args.agent}", file=sys.stderr)
                sys.exit(1)
        else:
            source_intents = backend.query_all(min_stability=args.min_stability)

        filtered = PythonGraphBackend()
        for intent in source_intents:
            filtered.publish(intent)
        resolver = IntentResolver(backend=filtered, min_stability=args.min_stability)

        if args.fmt == "table":
            output = text_table(resolver, show_evidence=args.show_evidence)
        elif args.fmt == "dot":
            output = dot_graph(resolver, min_stability=args.min_stability)
        elif args.fmt == "html":
            output = html_report(resolver)
        elif args.fmt == "matrix":
            output = overlap_matrix(resolver)
        else:
            output = text_table(resolver)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
                f.write("\n")
        else:
            print(output)
    finally:
        backend.close()


def _cmd_health(args: argparse.Namespace) -> None:
    import os

    from convergent.health import HealthChecker, health_report
    from convergent.score_store import ScoreStore
    from convergent.stigmergy import StigmergyField

    if not os.path.exists(args.db_path):
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    store = ScoreStore(args.db_path)
    stigmergy_db = args.db_path.replace(".db", ".stigmergy.db")
    stigmergy = StigmergyField(stigmergy_db) if os.path.exists(stigmergy_db) else None

    try:
        checker = HealthChecker(store=store, stigmergy=stigmergy)
        health = checker.check()
        print(health_report(health))
    finally:
        store.close()
        if stigmergy is not None:
            stigmergy.close()


def _cmd_events(args: argparse.Namespace) -> None:
    import os

    from convergent.event_log import EventLog, EventType, event_timeline

    if not os.path.exists(args.db_path):
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    event_type = None
    if args.event_type:
        try:
            event_type = EventType(args.event_type)
        except ValueError:
            valid = ", ".join(t.value for t in EventType)
            print(f"Error: invalid event type '{args.event_type}'. Valid: {valid}", file=sys.stderr)
            sys.exit(1)

    log = EventLog(args.db_path)
    try:
        events = log.query(event_type=event_type, agent_id=args.agent, limit=args.limit)
        print(event_timeline(events))
    finally:
        log.close()


def _cmd_cycles(args: argparse.Namespace) -> None:
    import os

    from convergent.cycles import find_cycles
    from convergent.resolver import IntentResolver, PythonGraphBackend
    from convergent.sqlite_backend import SQLiteBackend

    if not os.path.exists(args.db_path):
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        backend = SQLiteBackend(args.db_path)
    except Exception as exc:
        print(f"Error: cannot open database: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        intents = backend.query_all()
        filtered = PythonGraphBackend()
        for intent in intents:
            filtered.publish(intent)
        resolver = IntentResolver(backend=filtered)

        cycles = find_cycles(resolver)
        if not cycles:
            print("No dependency cycles found.")
        else:
            print(f"Found {len(cycles)} dependency cycle(s):\n")
            for i, cycle in enumerate(cycles, 1):
                print(f"  {i}. {cycle}")
    finally:
        backend.close()


def _cmd_demo() -> None:
    from convergent.demo import run_demo

    run_demo()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "health":
        _cmd_health(args)
    elif args.command == "events":
        _cmd_events(args)
    elif args.command == "cycles":
        _cmd_cycles(args)
    elif args.command == "demo":
        _cmd_demo()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
