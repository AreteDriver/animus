"""Entry point: python -m animus_kernel.head"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

from animus_kernel.head.repl import HeadREPL


def _parse_timer(value: str | None) -> timedelta | None:
    """Parse a timer string like '30m', '1h', '90s' into a timedelta."""
    if value is None:
        return None
    value = value.strip().lower()
    if value.endswith("h"):
        return timedelta(hours=int(value[:-1]))
    if value.endswith("m"):
        return timedelta(minutes=int(value[:-1]))
    if value.endswith("s"):
        return timedelta(seconds=int(value[:-1]))
    # Default to minutes if no suffix
    return timedelta(minutes=int(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animus Head — local-first agentic REPL",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:32b",
        help="Ollama model to use (default: qwen2.5:32b)",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Memory store directory (default: ~/.animus/memory)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Path to custom system prompt file",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=None,
        help="Checkpoint SQLite path (default: ~/.animus/sessions/head.db)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as JSON-RPC daemon (stdio) instead of interactive REPL",
    )
    parser.add_argument(
        "--session-timer",
        default=None,
        help="Session wall-clock limit, e.g. 30m, 1h, 90s (default: disabled)",
    )
    parser.add_argument(
        "--wrapup-at",
        type=float,
        default=1.0,
        help="Token utilization fraction (0.0–1.0) that triggers graceful finalize (default: 1.0 = disabled)",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Disable automatic session restart after wrap-up",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.daemon:
        from animus_kernel.head.daemon import HeadDaemon

        daemon = HeadDaemon(model=args.model)
        daemon.run()
        return

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    system_prompt = None
    if args.system_prompt:
        import pathlib

        system_prompt = pathlib.Path(args.system_prompt).read_text()

    session_timer = _parse_timer(args.session_timer)
    wrapup_threshold = args.wrapup_at if args.wrapup_at < 1.0 else 1.0

    try:
        repl = HeadREPL(
            model=args.model,
            project_root=args.project,
            memory_dir=args.memory_dir,
            system_prompt=system_prompt,
            session_timer=session_timer,
            wrapup_threshold=wrapup_threshold,
        )
        # Override auto_restart if --no-auto-restart was passed
        if args.no_auto_restart and repl._session_controller:
            repl._session_controller.policy.auto_restart = False
        repl.start()
    except RuntimeError as exc:
        print(f"Failed to start Head: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
