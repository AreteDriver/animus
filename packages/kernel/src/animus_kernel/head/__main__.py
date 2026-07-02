"""Entry point: python -m animus_kernel.head"""

from __future__ import annotations

import argparse
import logging
import sys

from animus_kernel.head.repl import HeadREPL


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
        "-v", "--verbose",
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

    try:
        repl = HeadREPL(
            model=args.model,
            project_root=args.project,
            memory_dir=args.memory_dir,
            system_prompt=system_prompt,
        )
        repl.start()
    except RuntimeError as exc:
        print(f"Failed to start Head: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
