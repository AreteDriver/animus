"""Entry point for ``python -m animus_bootstrap.daemon``.

Launches the Animus dashboard server which boots the full runtime
(identity, memory, tools, proactive engine, gateway, self-improvement).
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)


def main() -> None:
    """Start the Animus daemon (dashboard + runtime)."""
    from animus_bootstrap.dashboard.app import serve

    serve()


if __name__ == "__main__":
    main()
