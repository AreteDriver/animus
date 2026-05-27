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

logger = logging.getLogger("animus.daemon")


def _verify_integrity_or_exit() -> None:
    """10/10 hardening — refuse to boot if critical-path files drifted.

    Soft-import so bootstrap stays installable without animus-core (the
    standalone deployment path). When core is absent, integrity check
    is skipped with a warning rather than blocking startup.

    Behavior:
      - Baseline present + hashes match → silent.
      - Drift → log + exit non-zero. Operator must investigate, then
        either revert the change or regenerate the baseline.
      - Baseline missing → exit non-zero unless override is set. First
        install must regenerate via
        ``python -m animus.integrity.cli regenerate``.
      - ``ANIMUS_INTEGRITY_OVERRIDE=1`` → log warning, proceed.
    """
    try:
        from animus.integrity import (
            IntegrityMismatchError,
            IntegrityNotInitializedError,
            verify_or_raise,
        )
        from animus.integrity.checker import default_baseline_dir
    except ImportError:
        logger.warning(
            "animus-core not installed; tampering detection skipped. "
            "Install animus-core to enable the integrity gate."
        )
        return

    try:
        verify_or_raise(default_baseline_dir())
    except IntegrityMismatchError as e:
        logger.error("Integrity check FAILED — refusing to start daemon.\n%s", e)
        sys.exit(2)
    except IntegrityNotInitializedError as e:
        logger.error("Integrity baseline missing — refusing to start daemon.\n%s", e)
        sys.exit(3)


def main() -> None:
    """Start the Animus daemon (dashboard + runtime)."""
    _verify_integrity_or_exit()
    from animus_bootstrap.dashboard.app import serve

    serve()


if __name__ == "__main__":
    main()
