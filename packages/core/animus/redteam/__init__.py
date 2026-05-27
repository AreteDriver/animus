"""Red-team driver — adversarial probing of the hardening gates.

Two layers:

- ``animus.redteam.driver`` — the one-shot driver. Generates probes via
  an uncensored local model (default ``qwen2.5:14b``, preferred
  ``hauhaucs`` via llama-server), applies them against isolated test
  fixtures of every gate, emits a JSONL report.
- ``animus.redteam.standing`` — the standing capability. Runs the
  driver on a schedule (systemd timer), appends every probe to a
  persistent ledger, classifies novel-vs-known, renders a tail-friendly
  markdown dashboard.

Run::

    # One-shot
    python -m animus.redteam.driver
    python -m animus.redteam.driver --quick           # CI gate mode
    python -m animus.redteam.driver --category pi     # single-category run
    python -m animus.redteam.driver --model qwen2.5:14b   # override model

    # Standing sweep (writes to ledger + dashboard, exits non-zero on
    # new HIGH findings)
    python -m animus.redteam.standing --n-per-category 5
"""

from animus.redteam.driver import (
    CATEGORIES,
    AttackProbe,
    RedTeamDriver,
    RedTeamReport,
    Severity,
)
from animus.redteam.standing import (
    FindingsLedger,
    LedgerEntry,
    StandingSweepResult,
    attack_hash,
    normalize_attack,
    run_standing_sweep,
)

__all__ = [
    "AttackProbe",
    "CATEGORIES",
    "FindingsLedger",
    "LedgerEntry",
    "RedTeamDriver",
    "RedTeamReport",
    "Severity",
    "StandingSweepResult",
    "attack_hash",
    "normalize_attack",
    "run_standing_sweep",
]
