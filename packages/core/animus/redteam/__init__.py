"""Red-team driver — adversarial probing of the hardening gates.

Uses an uncensored local model (default: HauhauCS/Qwen3.6 Uncensored Aggressive
via Ollama) to generate attack inputs against each category of gate,
applies them against isolated test fixtures, classifies outcomes
(blocked / surfaced / partial / no-op), and emits a JSONL report.

Run::

    python -m animus.redteam.driver
    python -m animus.redteam.driver --quick           # CI gate mode
    python -m animus.redteam.driver --category pi     # single-category run
    python -m animus.redteam.driver --model qwen2.5:14b   # override model

This is the one-shot. Standing-capability follow-up (cron + dashboard +
novel-vs-known classification) is separate.
"""

from animus.redteam.driver import (
    CATEGORIES,
    AttackProbe,
    RedTeamDriver,
    RedTeamReport,
    Severity,
)

__all__ = [
    "AttackProbe",
    "CATEGORIES",
    "RedTeamDriver",
    "RedTeamReport",
    "Severity",
]
