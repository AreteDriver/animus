"""CostEnforcer — enforce budget ceilings and emit spend telemetry.

Tracks cumulative spend per mission (and globally) and blocks new tasks when
budgets are exhausted.  Spend data is fed by the scheduler after each task
completes; in a production system this would integrate with an LLM API cost
stream.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    task_id TEXT,
    operation TEXT NOT NULL,  -- e.g. 'llm_call', 'sandbox', 'file_sync'
    provider TEXT,             -- e.g. 'openai', 'anthropic', 'local'
    model TEXT,
    usage_tokens_input INTEGER DEFAULT 0,
    usage_tokens_output INTEGER DEFAULT 0,
    cost_usd TEXT NOT NULL DEFAULT '0.00',
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cost_mission ON cost_events(mission_id);
CREATE INDEX IF NOT EXISTS idx_cost_recorded ON cost_events(recorded_at);
"""

# Rough token pricing (per 1M tokens) — can be overridden via config
_DEFAULT_RATES: dict[str, dict[str, Decimal]] = {
    "openai": {
        "gpt-4o": Decimal("5.00"),
        "gpt-4o-mini": Decimal("0.15"),
    },
    "anthropic": {
        "claude-3-5-sonnet": Decimal("3.00"),
        "claude-3-haiku": Decimal("0.25"),
    },
    "local": {
        "default": Decimal("0.00"),
    },
}


class CostEnforcer:
    """Enforces mission-level and global spend limits.

    Args:
        backend: Shared ``DatabaseBackend``.
        default_mission_cap_usd: Default maximum USD per mission.
        global_cap_usd: Maximum USD across *all* missions in a window.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        *,
        default_mission_cap_usd: Decimal = Decimal("10.00"),
        global_cap_usd: Decimal = Decimal("100.00"),
    ):
        self._backend = backend
        self.default_mission_cap = default_mission_cap_usd
        self.global_cap = global_cap_usd
        self._rates = _DEFAULT_RATES.copy()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._backend.transaction():
            self._backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        mission_id: str,
        operation: str,
        *,
        task_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: Decimal | None = None,
    ) -> None:
        """Record a cost event.

        If ``cost_usd`` is supplied, use it directly; otherwise estimate
        from token counts and the rate card.
        """
        if cost_usd is None:
            cost_usd = self.estimate_cost(
                provider or "local",
                model or "default",
                tokens_input,
                tokens_output,
            )

        with self._backend.transaction():
            self._backend.execute(
                """
                INSERT INTO cost_events
                    (mission_id, task_id, operation, provider, model,
                     usage_tokens_input, usage_tokens_output, cost_usd, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mission_id,
                    task_id,
                    operation,
                    provider,
                    model,
                    tokens_input,
                    tokens_output,
                    str(cost_usd.quantize(Decimal("0.0001"))),
                    datetime.now().isoformat(),
                ),
            )

    def estimate_cost(
        self,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
    ) -> Decimal:
        """Estimate cost from rate card.

        Rates are stored as cost per *million* tokens.
        """
        rate = self._rates.get(provider, {}).get(model, Decimal("0.00"))
        total_tokens = tokens_input + tokens_output
        return rate * Decimal(total_tokens) / Decimal("1_000_000")

    def set_rate(self, provider: str, model: str, per_1m_tokens_usd: Decimal) -> None:
        """Override or add a rate card entry."""
        if provider not in self._rates:
            self._rates[provider] = {}
        self._rates[provider][model] = per_1m_tokens_usd
        logger.info("Rate set: %s/%s = %s per 1M tokens", provider, model, per_1m_tokens_usd)

    def mission_spend(self, mission_id: str) -> Decimal:
        """Return total spend (USD) for a mission."""
        row = self._backend.fetchone(
            "SELECT SUM(cost_usd) AS total FROM cost_events WHERE mission_id = ?",
            (mission_id,),
        )
        if row and row.get("total"):
            return Decimal(str(row["total"]))
        return Decimal("0.00")

    def global_spend(self, since: datetime | None = None) -> Decimal:
        """Return total spend across all missions."""
        if since:
            row = self._backend.fetchone(
                "SELECT SUM(cost_usd) AS total FROM cost_events WHERE recorded_at >= ?",
                (since.isoformat(),),
            )
        else:
            row = self._backend.fetchone(
                "SELECT SUM(cost_usd) AS total FROM cost_events",
            )
        if row and row.get("total"):
            return Decimal(str(row["total"]))
        return Decimal("0.00")

    def mission_remaining(self, mission_id: str, cap: Decimal | None = None) -> Decimal:
        """Remaining budget for a mission."""
        cap = cap or self.default_mission_cap
        return max(Decimal("0.00"), cap - self.mission_spend(mission_id))

    def can_start_task(
        self,
        mission_id: str,
        estimated_cost: Decimal = Decimal("0.10"),
        *,
        mission_cap: Decimal | None = None,
    ) -> tuple[bool, str]:
        """Check whether a new task can be started under current budgets.

        Returns:
            ``(ok, reason)`` tuple.
        """
        mission_cap = mission_cap or self.default_mission_cap

        if self.mission_remaining(mission_id, mission_cap) < estimated_cost:
            return (
                False,
                f"Mission {mission_id} budget exhausted "
                f"({self.mission_spend(mission_id)} / {mission_cap} USD)",
            )

        if self.global_spend() >= self.global_cap:
            return (
                False,
                f"Global budget cap reached ({self.global_spend()} / {self.global_cap} USD)",
            )

        return True, "ok"

    def spend_report(self, mission_id: str | None = None) -> dict[str, Any]:
        """Return a human-readable spend report."""
        if mission_id:
            rows = self._backend.fetchall(
                """
                SELECT operation, SUM(cost_usd) AS total,
                       SUM(usage_tokens_input) AS tokens_in,
                       SUM(usage_tokens_output) AS tokens_out
                FROM cost_events WHERE mission_id = ?
                GROUP BY operation
                """,
                (mission_id,),
            )
            return {
                "mission_id": mission_id,
                "total_spend_usd": str(self.mission_spend(mission_id)),
                "by_operation": [
                    {
                        "operation": r["operation"],
                        "spend_usd": str(Decimal(str(r["total"]))),
                        "tokens_in": r["tokens_in"],
                        "tokens_out": r["tokens_out"],
                    }
                    for r in rows
                ],
            }

        # Global report
        rows = self._backend.fetchall(
            """
            SELECT mission_id, SUM(cost_usd) AS total FROM cost_events
            GROUP BY mission_id
            ORDER BY total DESC
            """
        )
        return {
            "global_spend_usd": str(self.global_spend()),
            "global_cap_usd": str(self.global_cap),
            "by_mission": [
                {"mission_id": r["mission_id"], "spend_usd": str(Decimal(str(r["total"])))}
                for r in rows
            ],
        }
