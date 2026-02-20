"""Persistent Budget Manager.

Handles CRUD operations for budget entities stored in the database.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from animus_forge.state.backends import DatabaseBackend

from .models import (
    Budget,
    BudgetCreate,
    BudgetPeriod,
    BudgetSummary,
    BudgetUpdate,
)

logger = logging.getLogger(__name__)


class PersistentBudgetManager:
    """Manager for persistent budget storage and CRUD operations."""

    def __init__(self, backend: DatabaseBackend):
        """Initialize the budget manager.

        Args:
            backend: Database backend for persistence
        """
        self.backend = backend

    # =========================================================================
    # Budget CRUD
    # =========================================================================

    def list_budgets(
        self,
        agent_id: str | None = None,
        period: BudgetPeriod | None = None,
    ) -> list[Budget]:
        """List all budgets with optional filtering.

        Args:
            agent_id: Filter by agent ID (optional)
            period: Filter by budget period (optional)

        Returns:
            List of Budget objects
        """
        query = """
            SELECT id, name, total_amount, used_amount, period,
                   agent_id, created_at, updated_at
            FROM budgets
            WHERE 1=1
        """
        params = []

        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)

        if period is not None:
            query += " AND period = ?"
            params.append(period.value if isinstance(period, BudgetPeriod) else period)

        query += " ORDER BY created_at DESC"

        rows = self.backend.fetchall(query, tuple(params) if params else None)
        return [self._row_to_budget(row) for row in rows]

    def get_budget(self, budget_id: str) -> Budget | None:
        """Get a budget by ID.

        Args:
            budget_id: Budget ID

        Returns:
            Budget or None if not found
        """
        row = self.backend.fetchone(
            """
            SELECT id, name, total_amount, used_amount, period,
                   agent_id, created_at, updated_at
            FROM budgets
            WHERE id = ?
            """,
            (budget_id,),
        )
        return self._row_to_budget(row) if row else None

    def create_budget(self, data: BudgetCreate) -> Budget:
        """Create a new budget.

        Args:
            data: Budget creation input

        Returns:
            Created Budget
        """
        budget_id = str(uuid.uuid4())
        now = datetime.now()

        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO budgets
                    (id, name, total_amount, used_amount, period,
                     agent_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    budget_id,
                    data.name,
                    data.total_amount,
                    0.0,  # Initial used amount is 0
                    data.period.value if isinstance(data.period, BudgetPeriod) else data.period,
                    data.agent_id,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

        logger.info(f"Created budget: {data.name} ({budget_id})")
        return self.get_budget(budget_id)

    def update_budget(self, budget_id: str, data: BudgetUpdate) -> Budget | None:
        """Update a budget.

        Args:
            budget_id: Budget ID
            data: Update input

        Returns:
            Updated Budget or None if not found
        """
        existing = self.get_budget(budget_id)
        if not existing:
            return None

        # Build update fields
        updates = []
        params = []

        if data.name is not None:
            updates.append("name = ?")
            params.append(data.name)
        if data.total_amount is not None:
            updates.append("total_amount = ?")
            params.append(data.total_amount)
        if data.used_amount is not None:
            updates.append("used_amount = ?")
            params.append(data.used_amount)
        if data.period is not None:
            updates.append("period = ?")
            params.append(
                data.period.value if isinstance(data.period, BudgetPeriod) else data.period
            )
        if data.agent_id is not None:
            updates.append("agent_id = ?")
            params.append(data.agent_id if data.agent_id else None)

        if not updates:
            return existing

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(budget_id)

        with self.backend.transaction():
            self.backend.execute(
                f"UPDATE budgets SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

        logger.info(f"Updated budget: {budget_id}")
        return self.get_budget(budget_id)

    def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget.

        Args:
            budget_id: Budget ID

        Returns:
            True if deleted, False if not found
        """
        existing = self.get_budget(budget_id)
        if not existing:
            return False

        with self.backend.transaction():
            self.backend.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))

        logger.info(f"Deleted budget: {budget_id}")
        return True

    def add_usage(self, budget_id: str, amount: float) -> Budget | None:
        """Add usage to a budget.

        Args:
            budget_id: Budget ID
            amount: Amount to add to used_amount

        Returns:
            Updated Budget or None if not found
        """
        existing = self.get_budget(budget_id)
        if not existing:
            return None

        new_used = existing.used_amount + amount

        with self.backend.transaction():
            self.backend.execute(
                """
                UPDATE budgets
                SET used_amount = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_used, datetime.now().isoformat(), budget_id),
            )

        logger.info(f"Added ${amount:.2f} usage to budget {budget_id}")
        return self.get_budget(budget_id)

    def reset_usage(self, budget_id: str) -> Budget | None:
        """Reset usage for a budget.

        Args:
            budget_id: Budget ID

        Returns:
            Updated Budget or None if not found
        """
        existing = self.get_budget(budget_id)
        if not existing:
            return None

        with self.backend.transaction():
            self.backend.execute(
                """
                UPDATE budgets
                SET used_amount = 0, updated_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), budget_id),
            )

        logger.info(f"Reset usage for budget {budget_id}")
        return self.get_budget(budget_id)

    def get_summary(self) -> BudgetSummary:
        """Get summary of all budgets.

        Returns:
            BudgetSummary with aggregated statistics
        """
        budgets = self.list_budgets()

        if not budgets:
            return BudgetSummary(
                total_budget=0,
                total_used=0,
                total_remaining=0,
                percent_used=0,
                budget_count=0,
                exceeded_count=0,
                warning_count=0,
            )

        total_budget = sum(b.total_amount for b in budgets)
        total_used = sum(b.used_amount for b in budgets)
        exceeded_count = sum(1 for b in budgets if b.is_exceeded)
        warning_count = sum(1 for b in budgets if not b.is_exceeded and b.percent_used >= 80)

        return BudgetSummary(
            total_budget=round(total_budget, 2),
            total_used=round(total_used, 2),
            total_remaining=round(max(0, total_budget - total_used), 2),
            percent_used=round((total_used / total_budget) * 100, 1) if total_budget > 0 else 0,
            budget_count=len(budgets),
            exceeded_count=exceeded_count,
            warning_count=warning_count,
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _row_to_budget(self, row: dict) -> Budget:
        """Convert database row to Budget model."""
        return Budget(
            id=row["id"],
            name=row["name"],
            total_amount=float(row["total_amount"]),
            used_amount=float(row["used_amount"]),
            period=row["period"],
            agent_id=row.get("agent_id"),
            created_at=self._parse_datetime(row.get("created_at")) or datetime.now(),
            updated_at=self._parse_datetime(row.get("updated_at")) or datetime.now(),
        )

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime string from database."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
