"""Pydantic models for Budget management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class BudgetPeriod(str, Enum):
    """Budget period type."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Budget(BaseModel):
    """Budget entity."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    total_amount: float = Field(ge=0, description="Total budget amount in dollars")
    used_amount: float = Field(ge=0, description="Amount used so far in dollars")
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    agent_id: str | None = Field(
        default=None, description="Optional agent ID for agent-specific budgets"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def remaining_amount(self) -> float:
        """Get remaining budget amount."""
        return max(0, self.total_amount - self.used_amount)

    @property
    def percent_used(self) -> float:
        """Get percentage of budget used."""
        if self.total_amount <= 0:
            return 100.0
        return round((self.used_amount / self.total_amount) * 100, 1)

    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.used_amount > self.total_amount


class BudgetCreate(BaseModel):
    """Input for creating a budget."""

    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(..., min_length=1, max_length=255)
    total_amount: float = Field(ge=0, description="Total budget amount in dollars")
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    agent_id: str | None = Field(
        default=None, description="Optional agent ID for agent-specific budgets"
    )


class BudgetUpdate(BaseModel):
    """Input for updating a budget."""

    model_config = ConfigDict(use_enum_values=True)

    name: str | None = Field(default=None, min_length=1, max_length=255)
    total_amount: float | None = Field(
        default=None, ge=0, description="Total budget amount in dollars"
    )
    used_amount: float | None = Field(
        default=None, ge=0, description="Amount used so far in dollars"
    )
    period: BudgetPeriod | None = None
    agent_id: str | None = Field(
        default=None, description="Optional agent ID for agent-specific budgets"
    )


class BudgetSummary(BaseModel):
    """Summary of all budgets."""

    total_budget: float = Field(description="Sum of all budget amounts")
    total_used: float = Field(description="Sum of all used amounts")
    total_remaining: float = Field(description="Sum of all remaining amounts")
    percent_used: float = Field(description="Overall percentage used")
    budget_count: int = Field(description="Number of budgets")
    exceeded_count: int = Field(description="Number of exceeded budgets")
    warning_count: int = Field(description="Number of budgets over 80% used")
