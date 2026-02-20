"""Budget management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, Query

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, bad_request, not_found
from animus_forge.api_models import BudgetCreateRequest, BudgetUpdateRequest
from animus_forge.api_routes.auth import verify_auth

router = APIRouter()


@router.get("/budgets", responses=AUTH_RESPONSES)
def list_budgets(
    agent_id: str | None = Query(None, description="Filter by agent ID"),
    period: str | None = Query(None, description="Filter by period (daily/weekly/monthly)"),
    authorization: str | None = Header(None),
):
    """List all budgets with optional filtering."""
    verify_auth(authorization)

    from animus_forge.budget import BudgetPeriod

    period_enum = None
    if period:
        try:
            period_enum = BudgetPeriod(period)
        except ValueError:
            raise bad_request(
                "Invalid period",
                {"valid_periods": ["daily", "weekly", "monthly"]},
            )

    budgets = state.budget_manager.list_budgets(agent_id=agent_id, period=period_enum)
    return [b.model_dump(mode="json") for b in budgets]


@router.get("/budgets/summary", responses=AUTH_RESPONSES)
def get_budget_summary(authorization: str | None = Header(None)):
    """Get overall budget summary."""
    verify_auth(authorization)

    summary = state.budget_manager.get_summary()
    return summary.model_dump()


@router.get("/budgets/{budget_id}", responses=CRUD_RESPONSES)
def get_budget(budget_id: str, authorization: str | None = Header(None)):
    """Get a specific budget by ID."""
    verify_auth(authorization)

    budget = state.budget_manager.get_budget(budget_id)
    if not budget:
        raise not_found("Budget", budget_id)
    return budget.model_dump(mode="json")


@router.post("/budgets", responses=CRUD_RESPONSES)
def create_budget(
    request: BudgetCreateRequest,
    authorization: str | None = Header(None),
):
    """Create a new budget."""
    verify_auth(authorization)

    from animus_forge.budget import BudgetCreate, BudgetPeriod

    try:
        period = BudgetPeriod(request.period)
    except ValueError:
        raise bad_request(
            "Invalid period",
            {"valid_periods": ["daily", "weekly", "monthly"]},
        )

    if not request.name or len(request.name) < 1:
        raise bad_request("Budget name is required")

    if request.total_amount < 0:
        raise bad_request("Total amount must be non-negative")

    budget_create = BudgetCreate(
        name=request.name,
        total_amount=request.total_amount,
        period=period,
        agent_id=request.agent_id,
    )

    try:
        budget = state.budget_manager.create_budget(budget_create)
        return budget.model_dump(mode="json")
    except ValueError as e:
        raise bad_request(str(e))


@router.patch("/budgets/{budget_id}", responses=CRUD_RESPONSES)
def update_budget(
    budget_id: str,
    request: BudgetUpdateRequest,
    authorization: str | None = Header(None),
):
    """Update a budget."""
    verify_auth(authorization)

    from animus_forge.budget import BudgetPeriod, BudgetUpdate

    period = None
    if request.period is not None:
        try:
            period = BudgetPeriod(request.period)
        except ValueError:
            raise bad_request(
                "Invalid period",
                {"valid_periods": ["daily", "weekly", "monthly"]},
            )

    if request.total_amount is not None and request.total_amount < 0:
        raise bad_request("Total amount must be non-negative")
    if request.used_amount is not None and request.used_amount < 0:
        raise bad_request("Used amount must be non-negative")

    budget_update = BudgetUpdate(
        name=request.name,
        total_amount=request.total_amount,
        used_amount=request.used_amount,
        period=period,
        agent_id=request.agent_id,
    )

    budget = state.budget_manager.update_budget(budget_id, budget_update)
    if not budget:
        raise not_found("Budget", budget_id)
    return budget.model_dump(mode="json")


@router.delete("/budgets/{budget_id}", responses=CRUD_RESPONSES)
def delete_budget(budget_id: str, authorization: str | None = Header(None)):
    """Delete a budget."""
    verify_auth(authorization)

    if state.budget_manager.delete_budget(budget_id):
        return {"status": "success"}
    raise not_found("Budget", budget_id)


@router.post("/budgets/{budget_id}/add-usage", responses=CRUD_RESPONSES)
def add_budget_usage(
    budget_id: str,
    amount: float = Query(..., ge=0, description="Amount to add to usage"),
    authorization: str | None = Header(None),
):
    """Add usage to a budget."""
    verify_auth(authorization)

    budget = state.budget_manager.add_usage(budget_id, amount)
    if not budget:
        raise not_found("Budget", budget_id)
    return budget.model_dump(mode="json")


@router.post("/budgets/{budget_id}/reset", responses=CRUD_RESPONSES)
def reset_budget_usage(budget_id: str, authorization: str | None = Header(None)):
    """Reset usage for a budget."""
    verify_auth(authorization)

    budget = state.budget_manager.reset_usage(budget_id)
    if not budget:
        raise not_found("Budget", budget_id)
    return budget.model_dump(mode="json")
