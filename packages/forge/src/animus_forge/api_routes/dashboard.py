"""Dashboard and agent definition endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Header

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, not_found
from animus_forge.api_models import (
    AgentDefinitionResponse,
    AgentUsage,
    BudgetStatus,
    DailyUsage,
    DashboardBudget,
    DashboardStats,
    RecentExecution,
)
from animus_forge.api_routes.auth import verify_auth
from animus_forge.contracts.base import AgentRole
from animus_forge.contracts.definitions import _CONTRACT_REGISTRY
from animus_forge.state import get_database

router = APIRouter()

# ---------------------------------------------------------------------------
# Agent display metadata (used by frontend)
# ---------------------------------------------------------------------------

_AGENT_ICONS = {
    AgentRole.PLANNER: "Brain",
    AgentRole.BUILDER: "Code",
    AgentRole.TESTER: "TestTube",
    AgentRole.REVIEWER: "Search",
    AgentRole.ANALYST: "BarChart3",
    AgentRole.VISUALIZER: "PieChart",
    AgentRole.REPORTER: "FileOutput",
    AgentRole.DATA_ANALYST: "Database",
    AgentRole.DEVOPS: "Server",
    AgentRole.SECURITY_AUDITOR: "Shield",
    AgentRole.MIGRATOR: "ArrowRightLeft",
    AgentRole.MODEL_BUILDER: "Boxes",
}

_AGENT_COLORS = {
    AgentRole.PLANNER: "#8B5CF6",
    AgentRole.BUILDER: "#3B82F6",
    AgentRole.TESTER: "#10B981",
    AgentRole.REVIEWER: "#F59E0B",
    AgentRole.ANALYST: "#14B8A6",
    AgentRole.VISUALIZER: "#F97316",
    AgentRole.REPORTER: "#8B5CF6",
    AgentRole.DATA_ANALYST: "#06B6D4",
    AgentRole.DEVOPS: "#6366F1",
    AgentRole.SECURITY_AUDITOR: "#EF4444",
    AgentRole.MIGRATOR: "#EC4899",
    AgentRole.MODEL_BUILDER: "#A855F7",
}

_AGENT_CAPABILITIES = {
    AgentRole.PLANNER: [
        "Task decomposition",
        "Dependency analysis",
        "Resource estimation",
    ],
    AgentRole.BUILDER: ["Code generation", "Refactoring", "Implementation"],
    AgentRole.TESTER: ["Unit tests", "Integration tests", "Edge case coverage"],
    AgentRole.REVIEWER: ["Code review", "Security audit", "Best practices check"],
    AgentRole.ANALYST: ["Data analysis", "Pattern recognition", "Insights extraction"],
    AgentRole.VISUALIZER: [
        "Chart generation",
        "Dashboard design",
        "Data visualization",
    ],
    AgentRole.REPORTER: [
        "Summary generation",
        "Progress reports",
        "Stakeholder updates",
    ],
    AgentRole.DATA_ANALYST: ["SQL queries", "Pandas pipelines", "Statistical analysis"],
    AgentRole.DEVOPS: [
        "CI/CD pipelines",
        "Infrastructure as code",
        "Container orchestration",
    ],
    AgentRole.SECURITY_AUDITOR: [
        "Vulnerability scanning",
        "OWASP compliance",
        "Dependency audits",
    ],
    AgentRole.MIGRATOR: ["Framework upgrades", "Code refactoring", "API migrations"],
    AgentRole.MODEL_BUILDER: ["3D modeling", "Scene creation", "Asset generation"],
}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@router.get("/agents", responses=AUTH_RESPONSES)
def list_agents(
    authorization: str | None = Header(None),
) -> list[AgentDefinitionResponse]:
    """List all available agent role definitions."""
    verify_auth(authorization)

    agents = []
    for role, contract in _CONTRACT_REGISTRY.items():
        agents.append(
            AgentDefinitionResponse(
                id=role.value,
                name=role.value.replace("_", " ").title(),
                description=contract.description,
                capabilities=_AGENT_CAPABILITIES.get(role, []),
                icon=_AGENT_ICONS.get(role, "Bot"),
                color=_AGENT_COLORS.get(role, "#6B7280"),
            )
        )

    return agents


@router.get("/agents/{agent_id}", responses=CRUD_RESPONSES)
def get_agent(agent_id: str, authorization: str | None = Header(None)):
    """Get a specific agent role definition by ID."""
    verify_auth(authorization)

    try:
        role = AgentRole(agent_id)
    except ValueError:
        raise not_found("Agent", agent_id)

    if role not in _CONTRACT_REGISTRY:
        raise not_found("Agent", agent_id)

    contract = _CONTRACT_REGISTRY[role]
    return AgentDefinitionResponse(
        id=role.value,
        name=role.value.replace("_", " ").title(),
        description=contract.description,
        capabilities=_AGENT_CAPABILITIES.get(role, []),
        icon=_AGENT_ICONS.get(role, "Bot"),
        color=_AGENT_COLORS.get(role, "#6B7280"),
    )


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------


@router.get("/dashboard/stats", responses=AUTH_RESPONSES)
def get_dashboard_stats(authorization: str | None = Header(None)):
    """Get dashboard statistics."""
    verify_auth(authorization)

    from datetime import date

    from animus_forge.executions import ExecutionStatus

    today_start = datetime.combine(date.today(), datetime.min.time())

    workflows = state.workflow_engine.list_workflows()
    total_workflows = len(workflows) if workflows else 0

    backend = get_database()

    active_row = backend.fetchone(
        "SELECT COUNT(*) as count FROM executions WHERE status IN (?, ?)",
        (ExecutionStatus.RUNNING.value, ExecutionStatus.PAUSED.value),
    )
    active_executions = active_row["count"] if active_row else 0

    completed_row = backend.fetchone(
        "SELECT COUNT(*) as count FROM executions WHERE status = ? AND datetime(completed_at) >= datetime(?)",
        (ExecutionStatus.COMPLETED.value, today_start.isoformat()),
    )
    completed_today = completed_row["count"] if completed_row else 0

    failed_row = backend.fetchone(
        "SELECT COUNT(*) as count FROM executions WHERE status = ? AND datetime(completed_at) >= datetime(?)",
        (ExecutionStatus.FAILED.value, today_start.isoformat()),
    )
    failed_today = failed_row["count"] if failed_row else 0

    tokens_row = backend.fetchone(
        """
        SELECT COALESCE(SUM(m.total_tokens), 0) as tokens,
               COALESCE(SUM(m.total_cost_cents), 0) as cost_cents
        FROM execution_metrics m
        JOIN executions e ON e.id = m.execution_id
        WHERE datetime(e.created_at) >= datetime(?)
        """,
        (today_start.isoformat(),),
    )
    total_tokens_today = tokens_row["tokens"] if tokens_row else 0
    total_cost_today = (tokens_row["cost_cents"] / 100.0) if tokens_row else 0.0

    return DashboardStats(
        totalWorkflows=total_workflows,
        activeExecutions=active_executions,
        completedToday=completed_today,
        failedToday=failed_today,
        totalTokensToday=total_tokens_today,
        totalCostToday=total_cost_today,
    )


@router.get("/dashboard/recent-executions", responses=AUTH_RESPONSES)
def get_recent_executions(
    limit: int = 5,
    authorization: str | None = Header(None),
):
    """Get recent executions for dashboard display."""
    verify_auth(authorization)

    result = state.execution_manager.list_executions(page=1, page_size=limit)
    executions = []

    for exec in result.data:
        if exec.started_at:
            delta = datetime.now() - exec.started_at
            if delta.total_seconds() < 60:
                time_str = "just now"
            elif delta.total_seconds() < 3600:
                mins = int(delta.total_seconds() / 60)
                time_str = f"{mins} min ago"
            elif delta.total_seconds() < 86400:
                hours = int(delta.total_seconds() / 3600)
                time_str = f"{hours} hour{'s' if hours > 1 else ''} ago"
            else:
                days = int(delta.total_seconds() / 86400)
                time_str = f"{days} day{'s' if days > 1 else ''} ago"
        else:
            time_str = "pending"

        executions.append(
            RecentExecution(
                id=exec.id,
                name=exec.workflow_name,
                status=exec.status.value,
                time=time_str,
            )
        )

    return executions


@router.get("/dashboard/usage/daily", responses=AUTH_RESPONSES)
def get_daily_usage(
    days: int = 7,
    authorization: str | None = Header(None),
):
    """Get daily token and cost usage for the past N days."""
    verify_auth(authorization)

    from datetime import date

    backend = get_database()
    usage_data = []

    for i in range(days - 1, -1, -1):
        target_date = date.today() - timedelta(days=i)
        day_start = datetime.combine(target_date, datetime.min.time())
        day_end = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

        row = backend.fetchone(
            """
            SELECT COALESCE(SUM(m.total_tokens), 0) as tokens,
                   COALESCE(SUM(m.total_cost_cents), 0) as cost_cents
            FROM execution_metrics m
            JOIN executions e ON e.id = m.execution_id
            WHERE datetime(e.created_at) >= datetime(?)
            AND datetime(e.created_at) < datetime(?)
            """,
            (day_start.isoformat(), day_end.isoformat()),
        )

        day_name = target_date.strftime("%a")

        usage_data.append(
            DailyUsage(
                date=day_name,
                tokens=row["tokens"] if row else 0,
                cost=round((row["cost_cents"] / 100.0) if row else 0.0, 2),
            )
        )

    return usage_data


@router.get("/dashboard/usage/by-agent", responses=AUTH_RESPONSES)
def get_agent_usage(authorization: str | None = Header(None)):
    """Get token usage breakdown by agent role."""
    verify_auth(authorization)

    backend = get_database()

    rows = backend.fetchall(
        """
        SELECT e.workflow_name, COALESCE(SUM(m.total_tokens), 0) as tokens
        FROM executions e
        JOIN execution_metrics m ON e.id = m.execution_id
        WHERE datetime(e.created_at) >= datetime('now', '-30 days')
        GROUP BY e.workflow_name
        ORDER BY tokens DESC
        LIMIT 10
        """
    )

    agent_map = {
        "planner": 0,
        "builder": 0,
        "tester": 0,
        "reviewer": 0,
        "documenter": 0,
    }

    for row in rows:
        name_lower = (row["workflow_name"] or "").lower()
        tokens = row["tokens"] or 0

        if "plan" in name_lower or "analysis" in name_lower:
            agent_map["planner"] += tokens
        elif "build" in name_lower or "implement" in name_lower or "code" in name_lower:
            agent_map["builder"] += tokens
        elif "test" in name_lower:
            agent_map["tester"] += tokens
        elif "review" in name_lower:
            agent_map["reviewer"] += tokens
        elif "doc" in name_lower:
            agent_map["documenter"] += tokens
        else:
            agent_map["builder"] += tokens

    usage = [
        AgentUsage(agent=agent.title(), tokens=tokens)
        for agent, tokens in agent_map.items()
        if tokens > 0
    ]

    if not usage:
        usage = [
            AgentUsage(agent="Planner", tokens=0),
            AgentUsage(agent="Builder", tokens=0),
            AgentUsage(agent="Tester", tokens=0),
            AgentUsage(agent="Reviewer", tokens=0),
            AgentUsage(agent="Documenter", tokens=0),
        ]

    return usage


@router.get("/dashboard/budget", responses=AUTH_RESPONSES)
def get_dashboard_budget(authorization: str | None = Header(None)):
    """Get budget status for dashboard display."""
    verify_auth(authorization)

    budget_limits = {
        "Builder": 40.0,
        "Planner": 20.0,
        "Reviewer": 25.0,
        "Tester": 15.0,
    }
    total_budget = 100.0

    backend = get_database()

    from datetime import date

    month_start = date.today().replace(day=1)

    rows = backend.fetchall(
        """
        SELECT e.workflow_name, COALESCE(SUM(m.total_cost_cents), 0) as cost_cents
        FROM executions e
        JOIN execution_metrics m ON e.id = m.execution_id
        WHERE datetime(e.created_at) >= datetime(?)
        GROUP BY e.workflow_name
        """,
        (datetime.combine(month_start, datetime.min.time()).isoformat(),),
    )

    agent_costs = {agent: 0.0 for agent in budget_limits}

    for row in rows:
        name_lower = (row["workflow_name"] or "").lower()
        cost = (row["cost_cents"] or 0) / 100.0

        if "plan" in name_lower or "analysis" in name_lower:
            agent_costs["Planner"] += cost
        elif "review" in name_lower:
            agent_costs["Reviewer"] += cost
        elif "test" in name_lower:
            agent_costs["Tester"] += cost
        else:
            agent_costs["Builder"] += cost

    total_used = sum(agent_costs.values())

    by_agent = [
        BudgetStatus(agent=agent, used=round(cost, 2), limit=budget_limits[agent])
        for agent, cost in agent_costs.items()
    ]

    alert = None
    for bs in by_agent:
        if bs.limit > 0:
            percent = (bs.used / bs.limit) * 100
            if percent >= 80:
                alert = f"{bs.agent} agent at {int(percent)}% of monthly limit"
                break

    return DashboardBudget(
        totalBudget=total_budget,
        totalUsed=round(total_used, 2),
        percentUsed=round((total_used / total_budget) * 100, 1) if total_budget > 0 else 0,
        byAgent=by_agent,
        alert=alert,
    )
