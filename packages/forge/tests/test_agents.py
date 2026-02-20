"""Tests for agents module."""

from __future__ import annotations

from animus_forge.agents import (
    AgentDelegation,
)
from animus_forge.agents.supervisor import AgentRole, DelegationPlan


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_all_roles_defined(self):
        """Test all expected roles are defined."""
        expected_roles = [
            "supervisor",
            "planner",
            "builder",
            "tester",
            "reviewer",
            "architect",
            "documenter",
            "analyst",
        ]
        for role in expected_roles:
            assert role in [r.value for r in AgentRole]

    def test_role_values(self):
        """Test role enum values."""
        assert AgentRole.SUPERVISOR.value == "supervisor"
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.BUILDER.value == "builder"
        assert AgentRole.TESTER.value == "tester"
        assert AgentRole.REVIEWER.value == "reviewer"
        assert AgentRole.ARCHITECT.value == "architect"
        assert AgentRole.DOCUMENTER.value == "documenter"
        assert AgentRole.ANALYST.value == "analyst"


class TestAgentDelegation:
    """Tests for AgentDelegation dataclass."""

    def test_delegation_creation(self):
        """Test creating an AgentDelegation."""
        delegation = AgentDelegation(
            agent=AgentRole.PLANNER,
            task="Break down the feature into steps",
        )
        assert delegation.agent == AgentRole.PLANNER
        assert delegation.task == "Break down the feature into steps"
        assert delegation.context is None
        assert delegation.completed is False
        assert delegation.result is None
        assert delegation.error is None

    def test_delegation_with_context(self):
        """Test AgentDelegation with context."""
        delegation = AgentDelegation(
            agent=AgentRole.BUILDER,
            task="Implement the authentication API",
            context={"language": "python", "framework": "fastapi"},
        )
        assert delegation.context == {"language": "python", "framework": "fastapi"}

    def test_delegation_completed(self):
        """Test completed delegation."""
        delegation = AgentDelegation(
            agent=AgentRole.TESTER,
            task="Write unit tests",
            completed=True,
            result="Created 15 unit tests with 100% coverage",
        )
        assert delegation.completed is True
        assert delegation.result == "Created 15 unit tests with 100% coverage"

    def test_delegation_with_error(self):
        """Test delegation with error."""
        delegation = AgentDelegation(
            agent=AgentRole.REVIEWER,
            task="Review the code",
            completed=True,
            error="Code review failed: too many warnings",
        )
        assert delegation.error == "Code review failed: too many warnings"


class TestDelegationPlan:
    """Tests for DelegationPlan model."""

    def test_plan_creation(self):
        """Test creating a DelegationPlan."""
        plan = DelegationPlan(
            analysis="This is a feature request for user authentication",
            delegations=[
                {"agent": "planner", "task": "Create implementation plan"},
                {"agent": "builder", "task": "Implement the feature"},
            ],
            synthesis_approach="Combine plan and implementation into a complete solution",
        )
        assert plan.analysis == "This is a feature request for user authentication"
        assert len(plan.delegations) == 2
        assert plan.delegations[0]["agent"] == "planner"
        assert plan.synthesis_approach == "Combine plan and implementation into a complete solution"

    def test_plan_validation(self):
        """Test DelegationPlan field validation."""
        # Should work with empty delegations
        plan = DelegationPlan(
            analysis="Simple question",
            delegations=[],
            synthesis_approach="Direct response",
        )
        assert plan.delegations == []


# Note: AgentProvider and SupervisorAgent tests require
# matching the actual implementation's API. The model/enum
# tests above provide coverage of the data structures.
# Full integration tests would be added after API stabilization.
