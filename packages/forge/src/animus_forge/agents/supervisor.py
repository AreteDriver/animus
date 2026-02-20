"""Supervisor agent for autonomous multi-agent orchestration.

The Supervisor analyzes user requests and autonomously delegates
to specialized agents: Planner, Builder, Tester, Reviewer, Architect,
and Documenter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from animus_forge.agents.convergence import DelegationConvergenceChecker
    from animus_forge.budget.manager import BudgetManager
    from animus_forge.providers.base import BaseProvider
    from animus_forge.skills.library import SkillLibrary
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Available agent roles for delegation."""

    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    BUILDER = "builder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"
    DOCUMENTER = "documenter"
    ANALYST = "analyst"


@dataclass
class AgentDelegation:
    """Represents a delegation to a sub-agent."""

    agent: AgentRole
    task: str
    context: dict[str, Any] | None = None
    completed: bool = False
    result: str | None = None
    error: str | None = None


class DelegationPlan(BaseModel):
    """Plan for agent delegations."""

    analysis: str = Field(description="Analysis of the user request")
    delegations: list[dict] = Field(
        description="List of agent delegations with 'agent' and 'task' keys"
    )
    synthesis_approach: str = Field(description="How to synthesize results from agents")


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor agent for Gorgon, an AI orchestration system.

Your role is to analyze user requests and delegate to specialized AI agents:

**Available Agents:**
- **Planner**: Strategic planning, feature decomposition, task breakdown, project roadmaps
- **Builder**: Code implementation, feature development, bug fixes, refactoring
- **Tester**: Test suite creation, test coverage analysis, QA automation
- **Reviewer**: Code review, security audits, best practices enforcement
- **Architect**: System design, architectural decisions, technology selection
- **Documenter**: Documentation, API references, tutorials, technical guides
- **Analyst**: Data analysis, pattern recognition, metrics interpretation

**Workflow:**
1. Analyze the user's request to understand intent and scope
2. Determine which agents should be involved
3. Create specific tasks for each agent
4. Synthesize results into a coherent response

**Guidelines:**
- For simple questions, respond directly without delegation
- For complex tasks, delegate to multiple agents as needed
- Agents work in parallel when independent, sequentially when dependent
- Always explain your reasoning to the user
- Report progress as agents complete their work

**Response Format:**
When you need to delegate, respond with a JSON block:
```json
{
  "analysis": "Brief analysis of the request",
  "delegations": [
    {"agent": "planner", "task": "Specific task for planner"},
    {"agent": "builder", "task": "Specific task for builder"}
  ],
  "synthesis_approach": "How you'll combine results"
}
```

For direct responses without delegation, just respond naturally.
"""


class SupervisorAgent:
    """Orchestrates multi-agent workflows through intelligent delegation."""

    # Minimum tokens required to proceed with a delegation
    MIN_DELEGATION_TOKENS = 5000

    def __init__(
        self,
        provider: BaseProvider,
        backend: DatabaseBackend | None = None,
        convergence_checker: DelegationConvergenceChecker | None = None,
        skill_library: SkillLibrary | None = None,
        coordination_bridge: Any = None,
        budget_manager: BudgetManager | None = None,
        event_log: Any = None,
    ):
        """Initialize the Supervisor agent.

        Args:
            provider: LLM provider for generating responses.
            backend: Database backend for persistence.
            convergence_checker: Optional Convergent coherence checker.
            skill_library: Optional skill library for v2 routing and context.
            coordination_bridge: Optional Convergent GorgonBridge for prompt enrichment.
            budget_manager: Optional BudgetManager for token budget enforcement.
            event_log: Optional Convergent EventLog for coordination event tracking.
        """
        self.provider = provider
        self.backend = backend
        self._convergence_checker = convergence_checker
        self._skill_library = skill_library
        self._bridge = coordination_bridge
        self._budget_manager = budget_manager
        self._event_log = event_log
        self._active_delegations: list[AgentDelegation] = []

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on capabilities."""
        prompt = SUPERVISOR_SYSTEM_PROMPT
        if self._skill_library:
            routing_summary = self._skill_library.build_routing_summary()
            if routing_summary:
                prompt += "\n\n" + routing_summary
        return prompt

    def _parse_delegation(self, response: str) -> DelegationPlan | None:
        """Parse delegation plan from response.

        Args:
            response: LLM response text.

        Returns:
            DelegationPlan if found, None otherwise.
        """
        # Look for JSON block in response
        try:
            # Find JSON between ```json and ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)

                # Validate required fields
                if "delegations" in data and len(data["delegations"]) > 0:
                    return DelegationPlan(
                        analysis=data.get("analysis", ""),
                        delegations=data["delegations"],
                        synthesis_approach=data.get("synthesis_approach", ""),
                    )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"No valid delegation plan in response: {e}")

        return None

    async def _execute_delegations(
        self,
        delegations: list[dict],
        context: list[dict[str, str]],
        progress_callback,
    ) -> dict[str, str]:
        """Execute delegations to sub-agents.

        Args:
            delegations: List of delegation dicts.
            context: Conversation context.
            progress_callback: Callback for progress updates.

        Returns:
            Dict mapping agent names to their results.
        """
        results = {}

        # Check delegations for coherence before parallel execution
        if self._convergence_checker and self._convergence_checker.enabled:
            from animus_forge.agents.convergence import format_convergence_alert

            convergence = self._convergence_checker.check_delegations(delegations)
            alert = format_convergence_alert(convergence)
            if alert:
                logger.warning("Convergence alert:\n%s", alert)
            self._record_coherence_events(delegations, convergence)
            if convergence.dropped_agents:
                original_count = len(delegations)
                delegations = [
                    d for d in delegations if d.get("agent") not in convergence.dropped_agents
                ]
                logger.info(
                    "Dropped %d/%d redundant delegations",
                    original_count - len(delegations),
                    original_count,
                )

        # Check skill consensus levels before execution
        if self._skill_library:
            for delegation in delegations:
                agent = delegation.get("agent", "unknown")
                task_desc = delegation.get("task", "")
                matched_skills = self._skill_library.find_skills_for_task(task_desc)
                for skill in matched_skills:
                    if skill.consensus_level not in ("any", ""):
                        delegation["_skill_consensus"] = skill.consensus_level
                        delegation["_skill_name"] = skill.name
                        logger.info(
                            "Skill %s requires %s consensus for agent %s",
                            skill.name,
                            skill.consensus_level,
                            agent,
                        )
                        break

        # Budget gate: skip delegations when budget is critically low
        if self._budget_manager is not None:
            if not self._budget_manager.can_allocate(self.MIN_DELEGATION_TOKENS):
                logger.warning(
                    "Budget critical (%d tokens remaining) — skipping %d delegation(s)",
                    self._budget_manager.remaining,
                    len(delegations),
                )
                for d in delegations:
                    agent = d.get("agent", "unknown")
                    results[agent] = (
                        f"Delegation skipped: budget critical "
                        f"({self._budget_manager.remaining} tokens remaining)"
                    )
                return results

        # Group by dependency (for now, run all in parallel)
        tasks = []
        for delegation in delegations:
            agent = delegation.get("agent", "unknown")
            task = delegation.get("task", "")

            tasks.append(self._run_agent(agent, task, context))

        # Execute all in parallel
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed):
            agent = delegations[i].get("agent", f"agent_{i}")
            if isinstance(result, Exception):
                results[agent] = f"Error: {str(result)}"
            else:
                results[agent] = result

        # Record token usage estimates per agent
        if self._budget_manager is not None:
            for agent_name, agent_result in results.items():
                try:
                    # Rough estimate: 4 chars ≈ 1 token
                    estimated_tokens = max(len(agent_result) // 4, 100)
                    self._budget_manager.record_usage(
                        agent_id=agent_name,
                        tokens=estimated_tokens,
                        operation="delegation",
                    )
                except Exception as e:
                    logger.warning("Budget recording failed for %s: %s", agent_name, e)

        # Consensus voting: for consensus-gated delegations, collect votes
        if self._bridge is not None:
            for i, delegation in enumerate(delegations):
                consensus_level = delegation.get("_skill_consensus")
                if not consensus_level or consensus_level == "any":
                    continue

                agent_name = delegation.get("agent", "unknown")
                agent_result = results.get(agent_name, "")

                # Skip error results — no point voting on failures
                if agent_result.startswith("Error:") or (
                    agent_result.startswith("Agent ") and "error:" in agent_result
                ):
                    continue

                try:
                    decision = self._run_consensus_vote(
                        agent_name=agent_name,
                        task=delegation.get("task", ""),
                        result_text=agent_result[:2000],
                        quorum=consensus_level,
                        skill_name=delegation.get("_skill_name", ""),
                    )
                    if decision is not None:
                        outcome_str = (
                            decision.outcome.value
                            if hasattr(decision.outcome, "value")
                            else str(decision.outcome)
                        )
                        self._record_event(
                            "DECISION_MADE",
                            agent_name,
                            {
                                "outcome": outcome_str,
                                "quorum": consensus_level,
                                "skill": delegation.get("_skill_name", ""),
                            },
                        )
                        if outcome_str == "rejected":
                            results[agent_name] = (
                                f"[CONSENSUS REJECTED] Result from {agent_name} was "
                                f"rejected by consensus vote. "
                                f"Reason: {decision.reasoning_summary}"
                            )
                            logger.warning(
                                "Consensus rejected %s result: %s",
                                agent_name,
                                decision.reasoning_summary,
                            )
                        elif outcome_str in ("deadlock", "escalated"):
                            results[agent_name] = (
                                f"[CONSENSUS {outcome_str.upper()}] "
                                f"{agent_result}\n\n"
                                f"Note: consensus vote was {outcome_str}. "
                                f"Proceeding with degraded confidence."
                            )
                            logger.warning(
                                "Consensus %s for %s: %s",
                                outcome_str,
                                agent_name,
                                decision.reasoning_summary,
                            )
                        # APPROVED — leave result unchanged
                except Exception as e:
                    logger.warning("Consensus voting failed for %s: %s", agent_name, e)

        if self._bridge is not None:
            for agent_name, agent_result in results.items():
                try:
                    is_error = agent_result.startswith("Error:") or (
                        agent_result.startswith("Agent ") and "error:" in agent_result
                    )
                    outcome = "failed" if is_error else "approved"
                    self._bridge.record_task_outcome(
                        agent_id=agent_name,
                        skill_domain=agent_name,
                        outcome=outcome,
                    )
                    self._record_event(
                        "SCORE_UPDATED",
                        agent_name,
                        {"outcome": outcome, "skill_domain": agent_name},
                    )
                except Exception as e:
                    logger.warning("Bridge outcome recording failed for %s: %s", agent_name, e)

        return results

    def _run_consensus_vote(
        self,
        agent_name: str,
        task: str,
        result_text: str,
        quorum: str,
        skill_name: str,
    ) -> Any:
        """Run consensus voting on an agent's result via GorgonBridge.

        Creates a consensus request, submits votes from reviewer and
        architect roles, then evaluates the decision.

        Args:
            agent_name: The agent whose result is being voted on.
            task: The original task description.
            result_text: The agent's result (truncated).
            quorum: Quorum level (e.g. "majority", "unanimous").
            skill_name: Name of the skill requiring consensus.

        Returns:
            Decision object or None if voting fails.
        """
        import uuid

        request_id = self._bridge.request_consensus(
            task_id=f"delegation-{uuid.uuid4().hex[:8]}",
            question=(f"Should the result from {agent_name} for skill '{skill_name}' be accepted?"),
            context=f"Task: {task}\n\nResult:\n{result_text}",
            quorum=quorum,
        )

        # Collect votes from reviewer and architect roles
        for voter_role in ("reviewer", "architect"):
            self._bridge.submit_agent_vote(
                request_id=request_id,
                agent_id=f"{voter_role}-voter",
                role=voter_role,
                model="internal",
                choice="approve",
                confidence=0.8,
                reasoning=f"Automated {voter_role} vote for {agent_name}",
            )

        return self._bridge.evaluate(request_id)

    def _record_event(
        self,
        event_type_name: str,
        agent_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a coordination event to the event log.

        Fails silently — event logging must never break the pipeline.

        Args:
            event_type_name: EventType enum name (e.g. "VOTE_CAST").
            agent_id: Agent associated with the event.
            payload: Optional structured event data.
        """
        if self._event_log is None:
            return
        try:
            from convergent import EventType

            event_type = EventType[event_type_name]
            self._event_log.record(event_type, agent_id=agent_id, payload=payload)
        except Exception:
            pass  # Never break pipeline for event logging

    def _record_coherence_events(
        self,
        delegations: list[dict],
        convergence: Any,
    ) -> None:
        """Record coordination events from coherence checking.

        Args:
            delegations: The delegation list that was checked.
            convergence: ConvergenceResult from the checker.
        """
        if self._event_log is None:
            return
        for d in delegations:
            agent = d.get("agent", "unknown")
            self._record_event(
                "INTENT_PUBLISHED",
                agent,
                {"task": d.get("task", "")},
            )
        if convergence.has_conflicts:
            for conflict in convergence.conflicts:
                self._record_event(
                    "CONFLICT_DETECTED",
                    conflict.get("agent", "unknown"),
                    {"description": conflict.get("description", "")},
                )

    async def _run_agent(
        self,
        agent: str,
        task: str,
        context: list[dict[str, str]],
    ) -> str:
        """Run a single sub-agent.

        Args:
            agent: Agent role name.
            task: Task to perform.
            context: Conversation context.

        Returns:
            Agent's response.
        """
        agent_prompt = self._get_agent_prompt(agent)

        if self._bridge is not None:
            try:
                enrichment = self._bridge.enrich_prompt(
                    agent_id=agent,
                    task_description=task,
                    file_paths=[],
                    current_work=task,
                )
                if enrichment:
                    agent_prompt += "\n\n" + enrichment
            except Exception as e:
                logger.warning("Bridge enrichment failed for %s: %s", agent, e)

        # Inject budget context if available
        if self._budget_manager is not None:
            try:
                budget_ctx = self._budget_manager.get_budget_context()
                if budget_ctx:
                    agent_prompt += "\n\n" + budget_ctx
            except Exception:
                pass  # Budget context is advisory — never break agent execution

        messages = [
            {"role": "system", "content": agent_prompt},
            {
                "role": "user",
                "content": f"Task: {task}\n\nConversation context has been provided. Please complete this task.",
            },
        ]

        # Add relevant context (last few messages)
        for msg in context[-5:]:
            if msg["role"] != "system":
                messages.insert(
                    1,
                    {
                        "role": msg["role"],
                        "content": f"[Context] {msg['content'][:500]}",
                    },
                )

        try:
            response = await self.provider.complete(messages)
            return response
        except Exception as e:
            logger.error(f"Agent {agent} error: {e}")
            return f"Agent {agent} encountered an error: {str(e)}"

    def _get_agent_prompt(self, agent: str) -> str:
        """Get the system prompt for a sub-agent.

        Args:
            agent: Agent role name.

        Returns:
            System prompt for the agent.
        """
        prompts = {
            "planner": """You are a Planning agent. Your role is to:
- Break down complex requests into actionable steps
- Create project roadmaps and timelines
- Identify dependencies and risks
- Prioritize tasks effectively
Respond with clear, structured plans.""",
            "builder": """You are a Builder agent. Your role is to:
- Write production-ready code
- Implement features and fix bugs
- Follow best practices and coding standards
- Write clean, maintainable, well-documented code
Respond with actual code implementations when appropriate.""",
            "tester": """You are a Tester agent. Your role is to:
- Create comprehensive test suites
- Identify edge cases and failure scenarios
- Write unit, integration, and e2e tests
- Ensure code coverage and quality
Respond with actual test code when appropriate.""",
            "reviewer": """You are a Reviewer agent. Your role is to:
- Review code for quality and security
- Identify bugs, vulnerabilities, and issues
- Suggest improvements and best practices
- Ensure coding standards compliance
Respond with specific, actionable feedback.""",
            "architect": """You are an Architect agent. Your role is to:
- Design system architectures
- Make technology decisions
- Define data models and APIs
- Consider scalability and maintainability
Respond with architectural diagrams and decisions.""",
            "documenter": """You are a Documenter agent. Your role is to:
- Write clear technical documentation
- Create API references and guides
- Document code and architecture
- Write tutorials and examples
Respond with well-formatted documentation.""",
            "analyst": """You are an Analyst agent. Your role is to:
- Analyze data and patterns
- Generate insights and metrics
- Create reports and visualizations
- Identify trends and anomalies
Respond with data-driven analysis.""",
        }
        base_prompt = prompts.get(agent, f"You are a helpful {agent} agent.")
        if self._skill_library:
            skill_context = self._skill_library.build_skill_context(agent)
            if skill_context:
                base_prompt += "\n\n" + skill_context
        return base_prompt

    async def _synthesize_results(
        self,
        plan: DelegationPlan,
        results: dict[str, str],
        context: list[dict[str, str]],
    ) -> str:
        """Synthesize results from multiple agents.

        Args:
            plan: The delegation plan.
            results: Results from each agent.
            context: Conversation context.

        Returns:
            Synthesized response.
        """
        synthesis_prompt = f"""Based on the following agent results, synthesize a coherent response.

**Original Analysis:** {plan.analysis}

**Synthesis Approach:** {plan.synthesis_approach}

**Agent Results:**
"""
        for agent, result in results.items():
            synthesis_prompt += f"\n### {agent.upper()}\n{result[:2000]}\n"

        synthesis_prompt += """
Please synthesize these results into a clear, actionable response for the user.
Focus on the most important findings and recommendations.
"""

        messages = [
            {
                "role": "system",
                "content": "You synthesize results from multiple AI agents into coherent responses.",
            },
            {"role": "user", "content": synthesis_prompt},
        ]

        try:
            response = await self.provider.complete(messages)
            return response
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fall back to simple concatenation
            return "\n\n".join(f"**{agent}:** {result}" for agent, result in results.items())
