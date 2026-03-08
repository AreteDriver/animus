"""Supervisor agent for autonomous multi-agent orchestration.

The Supervisor analyzes user requests and autonomously delegates
to specialized agents: Planner, Builder, Tester, Reviewer, Architect,
and Documenter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from animus_forge.agents.agent_config import AgentConfig
    from animus_forge.agents.convergence import DelegationConvergenceChecker
    from animus_forge.agents.message_bus import AgentMessageBus
    from animus_forge.agents.process_registry import ProcessRegistry
    from animus_forge.agents.subagent_manager import SubAgentManager
    from animus_forge.budget.manager import BudgetManager
    from animus_forge.providers.base import BaseProvider
    from animus_forge.skills.library import SkillLibrary
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

# Cached agent prompts loaded from config/agent_prompts.json
_agent_prompts_cache: dict[str, dict[str, str]] | None = None

_AGENT_PROMPTS_PATH = Path(__file__).resolve().parents[3] / "config" / "agent_prompts.json"


def _load_agent_prompts(
    path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Load agent prompts from JSON config file.

    Reads and caches agent prompt definitions from the config file.
    Returns an empty dict if the file doesn't exist or is invalid.

    Args:
        path: Optional path override (for testing). Defaults to
            config/agent_prompts.json relative to the package root.

    Returns:
        Dict mapping role names to their prompt definitions.
    """
    global _agent_prompts_cache
    if path is not None:
        # Explicit path bypasses cache (used in tests)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to load agent prompts from %s: %s", path, exc)
        return {}

    if _agent_prompts_cache is not None:
        return _agent_prompts_cache

    try:
        data = json.loads(_AGENT_PROMPTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _agent_prompts_cache = data
            return _agent_prompts_cache
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to load agent prompts from %s: %s", _AGENT_PROMPTS_PATH, exc)
    _agent_prompts_cache = {}
    return _agent_prompts_cache


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


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor agent for Animus Forge, an AI orchestration system.

Your role is to analyze user requests and delegate to specialized AI agents.

**Available Agents:**

Tool-equipped agents (can read files, search code, write files, run commands):
- **Builder**: Code implementation, feature development, bug fixes, refactoring
- **Tester**: Test suite creation, test coverage analysis, QA automation
- **Reviewer**: Code review, security audits, best practices enforcement, file inspection
- **Analyst**: Data analysis, pattern recognition, codebase inspection, metrics

Text-only agents (generate responses without tool access):
- **Planner**: Strategic planning, feature decomposition, task breakdown, project roadmaps
- **Architect**: System design, architectural decisions, technology selection
- **Documenter**: Documentation, API references, tutorials, technical guides

**IMPORTANT: For any task that requires reading, inspecting, or modifying files, you MUST delegate to a tool-equipped agent (builder, tester, reviewer, or analyst). Text-only agents cannot access the filesystem.**

**Workflow:**
1. Analyze the user's request to understand intent and scope
2. Determine which agents should be involved — prefer tool-equipped agents for concrete tasks
3. Create specific tasks for each agent
4. Synthesize results into a coherent response

**Guidelines:**
- For simple questions, respond directly without delegation
- For tasks requiring file access, always use tool-equipped agents
- For complex tasks, delegate to multiple agents as needed
- Agents work in parallel when independent, sequentially when dependent
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

    # Agent roles that get access to tools for execution
    TOOL_EQUIPPED_ROLES = {"builder", "tester", "reviewer", "analyst"}

    def __init__(
        self,
        provider: BaseProvider,
        backend: DatabaseBackend | None = None,
        convergence_checker: DelegationConvergenceChecker | None = None,
        skill_library: SkillLibrary | None = None,
        coordination_bridge: Any = None,
        budget_manager: BudgetManager | None = None,
        event_log: Any = None,
        tool_registry: Any = None,
        subagent_manager: SubAgentManager | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        message_bus: AgentMessageBus | None = None,
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
            tool_registry: Optional ForgeToolRegistry for tool-equipped agents.
            subagent_manager: Optional SubAgentManager for async parallel execution.
            agent_configs: Optional per-role AgentConfig overrides for isolation.
            message_bus: Optional AgentMessageBus for inter-agent messaging.
        """
        self.provider = provider
        self.backend = backend
        self._convergence_checker = convergence_checker
        self._skill_library = skill_library
        self._bridge = coordination_bridge
        self._budget_manager = budget_manager
        self._event_log = event_log
        self._tool_registry = tool_registry
        self._subagent_manager = subagent_manager
        self._agent_configs = agent_configs
        self._message_bus = message_bus
        self._active_delegations: list[AgentDelegation] = []

    @property
    def message_bus(self) -> AgentMessageBus | None:
        """The agent message bus, if configured."""
        return self._message_bus

    @property
    def process_registry(self) -> ProcessRegistry | None:
        """Build a ProcessRegistry from available managers.

        Lazily constructs a ProcessRegistry aggregating all
        registered background task systems.
        """
        from animus_forge.agents.process_registry import ProcessRegistry

        if not any([self._subagent_manager, self._budget_manager]):
            return None
        return ProcessRegistry(
            subagent_manager=self._subagent_manager,
        )

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

        # Publish delegation start to message bus
        if self._message_bus is not None:
            for d in delegations:
                agent_name = d.get("agent", "unknown")
                self._message_bus.publish(
                    topic="delegation.started",
                    sender="supervisor",
                    payload={"agent": agent_name, "task": d.get("task", "")[:200]},
                )

        # Parallel execution via SubAgentManager (if available)
        if self._subagent_manager is not None:
            results = await self._execute_delegations_parallel(
                delegations,
                context,
                progress_callback,
                results,
            )
        else:
            # Sequential fallback: each agent receives prior outputs
            prior_results: dict[str, str] = {}
            for i, delegation in enumerate(delegations):
                agent = delegation.get("agent", f"agent_{i}")
                task = delegation.get("task", "")

                try:
                    result = await self._run_agent(
                        agent,
                        task,
                        context,
                        progress_callback,
                        prior_results=prior_results if prior_results else None,
                    )
                    results[agent] = result
                except Exception as e:
                    results[agent] = f"Error: {str(e)}"

                # Accumulate for subsequent agents
                prior_results[agent] = results[agent]

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

        # Publish delegation completions to message bus
        if self._message_bus is not None:
            for agent_name, agent_result in results.items():
                is_error = agent_result.startswith("Error:") or (
                    agent_result.startswith("Agent ") and "error:" in agent_result
                )
                self._message_bus.publish(
                    topic="delegation.completed" if not is_error else "delegation.failed",
                    sender="supervisor",
                    payload={
                        "agent": agent_name,
                        "success": not is_error,
                        "result_length": len(agent_result),
                    },
                )

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
                    result_text = agent_result
                    if len(result_text) > 2000:
                        logger.warning(
                            "Agent %s result truncated from %d to 2000 chars for consensus",
                            agent_name,
                            len(result_text),
                        )
                        result_text = result_text[:2000]
                    decision = self._run_consensus_vote(
                        agent_name=agent_name,
                        task=delegation.get("task", ""),
                        result_text=result_text,
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

    async def _execute_delegations_parallel(
        self,
        delegations: list[dict],
        context: list[dict[str, str]],
        progress_callback: Any,
        results: dict[str, str],
    ) -> dict[str, str]:
        """Execute delegations in parallel via SubAgentManager.

        Each agent gets its own AgentConfig controlling isolation,
        tool access, model routing, and timeouts.

        Args:
            delegations: List of delegation dicts.
            context: Conversation context.
            progress_callback: Optional progress callback.
            results: Results dict to populate.

        Returns:
            Populated results dict.
        """
        from animus_forge.agents.subagent_manager import RunStatus

        async def agent_executor(agent: str, task: str, config: Any) -> str:
            """Execute a single agent with per-agent config."""
            return await self._run_agent(
                agent,
                task,
                context,
                progress_callback,
                agent_config=config,
            )

        runs = await self._subagent_manager.spawn_batch(
            delegations,
            agent_executor,
        )

        for run in runs:
            if run.status == RunStatus.COMPLETED and run.result is not None:
                results[run.agent] = run.result
            elif run.status == RunStatus.TIMED_OUT:
                results[run.agent] = (
                    f"Agent {run.agent} timed out after {run.config.timeout_seconds}s"
                )
            elif run.status == RunStatus.CANCELLED:
                results[run.agent] = f"Agent {run.agent} was cancelled"
            else:
                results[run.agent] = f"Error: {run.error or 'unknown failure'}"

        return results

    def _get_agent_tool_registry(self, config: Any) -> Any:
        """Create a per-agent tool registry with filtered access.

        If the agent has allowed_tools or denied_tools, wraps the global
        registry with a filtered view. Otherwise returns the global one.

        Args:
            config: AgentConfig for the agent.

        Returns:
            Tool registry (possibly filtered) or None.
        """
        if self._tool_registry is None:
            return None

        # Check if config specifies tool filtering
        if not hasattr(config, "allowed_tools") and not hasattr(config, "denied_tools"):
            return self._tool_registry

        all_tools = [t.name for t in self._tool_registry.tools]
        effective = config.get_effective_tools(all_tools)

        # If all tools allowed, return global registry
        if set(effective) == set(all_tools):
            return self._tool_registry

        # Return a filtered wrapper
        return _FilteredToolRegistry(self._tool_registry, effective)

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

    # Maximum characters per prior agent output injected into context
    MAX_PRIOR_RESULT_CHARS = 2000

    async def _run_agent(
        self,
        agent: str,
        task: str,
        context: list[dict[str, str]],
        progress_callback: Any = None,
        prior_results: dict[str, str] | None = None,
        agent_config: AgentConfig | None = None,
    ) -> str:
        """Run a single sub-agent, optionally with tool access.

        Tool-equipped roles (builder, tester, reviewer, analyst) get an
        iterative tool loop — they can read files, search code, write files,
        and run commands. Other roles get text-only completion.

        When agent_config is provided, tool access and iteration limits
        are scoped per-agent for isolation.

        Args:
            agent: Agent role name.
            task: Task to perform.
            context: Conversation context.
            progress_callback: Optional callable(stage, detail) for updates.
            prior_results: Optional mapping of role_name -> output from
                previously completed subagents. Injected into the prompt
                so this agent can build on prior work.

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

        # Determine tool access (per-agent config overrides global)
        if agent_config is not None:
            effective_registry = self._get_agent_tool_registry(agent_config)
            effective_tools = (
                agent_config.get_effective_tools([t.name for t in self._tool_registry.tools])
                if self._tool_registry
                else []
            )
            use_tools = effective_registry is not None and len(effective_tools) > 0
        else:
            effective_registry = self._tool_registry
            use_tools = self._tool_registry is not None and agent in self.TOOL_EQUIPPED_ROLES

        if use_tools:
            tool_names = [t.name for t in effective_registry.tools] if effective_registry else []
            agent_prompt += (
                "\n\nYou have access to tools for executing your task. "
                f"Available tools: {', '.join(tool_names)}. "
                "Use tools to read files, search code, and verify your work. "
                "When you're done, provide your final response as text."
            )

        # Build prior agent outputs section if available
        prior_outputs_section = ""
        if prior_results:
            parts = []
            for role_name, output in prior_results.items():
                truncated = output[: self.MAX_PRIOR_RESULT_CHARS]
                parts.append(f"### {role_name}\n{truncated}")
            prior_outputs_section = "\n\n## Prior Agent Outputs\n\n" + "\n\n".join(parts)

        user_content = f"Task: {task}"
        if prior_outputs_section:
            user_content += prior_outputs_section
        user_content += "\n\nConversation context has been provided. Please complete this task."

        messages: list[dict] = [
            {"role": "system", "content": agent_prompt},
            {"role": "user", "content": user_content},
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

        import time as _time

        _start_ns = _time.perf_counter_ns()
        try:
            if use_tools:
                max_iters = agent_config.max_tool_iterations if agent_config else 8
                response = await self.provider.complete_with_tools(
                    messages=messages,
                    tool_registry=effective_registry,
                    progress_callback=progress_callback,
                    agent_id=agent,
                    max_iterations=max_iters,
                )
            else:
                response = await self.provider.complete(messages)

            # Trace to ChainLog (fire-and-forget, never blocks)
            _duration = (_time.perf_counter_ns() - _start_ns) // 1_000_000
            try:
                from animus_forge.agents.chainlog_bridge import trace_agent_action

                await trace_agent_action(
                    agent_id=agent,
                    action_type="delegation",
                    input_data={"task": task[:500]},
                    output=response[:500] if isinstance(response, str) else str(response)[:500],
                    duration_ms=_duration,
                )
            except Exception:
                pass  # ChainLog tracing is advisory — never break agent execution

            return response
        except Exception as e:
            logger.error(f"Agent {agent} error: {e}")
            return f"Agent {agent} encountered an error: {str(e)}"

    def _get_agent_prompt(self, agent: str) -> str:
        """Get the system prompt for a sub-agent.

        Loads from config/agent_prompts.json first, falling back to
        hardcoded defaults if the file is missing or the role isn't found.

        Args:
            agent: Agent role name.

        Returns:
            System prompt for the agent.
        """
        # Try file-based prompts first
        file_prompts = _load_agent_prompts()
        if agent in file_prompts:
            entry = file_prompts[agent]
            if isinstance(entry, dict) and "system_prompt" in entry:
                base_prompt = entry["system_prompt"]
                if self._skill_library:
                    skill_context = self._skill_library.build_skill_context(agent)
                    if skill_context:
                        base_prompt += "\n\n" + skill_context
                return base_prompt

        # Hardcoded fallback prompts
        prompts = {
            "planner": """You are a Planning agent. Your role is to:
- Break down complex requests into actionable steps
- Create project roadmaps and timelines
- Identify dependencies and risks
- Prioritize tasks effectively
Respond with clear, structured plans.""",
            "builder": """You are a Builder agent with tool access. Your role is to:
- Write production-ready code
- Implement features and fix bugs
- Follow best practices and coding standards
- Write clean, maintainable, well-documented code

IMPORTANT: Always use your tools before writing code:
1. Use read_file to understand existing code before modifying it
2. Use search_code to find related patterns and conventions
3. Use list_files to understand project structure
4. Use write_file to create or modify files
5. Use run_command to verify your changes (e.g., run tests)
Never guess at file contents — always read first.""",
            "tester": """You are a Tester agent with tool access. Your role is to:
- Create comprehensive test suites
- Identify edge cases and failure scenarios
- Write unit, integration, and e2e tests
- Ensure code coverage and quality

IMPORTANT: Always use your tools:
1. Use read_file to read the code you're testing
2. Use search_code to find existing test patterns
3. Use write_file to create test files
4. Use run_command to run tests and verify they pass
Never write tests without first reading the source code.""",
            "reviewer": """You are a Reviewer agent with tool access. Your role is to:
- Review code for quality and security
- Identify bugs, vulnerabilities, and issues
- Suggest improvements and best practices
- Ensure coding standards compliance

IMPORTANT: Always use your tools:
1. Use read_file to read every file you're reviewing
2. Use search_code to check for patterns across the codebase
3. Use run_command to run linters or tests
Never make claims about code without reading it first.""",
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
            "analyst": """You are an Analyst agent with tool access. Your role is to:
- Analyze code, data, and patterns in the codebase
- Generate insights and metrics
- Inspect files and project structure
- Identify trends, anomalies, and improvement opportunities

IMPORTANT: Always use your tools:
1. Use read_file to read files you're analyzing
2. Use search_code to find patterns across the codebase
3. Use get_project_structure to understand layout
4. Use list_files to enumerate directory contents
Never make claims about file contents without reading them first.""",
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

    async def process_message(
        self,
        message: str,
        context: list[dict[str, str]] | None = None,
        progress_callback: Any = None,
        max_rounds: int = 1,
    ) -> str:
        """Process a user message through intelligent agent delegation.

        This is the main public entry point. Analyzes the message via the
        supervisor LLM, decides whether to delegate to sub-agents or respond
        directly, executes delegations in parallel, and synthesizes results.

        When max_rounds > 1, after the first round the supervisor reviews the
        results and may delegate a verification/fix pass to catch issues.

        Args:
            message: The user's task or question.
            context: Optional conversation history.
            progress_callback: Optional callable(stage, detail) for progress updates.
            max_rounds: Maximum delegation rounds (1 = single pass, 2+ = verify/fix loop).

        Returns:
            Final synthesized response from agent delegation, or direct LLM response.
        """
        if context is None:
            context = []
        if progress_callback is None:

            def progress_callback(stage: str, detail: str = "") -> None:
                pass

        # Build messages for the supervisor LLM call
        system_prompt = self._build_system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        # Add recent context
        for msg in context[-5:]:
            if msg["role"] != "system":
                messages.insert(
                    1,
                    {
                        "role": msg["role"],
                        "content": f"[Context] {msg['content'][:500]}",
                    },
                )

        # Get supervisor's analysis
        progress_callback("analyzing", "Supervisor analyzing task...")
        try:
            response = await self.provider.complete(messages)
        except Exception as e:
            logger.error("Supervisor LLM call failed: %s", e)
            return f"Error: Supervisor could not analyze the task: {e}"

        # Check if supervisor wants to delegate
        plan = self._parse_delegation(response)

        if plan is None:
            # Direct response — supervisor handled it without delegation
            return response

        # Execute delegations
        progress_callback("delegating", plan.analysis)
        results = await self._execute_delegations(plan.delegations, context, progress_callback)

        # Multi-round loop: verify and fix
        for round_num in range(1, max_rounds):
            progress_callback(
                "verifying",
                f"Round {round_num + 1}/{max_rounds}: reviewing results...",
            )
            follow_up = await self._get_follow_up_plan(
                message,
                plan,
                results,
                context,
            )
            if follow_up is None:
                break  # Supervisor satisfied with results

            progress_callback("delegating", f"Round {round_num + 1}: {follow_up.analysis}")
            results = await self._execute_delegations(
                follow_up.delegations,
                context,
                progress_callback,
            )
            plan = follow_up

        # Synthesize results
        progress_callback("synthesizing", "Combining agent results...")
        synthesis = await self._synthesize_results(plan, results, context)

        return synthesis

    async def _get_follow_up_plan(
        self,
        original_message: str,
        plan: DelegationPlan,
        results: dict[str, str],
        context: list[dict[str, str]],
    ) -> DelegationPlan | None:
        """Ask supervisor if follow-up delegation is needed.

        Reviews the results from the first round and decides whether
        to delegate additional work (e.g., verification, fixes).

        Returns:
            A new DelegationPlan if follow-up needed, None if satisfied.
        """
        results_summary = "\n".join(
            f"- {agent}: {result[:500]}" for agent, result in results.items()
        )
        follow_up_prompt = f"""Review the results from the previous round of delegation.

**Original task:** {original_message}

**Previous plan:** {plan.analysis}

**Agent results:**
{results_summary}

If the results are complete and correct, respond with just "SATISFIED".

If there are issues, gaps, or the task needs more work, respond with a new delegation plan in the same JSON format to fix the issues. Use tool-equipped agents (builder, tester, reviewer, analyst) for any follow-up that requires file access."""

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": follow_up_prompt},
        ]

        try:
            response = await self.provider.complete(messages)
        except Exception as e:
            logger.warning("Follow-up analysis failed: %s", e)
            return None

        if "SATISFIED" in response.upper():
            return None

        return self._parse_delegation(response)


class _FilteredToolRegistry:
    """Wraps a ForgeToolRegistry with a filtered tool list.

    Delegates execution to the real registry but only exposes
    tools in the allowed set. Used for per-agent tool isolation.
    """

    def __init__(self, registry: Any, allowed_tools: list[str]):
        self._registry = registry
        self._allowed = set(allowed_tools)

    @property
    def tools(self) -> list:
        """Return only allowed tools."""
        return [t for t in self._registry.tools if t.name in self._allowed]

    def execute(self, tool_name: str, arguments: dict, agent_id: str = "") -> str:
        """Execute a tool if it's in the allowed set."""
        if tool_name not in self._allowed:
            return f"Error: Tool '{tool_name}' is not available for this agent."
        return self._registry.execute(tool_name, arguments, agent_id=agent_id)
