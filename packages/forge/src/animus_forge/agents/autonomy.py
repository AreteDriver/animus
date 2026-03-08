"""Agent autonomy loop: observe → plan → act → reflect.

Provides a self-correcting execution cycle where agents can
observe their environment, plan next actions, execute them,
and reflect on results to decide whether to continue or stop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class LoopPhase(StrEnum):
    """Current phase in the autonomy cycle."""

    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    STOPPED = "stopped"


class StopReason(StrEnum):
    """Why the autonomy loop terminated."""

    GOAL_ACHIEVED = "goal_achieved"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ERROR_THRESHOLD = "error_threshold"
    MANUAL_STOP = "manual_stop"
    REFLECTION_HALT = "reflection_halt"


@dataclass
class LoopIteration:
    """Record of a single autonomy loop cycle.

    Attributes:
        iteration: Cycle number (1-indexed).
        phase: Last phase completed.
        observation: What the agent observed.
        plan: What the agent planned.
        action_result: Result of the action.
        reflection: Agent's self-assessment.
        tokens_used: Tokens consumed this iteration.
        duration_ms: Wall-clock time for this iteration.
    """

    iteration: int
    phase: LoopPhase
    observation: str = ""
    plan: str = ""
    action_result: str = ""
    reflection: str = ""
    tokens_used: int = 0
    duration_ms: int = 0


@dataclass
class AutonomyResult:
    """Final result of an autonomy loop execution.

    Attributes:
        goal: Original goal description.
        stop_reason: Why the loop stopped.
        iterations: All iteration records.
        final_output: The last action result or synthesis.
        total_tokens: Total tokens across all iterations.
        total_duration_ms: Total wall-clock time.
    """

    goal: str
    stop_reason: StopReason = StopReason.MAX_ITERATIONS
    iterations: list[LoopIteration] = field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    total_duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal": self.goal,
            "stop_reason": self.stop_reason.value,
            "iteration_count": len(self.iterations),
            "final_output": self.final_output[:2000],
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
        }


class AutonomyLoop:
    """Self-correcting agent execution cycle.

    Runs an observe → plan → act → reflect loop where each phase
    is driven by an LLM provider. The reflect phase decides whether
    to continue or stop based on goal progress.

    Args:
        provider: LLM provider for generating responses.
        max_iterations: Maximum loop cycles before forced stop.
        max_errors: Maximum consecutive errors before halt.
        budget_manager: Optional budget tracker for token limits.
        message_bus: Optional message bus for publishing loop events.
    """

    # Prompt templates for each phase
    OBSERVE_PROMPT = (
        "You are an autonomous agent working toward a goal.\n"
        "Goal: {goal}\n\n"
        "Current state:\n{state}\n\n"
        "Previous actions:\n{history}\n\n"
        "Observe the current situation. What is the state of progress? "
        "What information do you have? What gaps exist? "
        "Respond with a concise observation."
    )

    PLAN_PROMPT = (
        "You are an autonomous agent working toward a goal.\n"
        "Goal: {goal}\n\n"
        "Observation: {observation}\n\n"
        "Based on this observation, what is the single most important "
        "next action to take? Be specific and actionable. "
        "Respond with a clear, concise plan for one action."
    )

    REFLECT_PROMPT = (
        "You are an autonomous agent evaluating your progress.\n"
        "Goal: {goal}\n\n"
        "Action taken: {plan}\n"
        "Result: {result}\n\n"
        "Iteration {iteration} of {max_iterations}.\n\n"
        "Evaluate: Is the goal achieved? Should you continue? "
        "Respond with JSON: "
        '{{"assessment": "...", "goal_achieved": true/false, '
        '"should_continue": true/false, "reason": "..."}}'
    )

    def __init__(
        self,
        provider: Any,
        max_iterations: int = 10,
        max_errors: int = 3,
        budget_manager: Any = None,
        message_bus: Any = None,
    ):
        self._provider = provider
        self._max_iterations = max_iterations
        self._max_errors = max_errors
        self._budget_manager = budget_manager
        self._message_bus = message_bus
        self._stopped = False

    @property
    def max_iterations(self) -> int:
        """Maximum allowed iterations."""
        return self._max_iterations

    def stop(self) -> None:
        """Request the loop to stop after the current iteration."""
        self._stopped = True

    async def run(
        self,
        goal: str,
        initial_state: str = "",
        act_fn: Any = None,
    ) -> AutonomyResult:
        """Execute the autonomy loop.

        Args:
            goal: The goal to achieve.
            initial_state: Starting context/state description.
            act_fn: Optional async callable(plan: str) -> str for the act phase.
                If not provided, the provider generates the action result.

        Returns:
            AutonomyResult with all iteration records.
        """
        result = AutonomyResult(goal=goal)
        state = initial_state
        history_lines: list[str] = []
        consecutive_errors = 0
        loop_start = time.time()

        self._publish_event("autonomy.started", {"goal": goal[:200]})

        for i in range(1, self._max_iterations + 1):
            if self._stopped:
                result.stop_reason = StopReason.MANUAL_STOP
                break

            # Budget check
            if self._budget_manager is not None:
                if not self._budget_manager.can_allocate(1000):
                    result.stop_reason = StopReason.BUDGET_EXHAUSTED
                    break

            iteration = LoopIteration(iteration=i, phase=LoopPhase.OBSERVE)
            iter_start = time.time()
            iter_tokens = 0

            try:
                # OBSERVE
                observation = await self._generate(
                    self.OBSERVE_PROMPT.format(
                        goal=goal,
                        state=state,
                        history="\n".join(history_lines[-5:]) or "(none)",
                    )
                )
                iteration.observation = observation
                iter_tokens += self._estimate_tokens(observation)

                # PLAN
                iteration.phase = LoopPhase.PLAN
                plan = await self._generate(
                    self.PLAN_PROMPT.format(
                        goal=goal,
                        observation=observation,
                    )
                )
                iteration.plan = plan
                iter_tokens += self._estimate_tokens(plan)

                # ACT
                iteration.phase = LoopPhase.ACT
                if act_fn is not None:
                    action_result = await act_fn(plan)
                else:
                    action_result = await self._generate(
                        f"Execute this plan:\n{plan}\n\n"
                        f"Context:\n{state}\n\n"
                        f"Provide the result of executing this action."
                    )
                iteration.action_result = action_result
                iter_tokens += self._estimate_tokens(action_result)

                # REFLECT
                iteration.phase = LoopPhase.REFLECT
                reflection = await self._generate(
                    self.REFLECT_PROMPT.format(
                        goal=goal,
                        plan=plan,
                        result=action_result[:1000],
                        iteration=i,
                        max_iterations=self._max_iterations,
                    )
                )
                iteration.reflection = reflection
                iter_tokens += self._estimate_tokens(reflection)

                # Parse reflection for stop decision
                should_stop, stop_reason = self._parse_reflection(reflection)

                # Update state for next iteration
                state = action_result
                history_lines.append(f"[{i}] {plan[:100]} → {action_result[:100]}")
                consecutive_errors = 0

                iteration.tokens_used = iter_tokens
                iteration.duration_ms = int((time.time() - iter_start) * 1000)
                result.iterations.append(iteration)
                result.total_tokens += iter_tokens

                self._publish_event(
                    "autonomy.iteration",
                    {"iteration": i, "phase": "reflect", "tokens": iter_tokens},
                )

                if should_stop:
                    result.stop_reason = stop_reason
                    result.final_output = action_result
                    break

            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    "Autonomy loop error (iteration %d, consecutive %d): %s",
                    i,
                    consecutive_errors,
                    e,
                )
                iteration.phase = LoopPhase.STOPPED
                iteration.reflection = f"Error: {e}"
                iteration.duration_ms = int((time.time() - iter_start) * 1000)
                result.iterations.append(iteration)

                if consecutive_errors >= self._max_errors:
                    result.stop_reason = StopReason.ERROR_THRESHOLD
                    break
        else:
            # Loop exhausted max_iterations without breaking
            result.stop_reason = StopReason.MAX_ITERATIONS

        if not result.stop_reason:
            result.stop_reason = StopReason.MAX_ITERATIONS

        if not result.final_output and result.iterations:
            last = result.iterations[-1]
            result.final_output = last.action_result or last.observation

        result.total_duration_ms = int((time.time() - loop_start) * 1000)

        self._publish_event(
            "autonomy.completed",
            {
                "goal": goal[:200],
                "stop_reason": result.stop_reason.value,
                "iterations": len(result.iterations),
            },
        )

        return result

    async def _generate(self, prompt: str) -> str:
        """Generate a response from the provider.

        Args:
            prompt: The prompt text.

        Returns:
            Provider response text.
        """
        response = await self._provider.complete(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _parse_reflection(self, reflection: str) -> tuple[bool, StopReason]:
        """Parse reflection text for stop decision.

        Looks for JSON with goal_achieved/should_continue flags.
        Falls back to keyword detection.

        Args:
            reflection: Reflection text from the LLM.

        Returns:
            Tuple of (should_stop, stop_reason).
        """
        import json
        import re

        # Try JSON extraction
        json_match = re.search(r"\{[^}]+\}", reflection)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if data.get("goal_achieved", False):
                    return True, StopReason.GOAL_ACHIEVED
                if not data.get("should_continue", True):
                    return True, StopReason.REFLECTION_HALT
                return False, StopReason.MAX_ITERATIONS
            except (json.JSONDecodeError, KeyError):
                pass

        # Keyword fallback
        lower = reflection.lower()
        if "goal achieved" in lower or "goal is achieved" in lower:
            return True, StopReason.GOAL_ACHIEVED
        if "should stop" in lower or "no further action" in lower:
            return True, StopReason.REFLECTION_HALT

        return False, StopReason.MAX_ITERATIONS

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return max(len(text) // 4, 1)

    def _publish_event(self, topic: str, payload: dict) -> None:
        """Publish to message bus if available."""
        if self._message_bus is not None:
            try:
                self._message_bus.publish(
                    topic=topic,
                    sender="autonomy_loop",
                    payload=payload,
                )
            except Exception:
                pass  # Never break the loop for messaging
