"""Core orchestration engine for Forge workflows."""

from pathlib import Path

from animus.cognitive import CognitiveLayer
from animus.forge.agent import ForgeAgent
from animus.forge.budget import BudgetTracker
from animus.forge.checkpoint import CheckpointStore
from animus.forge.gates import evaluate_gate
from animus.forge.models import (
    BudgetExhaustedError,
    ForgeError,
    GateFailedError,
    ReviseRequestedError,
    StepResult,
    WorkflowConfig,
    WorkflowState,
)
from animus.logging import get_logger
from animus.tools import ToolRegistry

logger = get_logger("forge.engine")


class ForgeEngine:
    """Sequential workflow execution engine.

    Executes agents in order, enforces budgets, evaluates quality gates,
    and persists state via SQLite checkpoints.
    """

    def __init__(
        self,
        cognitive: CognitiveLayer,
        checkpoint_dir: Path | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.cognitive = cognitive
        self.tools = tools
        self._checkpoint: CheckpointStore | None = None

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            db_path = checkpoint_dir / "forge_checkpoints.db"
            self._checkpoint = CheckpointStore(db_path)

    def run(
        self,
        config: WorkflowConfig,
        resume: bool = False,
    ) -> WorkflowState:
        """Execute a workflow sequentially.

        Args:
            config: Workflow configuration.
            resume: If True, resume from last checkpoint.

        Returns:
            Final WorkflowState with all results.

        Raises:
            BudgetExhaustedError: If any agent or the workflow exceeds budget.
            GateFailedError: If a gate fails with on_fail='halt'.
            ForgeError: On other errors.
        """
        state = self._init_state(config, resume)
        budget = BudgetTracker.from_config(config)

        # Restore budget from prior results if resuming
        for result in state.results:
            if result.success:
                budget.agent_usage[result.agent_name] = (
                    budget.agent_usage.get(result.agent_name, 0) + result.tokens_used
                )
                budget.total_cost += result.cost_usd

        state.status = "running"
        self._save_checkpoint(state)

        logger.info(
            f"Starting workflow {config.name!r} from step {state.current_step}/{len(config.agents)}"
        )

        # Collect all outputs for gate evaluation
        all_outputs = self._collect_outputs(state.results)

        try:
            step_idx = state.current_step
            while step_idx < len(config.agents):
                agent_config = config.agents[step_idx]

                # Check budget before running
                if not budget.check(agent_config.name):
                    raise BudgetExhaustedError(
                        f"Agent {agent_config.name!r} has no remaining budget"
                    )

                # Resolve inputs from prior results
                inputs = self._resolve_inputs(agent_config.inputs, all_outputs)

                # Inject gate feedback if present
                feedback_key = f"_gate_feedback.{agent_config.name}"
                if feedback_key in all_outputs:
                    inputs["_gate_feedback"] = all_outputs.pop(feedback_key)

                # Execute agent
                agent = ForgeAgent(agent_config, self.cognitive, self.tools)
                result = agent.run(inputs)

                if not result.success:
                    state.status = "failed"
                    state.results.append(result)
                    state.current_step = step_idx + 1
                    self._save_checkpoint(state)
                    raise ForgeError(f"Agent {agent_config.name!r} failed: {result.error}")

                # Record budget
                budget.record(agent_config.name, result.tokens_used, result.cost_usd)

                # Update state
                state.results.append(result)
                state.total_tokens += result.tokens_used
                state.total_cost += result.cost_usd
                state.current_step = step_idx + 1

                # Update outputs
                for key, val in result.outputs.items():
                    all_outputs[f"{agent_config.name}.{key}"] = val
                    all_outputs[key] = val

                self._save_checkpoint(state)

                # Evaluate gates — may raise ReviseRequestedError
                try:
                    self._evaluate_gates(config, agent_config.name, all_outputs, state)
                except ReviseRequestedError as req:
                    step_idx = self._handle_revise(
                        req,
                        config,
                        state,
                        all_outputs,
                    )
                    continue

                logger.info(
                    f"Step {step_idx}/{len(config.agents)}: "
                    f"{agent_config.name} complete "
                    f"({result.tokens_used} tokens)"
                )

                step_idx += 1

        except (BudgetExhaustedError, GateFailedError):
            state.status = "failed"
            self._save_checkpoint(state)
            raise
        except ForgeError:
            state.status = "failed"
            self._save_checkpoint(state)
            raise
        except Exception as exc:
            state.status = "failed"
            self._save_checkpoint(state)
            raise ForgeError(f"Unexpected error: {exc}") from exc

        state.status = "completed"
        self._save_checkpoint(state)

        logger.info(
            f"Workflow {config.name!r} completed: "
            f"{len(state.results)} steps, "
            f"{state.total_tokens} tokens, "
            f"${state.total_cost:.4f}"
        )

        return state

    def pause(self, workflow_name: str) -> None:
        """Mark a workflow as paused in the checkpoint store."""
        if not self._checkpoint:
            raise ForgeError("No checkpoint store configured")

        state = self._checkpoint.load_state(workflow_name)
        if state is None:
            raise ForgeError(f"No checkpoint found for {workflow_name!r}")

        state.status = "paused"
        self._checkpoint.save_state(state)
        logger.info(f"Paused workflow {workflow_name!r} at step {state.current_step}")

    def status(self, workflow_name: str) -> WorkflowState | None:
        """Get the current state of a workflow."""
        if not self._checkpoint:
            return None
        return self._checkpoint.load_state(workflow_name)

    def list_workflows(self) -> list[tuple[str, str, int]]:
        """List all checkpointed workflows."""
        if not self._checkpoint:
            return []
        return self._checkpoint.list_workflows()

    def _init_state(
        self,
        config: WorkflowConfig,
        resume: bool,
    ) -> WorkflowState:
        """Initialize or resume workflow state."""
        if resume and self._checkpoint:
            state = self._checkpoint.load_state(config.name)
            if state is not None:
                logger.info(f"Resuming {config.name!r} from step {state.current_step}")
                return state

        return WorkflowState(workflow_name=config.name)

    def _save_checkpoint(self, state: WorkflowState) -> None:
        """Save state to checkpoint if store is configured."""
        if self._checkpoint:
            self._checkpoint.save_state(state)

    def _resolve_inputs(
        self,
        input_refs: list[str],
        all_outputs: dict[str, str],
    ) -> dict[str, str]:
        """Resolve input references to actual values."""
        inputs: dict[str, str] = {}
        for ref in input_refs:
            if ref in all_outputs:
                inputs[ref] = all_outputs[ref]
            else:
                logger.warning(f"Input {ref!r} not found in outputs")
        return inputs

    def _collect_outputs(
        self,
        results: list[StepResult],
    ) -> dict[str, str]:
        """Collect all outputs from completed steps into a flat dict."""
        all_outputs: dict[str, str] = {}
        for result in results:
            for key, val in result.outputs.items():
                all_outputs[f"{result.agent_name}.{key}"] = val
                all_outputs[key] = val
        return all_outputs

    def _evaluate_gates(
        self,
        config: WorkflowConfig,
        agent_name: str,
        all_outputs: dict[str, str],
        state: WorkflowState,
    ) -> None:
        """Evaluate all gates that follow the given agent."""
        for gate in config.gates:
            if gate.after != agent_name:
                continue

            passed, reason = evaluate_gate(gate, all_outputs)

            if passed:
                logger.debug(f"Gate {gate.name!r} passed")
                continue

            logger.warning(f"Gate {gate.name!r} failed: {reason}")

            if gate.on_fail == "skip":
                logger.info(f"Gate {gate.name!r}: skipping per on_fail policy")
                continue
            elif gate.on_fail == "halt":
                raise GateFailedError(reason)
            elif gate.on_fail == "revise":
                raise ReviseRequestedError(
                    target=gate.revise_target,
                    gate_name=gate.name,
                    reason=reason,
                    max_revisions=gate.max_revisions,
                )

    def _handle_revise(
        self,
        req: ReviseRequestedError,
        config: WorkflowConfig,
        state: WorkflowState,
        all_outputs: dict[str, str],
    ) -> int:
        """Handle a revise request by looping back to the target agent.

        Returns the step index to resume from.

        Raises:
            GateFailedError: If max revisions exceeded.
        """
        gate_key = req.gate_name
        count = state.revision_counts.get(gate_key, 0) + 1

        if count > req.max_revisions:
            raise GateFailedError(
                f"Max revisions ({req.max_revisions}) exceeded for gate {gate_key!r}: {req.reason}"
            )

        state.revision_counts[gate_key] = count

        # Find target index
        target_idx = next(i for i, a in enumerate(config.agents) if a.name == req.target)

        # Clear downstream results (keep only agents before target)
        keep_agents = {config.agents[j].name for j in range(target_idx)}
        state.results = [r for r in state.results if r.agent_name in keep_agents]

        # Rebuild outputs from remaining results
        all_outputs.clear()
        all_outputs.update(self._collect_outputs(state.results))

        # Inject gate feedback for the target agent
        all_outputs[f"_gate_feedback.{req.target}"] = (
            f"Gate {req.gate_name!r} failed: {req.reason}. "
            f"Please revise your output. (attempt {count}/{req.max_revisions})"
        )

        # Reset step pointer
        state.current_step = target_idx
        self._save_checkpoint(state)

        logger.info(
            f"Revise loop: gate {req.gate_name!r} -> re-running from "
            f"{req.target!r} (attempt {count}/{req.max_revisions})"
        )

        return target_idx
