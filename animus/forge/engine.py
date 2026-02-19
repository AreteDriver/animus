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
            for i in range(state.current_step, len(config.agents)):
                agent_config = config.agents[i]

                # Check budget before running
                if not budget.check(agent_config.name):
                    raise BudgetExhaustedError(
                        f"Agent {agent_config.name!r} has no remaining budget"
                    )

                # Resolve inputs from prior results
                inputs = self._resolve_inputs(agent_config.inputs, all_outputs)

                # Execute agent
                agent = ForgeAgent(agent_config, self.cognitive, self.tools)
                result = agent.run(inputs)

                if not result.success:
                    state.status = "failed"
                    state.results.append(result)
                    state.current_step = i + 1
                    self._save_checkpoint(state)
                    raise ForgeError(f"Agent {agent_config.name!r} failed: {result.error}")

                # Record budget
                budget.record(agent_config.name, result.tokens_used, result.cost_usd)

                # Update state
                state.results.append(result)
                state.total_tokens += result.tokens_used
                state.total_cost += result.cost_usd
                state.current_step = i + 1

                # Update outputs
                for key, val in result.outputs.items():
                    all_outputs[f"{agent_config.name}.{key}"] = val
                    all_outputs[key] = val

                self._save_checkpoint(state)

                # Evaluate gates after this agent
                self._evaluate_gates(config, agent_config.name, all_outputs, state)

                logger.info(
                    f"Step {i + 1}/{len(config.agents)}: "
                    f"{agent_config.name} complete "
                    f"({result.tokens_used} tokens)"
                )

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
                # For MVP, revise is treated as halt with feedback.
                # Full loop-back requires re-running from revise_target,
                # which is a Phase 2 enhancement.
                raise GateFailedError(f"{reason} (revise requested but not yet implemented)")
