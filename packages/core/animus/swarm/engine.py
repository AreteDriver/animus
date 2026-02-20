"""Parallel orchestration engine for Swarm workflows."""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from animus.cognitive import CognitiveLayer
from animus.forge.agent import ForgeAgent
from animus.forge.budget import BudgetTracker
from animus.forge.checkpoint import CheckpointStore
from animus.forge.gates import evaluate_gate
from animus.forge.models import (
    AgentConfig,
    BudgetExhaustedError,
    ForgeError,
    GateFailedError,
    ReviseRequestedError,
    StepResult,
    WorkflowConfig,
    WorkflowState,
)
from animus.logging import get_logger
from animus.swarm.graph import build_dag, derive_stages
from animus.swarm.intent import IntentGraph, IntentResolver
from animus.swarm.models import IntentEntry, SwarmConfig, SwarmError, SwarmStage
from animus.tools import ToolRegistry

logger = get_logger("swarm.engine")


class SwarmEngine:
    """Parallel workflow execution engine with stigmergic coordination.

    Analyzes agent dependencies to build a DAG, derives parallel execution
    stages, and runs independent agents within each stage concurrently
    using :class:`~concurrent.futures.ThreadPoolExecutor`.
    """

    def __init__(
        self,
        cognitive: CognitiveLayer,
        checkpoint_dir: Path | None = None,
        tools: ToolRegistry | None = None,
        swarm_config: SwarmConfig | None = None,
    ):
        self.cognitive = cognitive
        self.tools = tools
        self._config = swarm_config or SwarmConfig()
        self._intent_graph = IntentGraph()
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
        """Execute a workflow with parallel stages.

        Args:
            config: Workflow configuration.
            resume: If True, resume from last checkpoint.

        Returns:
            Final WorkflowState with all results.

        Raises:
            BudgetExhaustedError: If any agent exceeds budget.
            GateFailedError: If a gate fails with on_fail='halt'.
            SwarmError: On DAG or coordination errors.
            ForgeError: On other errors.
        """
        # 1. Build DAG and derive stages
        dag = build_dag(config.agents)
        stages = derive_stages(config.agents, dag)

        logger.info(f"Workflow {config.name!r}: {len(stages)} stages, {len(config.agents)} agents")
        for stage in stages:
            logger.info(f"  Stage {stage.index}: {stage.agent_names}")

        # 2. Initialize state
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

        # 3. Collect outputs from prior results (for resume)
        all_outputs = self._collect_outputs(state.results)
        completed_agents = {r.agent_name for r in state.results if r.success}

        # 4. Determine starting stage
        start_stage = self._find_resume_stage(stages, completed_agents)

        # 5. Clear intent graph for fresh coordination
        self._intent_graph.clear()

        # 6. Execute stages
        agent_configs = {a.name: a for a in config.agents}

        try:
            stage_idx = start_stage
            while stage_idx < len(stages):
                stage = stages[stage_idx]
                agents_to_run = [name for name in stage.agent_names if name not in completed_agents]

                if not agents_to_run:
                    logger.info(f"Stage {stage.index}: all agents already complete")
                    stage_idx += 1
                    continue

                # Pre-check budget for all agents in stage
                for agent_name in agents_to_run:
                    if not budget.check(agent_name):
                        raise BudgetExhaustedError(f"Agent {agent_name!r} has no remaining budget")

                # Publish intents
                available_keys = set(all_outputs.keys())
                self._publish_stage_intents(agents_to_run, agent_configs, available_keys)

                # Execute agents in parallel
                results = self._execute_stage(agents_to_run, agent_configs, all_outputs)

                # Process results
                for result in results:
                    if not result.success:
                        state.status = "failed"
                        state.results.append(result)
                        self._save_checkpoint(state)
                        raise ForgeError(f"Agent {result.agent_name!r} failed: {result.error}")

                    budget.record(result.agent_name, result.tokens_used, result.cost_usd)
                    state.results.append(result)
                    state.total_tokens += result.tokens_used
                    state.total_cost += result.cost_usd
                    completed_agents.add(result.agent_name)

                    # Update outputs
                    for key, val in result.outputs.items():
                        all_outputs[f"{result.agent_name}.{key}"] = val
                        all_outputs[key] = val

                    # Mark intent as completed
                    intent = self._intent_graph.get(result.agent_name)
                    if intent:
                        intent.stability = 1.0
                        intent.status = "completed"
                        self._intent_graph.publish(intent)

                # Stage-level checkpoint
                state.current_step = sum(len(s.agent_names) for s in stages[: stage.index + 1])
                self._save_checkpoint(state)

                # Evaluate gates — may raise ReviseRequestedError
                revise_restart = None
                for agent_name in agents_to_run:
                    try:
                        self._evaluate_gates(config, agent_name, all_outputs, state)
                    except ReviseRequestedError as req:
                        revise_restart = self._handle_revise(
                            req,
                            config,
                            state,
                            stages,
                            all_outputs,
                            completed_agents,
                        )
                        break

                if revise_restart is not None:
                    stage_idx = revise_restart
                    continue

                logger.info(
                    f"Stage {stage.index} complete: "
                    f"{len(agents_to_run)} agents, "
                    f"{sum(r.tokens_used for r in results)} tokens"
                )

                stage_idx += 1

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
            raise SwarmError(f"Unexpected error: {exc}") from exc

        state.status = "completed"
        self._save_checkpoint(state)

        logger.info(
            f"Workflow {config.name!r} completed: "
            f"{len(stages)} stages, {len(state.results)} agents, "
            f"{state.total_tokens} tokens, ${state.total_cost:.4f}"
        )
        return state

    def _publish_stage_intents(
        self,
        agent_names: list[str],
        agent_configs: dict[str, AgentConfig],
        available_keys: set[str],
    ) -> None:
        """Publish intents for all agents in a stage."""
        for agent_name in agent_names:
            ac = agent_configs[agent_name]
            stability = IntentResolver.compute_stability(ac, available_keys)
            intent = IntentEntry(
                agent=agent_name,
                action="execute_step",
                provides=[f"{agent_name}.{o}" for o in ac.outputs],
                requires=list(ac.inputs),
                stability=stability,
                status="pending",
            )
            conflicts = self._intent_graph.find_conflicts(intent)
            if conflicts:
                resolution = IntentResolver.resolve(intent, conflicts)
                logger.warning(
                    f"Conflict for {agent_name}: provides overlap "
                    f"with {[c.agent for c in conflicts]}, "
                    f"resolution={resolution}"
                )
                intent.evidence.append(f"conflict_resolved:{resolution}")
            self._intent_graph.publish(intent)

    def _execute_stage(
        self,
        agent_names: list[str],
        agent_configs: dict[str, AgentConfig],
        all_outputs: dict[str, str],
    ) -> list[StepResult]:
        """Execute all agents in a stage in parallel.

        Single-agent stages skip the thread pool for efficiency.
        """
        if len(agent_names) == 1:
            ac = agent_configs[agent_names[0]]
            inputs = self._resolve_inputs(ac.inputs, all_outputs)
            feedback_key = f"_gate_feedback.{agent_names[0]}"
            if feedback_key in all_outputs:
                inputs["_gate_feedback"] = all_outputs.pop(feedback_key)
            agent = ForgeAgent(ac, self.cognitive, self.tools)
            return [agent.run(inputs)]

        results: dict[str, StepResult] = {}

        with ThreadPoolExecutor(
            max_workers=min(self._config.max_workers, len(agent_names))
        ) as executor:
            futures: dict[Future, str] = {}
            for name in agent_names:
                ac = agent_configs[name]
                inputs = self._resolve_inputs(ac.inputs, all_outputs)
                feedback_key = f"_gate_feedback.{name}"
                if feedback_key in all_outputs:
                    inputs["_gate_feedback"] = all_outputs.pop(feedback_key)
                agent = ForgeAgent(ac, self.cognitive, self.tools)
                future = executor.submit(agent.run, inputs)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=self._config.stage_timeout_seconds)
                except Exception as exc:
                    result = StepResult(
                        agent_name=name,
                        success=False,
                        error=f"Thread execution error: {exc}",
                    )
                results[name] = result

        # Return in original order for determinism
        return [results[name] for name in agent_names]

    # --- Helper methods ---

    def _init_state(self, config: WorkflowConfig, resume: bool) -> WorkflowState:
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

    def _collect_outputs(self, results: list[StepResult]) -> dict[str, str]:
        """Collect all outputs from completed steps into a flat dict."""
        all_outputs: dict[str, str] = {}
        for result in results:
            for key, val in result.outputs.items():
                all_outputs[f"{result.agent_name}.{key}"] = val
                all_outputs[key] = val
        return all_outputs

    def _find_resume_stage(
        self,
        stages: list[SwarmStage],
        completed_agents: set[str],
    ) -> int:
        """Find the first stage with incomplete agents."""
        for stage in stages:
            if not all(name in completed_agents for name in stage.agent_names):
                return stage.index
        return len(stages)

    def _evaluate_gates(
        self,
        config: WorkflowConfig,
        agent_name: str,
        all_outputs: dict[str, str],
        state: WorkflowState,
    ) -> None:
        """Evaluate gates after an agent completes."""
        for gate in config.gates:
            if gate.after != agent_name:
                continue

            passed, reason = evaluate_gate(gate, all_outputs)

            if passed:
                logger.debug(f"Gate {gate.name!r} passed")
                continue

            logger.warning(f"Gate {gate.name!r} failed: {reason}")

            if gate.on_fail == "skip":
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
        stages: list[SwarmStage],
        all_outputs: dict[str, str],
        completed_agents: set[str],
    ) -> int:
        """Handle a revise request by looping back to the target agent's stage.

        Returns the stage index to resume from.

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

        # Find the stage containing the revise target
        target_stage_idx = next(s.index for s in stages if req.target in s.agent_names)

        # Clear results for agents in target stage and later stages
        agents_to_clear = set()
        for s in stages[target_stage_idx:]:
            agents_to_clear.update(s.agent_names)

        state.results = [r for r in state.results if r.agent_name not in agents_to_clear]
        completed_agents -= agents_to_clear

        # Rebuild outputs from remaining results
        all_outputs.clear()
        all_outputs.update(self._collect_outputs(state.results))

        # Inject gate feedback for the target agent
        all_outputs[f"_gate_feedback.{req.target}"] = (
            f"Gate {req.gate_name!r} failed: {req.reason}. "
            f"Please revise your output. (attempt {count}/{req.max_revisions})"
        )

        # Reset step counter
        state.current_step = sum(len(s.agent_names) for s in stages[:target_stage_idx])
        self._save_checkpoint(state)

        logger.info(
            f"Revise loop: gate {req.gate_name!r} -> re-running from stage "
            f"{target_stage_idx} ({req.target!r}) "
            f"(attempt {count}/{req.max_revisions})"
        )

        return target_stage_idx

    # --- Parity with ForgeEngine ---

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
