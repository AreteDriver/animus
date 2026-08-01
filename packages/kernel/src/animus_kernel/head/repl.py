"""Animus Head REPL — local-first agentic conversation loop.

Runs an iterative turn-by-turn loop with a local Ollama model,
auto-loading context, executing tools, and persisting state.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore
from animus_kernel.head.context_manager import HeadContextManager
from animus_kernel.head.fallback_controller import HeadFallbackController
from animus_kernel.head.intent_parser import HeadIntentParser, IntentType
from animus_kernel.head.planner import HeadPlanner
from animus_kernel.head.quality_gate import HeadQualityGate
from animus_kernel.head.session_bootstrap import SessionBootstrap
from animus_kernel.head.session_controller import (
    SessionController,
    SessionLifecycleEvent,
    SessionPolicy,
)
from animus_kernel.head.synthesizer import HeadSynthesizer
from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.head.tool_validator import RetryableToolExecutor
from animus_kernel.providers.base import CompletionRequest
from animus_kernel.providers.manager import ProviderManager
from animus_kernel.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class HeadREPL:
    """Persistent REPL for Animus Head.

    Args:
        model: Ollama model to use (default: qwen2.5:32b)
        project_root: Project directory for filesystem tools
        memory_dir: Directory for semantic memory store
        checkpoint_store: SQLite checkpoint store
        system_prompt: Override system prompt (default: loaded from prompts/)
        max_turns: Safety limit per session (default: 1000)
        checkpoint_every: Turns between auto-checkpoints (default: 5)
        fallback_enabled: Allow cloud fallback on quality failures (default: False)
        fallback_provider: Cloud provider name for fallback (default: anthropic)
        max_fallbacks_per_session: Hard cap on cloud calls per session
        auto_execute_direct: Whether to fast-path direct commands without model call
        session_timer: Max wall-clock duration per session (default: None — disabled)
        wrapup_threshold: Token utilization fraction (0.0–1.0) that triggers
            graceful finalize. 1.0 disables token-based wrap-up (default: 1.0)
        session_controller: Optional SessionController instance. If None and
            session_timer or wrapup_threshold is set, one is created automatically.
    """

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        project_root: str | Path | None = None,
        memory_dir: str | Path | None = None,
        checkpoint_store: HeadCheckpointStore | None = None,
        system_prompt: str | None = None,
        max_turns: int = 1000,
        checkpoint_every: int = 5,
        fallback_enabled: bool = False,
        fallback_provider: str = "anthropic",
        max_fallbacks_per_session: int = 10,
        auto_execute_direct: bool = True,
        session_timer: timedelta | None = None,
        wrapup_threshold: float = 1.0,
        session_controller: SessionController | None = None,
    ) -> None:
        self.model = model
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.session_id = self._generate_session_id()
        self.max_turns = max_turns
        self.checkpoint_every = checkpoint_every
        self.auto_execute_direct = auto_execute_direct

        # Provider
        self.provider = OllamaProvider(model=model)
        if not self.provider.is_configured():
            raise RuntimeError(
                f"Ollama is not running or model '{model}' is not available. "
                "Start Ollama and pull the model first."
            )

        # Subsystems
        self.checkpoint_store = checkpoint_store or HeadCheckpointStore()
        self.session_bootstrap = SessionBootstrap(
            project_root=self.project_root,
            memory_dir=memory_dir,
            checkpoint_store=self.checkpoint_store,
        )
        self.tools = HeadToolOrchestrator(
            project_root=self.project_root,
            memory_dir=memory_dir,
            enable_shell=True,
        )

        # Retryable executor with schema validation
        self._retryable = RetryableToolExecutor(
            orchestrator=self.tools,
            registry=self.tools._forge,
            max_retries=3,
        )

        # Context manager with token budgeting
        self.context = HeadContextManager(model=model)

        # Quality gate and cloud fallback
        self._quality_gate = HeadQualityGate(max_failure_streak=3)
        self._fallback = HeadFallbackController(
            provider_manager=ProviderManager(),
            fallback_provider=fallback_provider,
            enabled=fallback_enabled,
            max_fallbacks_per_session=max_fallbacks_per_session,
        )

        # Phase 5: Natural language interface
        self._intent_parser = HeadIntentParser()
        self._planner = HeadPlanner()
        self._synthesizer = HeadSynthesizer()

        # Load or build system prompt
        if system_prompt is None:
            prompt_path = Path(__file__).parent / "prompts" / "system_repl.md"
            if prompt_path.exists():
                system_prompt = prompt_path.read_text()
            else:
                system_prompt = "You are Animus Head — a local-first agentic assistant."
        self.system_prompt = system_prompt

        # Conversation state
        self.turns = 0
        self.total_tokens = 0

        # Per-model performance telemetry: model_name -> {calls, latency_ms, tokens}
        self._model_telemetry: dict[str, dict] = {}

        # Session lifecycle management
        self._session_started_at: datetime | None = None
        self._session_timer = session_timer
        self._wrapup_threshold = wrapup_threshold
        self._session_wrapped_up = False
        self._session_controller = session_controller
        if session_controller is None and (session_timer is not None or wrapup_threshold < 1.0):
            policy = SessionPolicy(
                wrapup_threshold=wrapup_threshold,
                session_timer=session_timer or timedelta.max,
                auto_restart=True,
            )
            self._session_controller = SessionController(policy=policy)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Bootstrap session context and initialize message history.

        Called automatically by `start()`, but can be invoked directly
        for testing or programmatic use.
        """
        if self._session_started_at is None:
            self._session_started_at = datetime.now(UTC)

        context = self.session_bootstrap.bootstrap()
        system = self.session_bootstrap.build_system_prompt(context)

        # Append the base system prompt
        full_system = f"{self.system_prompt}\n\n{system}"

        # Check for previous session
        prev = context.get("previous_session")
        if prev and isinstance(prev, dict) and prev.get("messages"):
            filtered = self._filter_messages(prev["messages"])
            self.turns = prev.get("turns", 0)
            self.total_tokens = prev.get("total_tokens", 0)
            # Restore model if present and still available
            prev_model = prev.get("model", "")
            if prev_model and prev_model != self.model:
                try:
                    installed = self.provider.list_models()
                    if prev_model in installed:
                        print(f"   📥 Restoring previous model: {prev_model}")
                        self._swap_model(prev_model)
                    else:
                        print(f"   ⚠ Previous model '{prev_model}' not installed, using default.")
                except Exception:
                    pass  # Non-critical: keep default model
            # Restore summary if present
            if prev.get("summary"):
                self.context.set_summary(prev["summary"])

            # Replace the first system message with fresh context
            if filtered and filtered[0].get("role") == "system":
                filtered[0] = {"role": "system", "content": full_system}
            else:
                filtered.insert(0, {"role": "system", "content": full_system})

            for msg in filtered:
                self.context.add_message(msg)
        else:
            self.context.add_message({"role": "system", "content": full_system})

    def start(self) -> None:
        """Bootstrap and enter the REPL loop."""
        print("\n🧠 Animus Head — local-first agentic loop")
        print(f"   Model: {self.model}")
        print(f"   Project: {self.project_root}")
        print(f"   Session: {self.session_id}")
        print(f"   Context window: {self.context.max_tokens:,} tokens")
        if self._fallback.enabled:
            fb_status = "enabled" if self._fallback.is_configured() else "enabled (not configured)"
            print(f"   Fallback: {fb_status} ({self._fallback.fallback_provider})")
        else:
            print("   Fallback: disabled (local-only)")
        print(f"   Auto-execute direct: {'on' if self.auto_execute_direct else 'off'}")
        print("   Type 'exit', 'quit', or Ctrl+D to leave.")
        print("   Type '!!' to see available tools.")
        print("   Type '/model' for model info, '/model <name>' to swap.")
        print("   Type '/model recommend' for hardware-aware suggestions.")
        print("   Type '/model stats' for per-model performance telemetry.")
        print("   Type '/model pin <name>' to pin a model digest.")
        print("   Type '/hardware' for GPU info.")
        print()

        self.bootstrap()
        msgs = self.context.get_messages()
        if len(msgs) > 1 or any(m.get("role") != "system" for m in msgs):
            print("   📥 Restored previous session context.")

        # REPL loop
        try:
            self._loop()
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving checkpoint...")
        finally:
            self._checkpoint()
            print("\n   💾 Checkpoint saved. Goodbye.\n")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main REPL loop."""
        while self.turns < self.max_turns:
            try:
                user_input = input("animus > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":q"):
                break

            if user_input == "!!":
                self._show_tools()
                continue

            if user_input.startswith("/"):
                self._handle_slash(user_input)
                continue

            # Normal turn
            self._turn(user_input)
            self.turns += 1

            # Session lifecycle check
            if self._check_session_limits():
                break

            # Auto-checkpoint
            if self.turns % self.checkpoint_every == 0:
                self._checkpoint()

    def _turn(self, user_input: str) -> None:
        """Process one user turn with intent parsing, planning, and synthesis."""
        # Parse intent
        intent = self._intent_parser.parse(user_input)

        # Handle clarification needed immediately
        if intent.intent_type == IntentType.CLARIFICATION_NEEDED:
            plan = self._planner.plan(intent, str(self.project_root))
            if plan.requires_clarification:
                print(f"\n🤔 {plan.clarification_prompt}\n")
                return

        # Handle conversational input immediately
        if intent.intent_type == IntentType.CONVERSATIONAL:
            self.context.add_message({"role": "user", "content": user_input})
            response = self._call_model()
            if response:
                self.context.add_message({"role": "assistant", "content": response.content or ""})
                print(f"\n{response.content}\n")
            return

        # Generate plan from intent
        plan = self._planner.plan(intent, str(self.project_root))

        # Fast-path: auto-execute direct commands with high confidence
        if (
            self.auto_execute_direct
            and intent.intent_type == IntentType.DIRECT_COMMAND
            and plan.confidence >= 0.75
            and plan.steps
        ):
            self._execute_plan(user_input, plan)
            return

        # Standard path: add user message, let model decide tool calls
        self.context.add_message({"role": "user", "content": user_input})

        # For vague requests, inject a planning hint
        if intent.intent_type == IntentType.VAGUE_REQUEST and plan.steps:
            hint = self._build_planning_hint(plan)
            self.context.add_message({"role": "system", "content": hint})

        # Get available tools
        tools = self.tools.list_tools()

        # Call model
        response = self._call_model(tools=tools if tools else None)
        if not response:
            print("   [Model returned no response]")
            return

        # Handle tool_calls loop with synthesis
        self._handle_tool_loop(user_input, response, tools)

    def _execute_plan(self, user_input: str, plan) -> None:
        """Fast-path execution of a direct-command plan without model involvement."""
        self.context.add_message({"role": "user", "content": user_input})
        print(
            f"   ⚡ Executing plan ({len(plan.steps)} step{'s' if len(plan.steps) > 1 else ''})..."
        )

        results: list[tuple[str, dict, str]] = []
        for i, step in enumerate(plan.steps, 1):
            print(f"   → Step {i}: {step.tool_name}")
            result = self.tools.execute(step.tool_name, step.arguments)
            results.append((step.tool_name, step.arguments, result))

            # Add to context as tool result (simulating assistant + tool exchange)
            tool_call_id = f"fastpath-{i}"
            self.context.add_message(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": step.tool_name,
                                "arguments": json.dumps(step.arguments),
                            },
                        }
                    ],
                }
            )
            self.context.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": step.tool_name,
                    "content": result,
                }
            )

        # Synthesize results
        synthesis = self._synthesizer.synthesize_multi(results)
        summary = synthesis.summary
        if synthesis.detail:
            summary += f"\n\n```\n{synthesis.detail[:1000]}\n```"

        self.context.add_message({"role": "assistant", "content": summary})
        print(f"\n{summary}\n")

    def _handle_tool_loop(self, user_input: str, response, tools: list[dict] | None) -> None:
        """Handle the tool-calls loop with validation, retry, fallback, and synthesis."""
        max_tool_rounds = 10
        invalid_calls_history: list = []
        valid_calls_history: list = []
        tool_results: list[tuple[str, dict, str]] = []

        for _ in range(max_tool_rounds):
            if not response.tool_calls:
                break

            # Validate tool calls before executing
            invalid_calls = []
            valid_calls = []
            for tc in response.tool_calls:
                val_result = self._retryable.validator.validate(tc.name, tc.arguments)
                if val_result.valid:
                    valid_calls.append(tc)
                else:
                    invalid_calls.append(val_result)

            invalid_calls_history.extend(invalid_calls)
            valid_calls_history.extend(valid_calls)

            # If any invalid, inform model and retry once per round
            if invalid_calls:
                logger.warning(
                    "Invalid tool calls detected: %s",
                    [c.tool_name for c in invalid_calls],
                )
                retry_prompt = self._retryable.validator.build_retry_prompt(invalid_calls)
                self.context.add_message({"role": "user", "content": retry_prompt})

                # Evaluate whether to trigger fallback mid-turn
                score = self._quality_gate.evaluate(user_input, response, [], invalid_calls)
                if self._quality_gate.should_fallback(score) and self._fallback.can_fallback():
                    fb_response = self._fallback.try_fallback(
                        messages=self.context.get_messages(),
                        tools=tools if tools else None,
                        reason=f"tool validation failed ({score.reason})",
                    )
                    if fb_response:
                        print("   ☁️ Escalated to cloud model for this turn.")
                        response = fb_response
                        self.total_tokens += response.tokens_used
                        self._record_telemetry(
                            model=response.model or self._fallback.fallback_provider,
                            latency_ms=response.latency_ms or 0.0,
                            tokens=response.tokens_used,
                            fallback=True,
                        )
                        break

                # Re-call model for corrected calls
                response = self._call_model(tools=tools if tools else None)
                if not response:
                    print("   [Model returned no response during retry]")
                    return
                continue  # Re-process the corrected calls

            # Add assistant message with tool_calls
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else str(tc.arguments)
                            ),
                        },
                    }
                    for tc in valid_calls
                ],
            }
            self.context.add_message(assistant_msg)

            # Execute each validated tool with synthesis
            for tc in valid_calls:
                result = self.tools.execute(tc.name, tc.arguments)
                tool_results.append((tc.name, tc.arguments, result))

                # Synthesize for display
                synth = self._synthesizer.synthesize(tc.name, tc.arguments, result)
                if synth.summary:
                    print(f"   {synth.summary}")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                }
                self.context.add_message(tool_msg)

            # Call model again with tool results
            response = self._call_model(tools=tools if tools else None)
            if not response:
                print("   [Model returned no response after tool execution]")
                return

        # Final quality evaluation before presenting to user
        score = self._quality_gate.evaluate(
            user_input, response, valid_calls_history, invalid_calls_history
        )
        if self._quality_gate.should_fallback(score) and self._fallback.can_fallback():
            fb_response = self._fallback.try_fallback(
                messages=self.context.get_messages(),
                tools=tools if tools else None,
                reason=f"final response quality low ({score.reason})",
            )
            if fb_response:
                print("   ☁️ Escalated to cloud model for this turn.")
                response = fb_response
                self.total_tokens += response.tokens_used
                self._record_telemetry(
                    model=response.model or self._fallback.fallback_provider,
                    latency_ms=response.latency_ms or 0.0,
                    tokens=response.tokens_used,
                    fallback=True,
                )

        # Final assistant response
        self.context.add_message(
            {
                "role": "assistant",
                "content": response.content or "",
            }
        )
        print(f"\n{response.content}\n")

    @staticmethod
    def _build_planning_hint(plan) -> str:
        """Build a system hint for vague requests based on the heuristic plan."""
        steps = " → ".join(s.tool_name for s in plan.steps[:5])
        return (
            f"[Planner hint: The user made a vague request. "
            f"Consider starting with: {steps}. "
            f"Ask the user for clarification if the request is ambiguous.]"
        )

    def process_message(self, user_input: str) -> dict:
        """Process a message programmatically and return structured results.

        Used by the daemon and other programmatic consumers. Captures
        printed output and returns it along with metadata.

        Returns:
            Dict with keys: response, tokens_used, fallback_used, turns
        """
        import io
        import sys

        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf

        fb_before = self._fallback.status.fallbacks_this_session
        try:
            self._turn(user_input)
            self.turns += 1
            if self.turns % self.checkpoint_every == 0:
                self._checkpoint()
            self._check_session_limits()
        finally:
            sys.stdout = old_stdout

        fb_after = self._fallback.status.fallbacks_this_session
        return {
            "response": buf.getvalue(),
            "tokens_used": self.total_tokens,
            "fallback_used": fb_after > fb_before,
            "turns": self.turns,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _check_session_limits(self) -> bool:
        """Check session limits and trigger graceful finalize if breached.

        Returns True if the session was wrapped up (and optionally restarted).
        """
        if self._session_wrapped_up or not self._session_controller:
            return False

        stats = self.context.get_stats()
        elapsed = (
            (datetime.now(UTC) - self._session_started_at).total_seconds()
            if self._session_started_at
            else 0.0
        )

        breached, reason = self._session_controller.should_finalize(
            self.session_id,
            stats.utilization_percent,
            elapsed,
            self.turns,
        )
        if not breached:
            return False

        self._session_controller.log_event(
            self.session_id,
            SessionLifecycleEvent.WRAPPING_UP,
            stats.utilization_percent,
            elapsed,
            self.turns,
            reason,
        )
        print(f"\n   ⏳ Session limit reached: {reason}")
        print("   Wrapping up session gracefully...\n")

        self._graceful_finalize()
        return True

    def _graceful_finalize(self) -> None:
        """Generate a summary, checkpoint, and optionally restart the session."""
        policy = self._session_controller.policy if self._session_controller else None
        wrapup_prompt = (
            policy.wrapup_prompt
            if policy and policy.wrapup_prompt
            else SessionController.DEFAULT_WRAPUP_PROMPT
        )

        # Inject wrap-up prompt as a system message
        self.context.add_message({"role": "system", "content": wrapup_prompt})

        # Call model for summary (no tools, single turn)
        response = self._call_model()
        summary = response.content.strip() if response and response.content else ""

        if summary:
            self.context.add_message({"role": "assistant", "content": summary})
            existing = self.context._summary
            if existing:
                self.context.set_summary(f"{existing}\n\n{summary}")
            else:
                self.context.set_summary(summary)

        # Checkpoint before any restart
        self._checkpoint()
        self._session_wrapped_up = True

        elapsed = (
            (datetime.now(UTC) - self._session_started_at).total_seconds()
            if self._session_started_at
            else 0.0
        )
        if self._session_controller:
            self._session_controller.log_event(
                self.session_id,
                SessionLifecycleEvent.CHECKPOINTING,
                self.context.get_stats().utilization_percent,
                elapsed,
                self.turns,
                "Checkpoint saved before restart",
            )

        # Restart or finish
        policy = self._session_controller.policy if self._session_controller else None
        if policy and policy.auto_restart:
            self._restart_session()
        else:
            print("\n   ✅ Session wrapped up. Exiting.\n")

    def _restart_session(self) -> None:
        """Start a fresh session bootstrapped from the current checkpoint."""
        print("\n   🔄 Restarting session from checkpoint...\n")

        old_session_id = self.session_id
        old_summary = self.context._summary

        # Generate new session id
        self.session_id = self._generate_session_id()
        self.turns = 0
        self.total_tokens = 0
        self._session_wrapped_up = False

        # Preserve summary and clear messages
        self.context.clear()
        if old_summary:
            self.context.set_summary(old_summary)

        # Re-bootstrap with fresh system prompt but previous context
        self.bootstrap()
        self._session_started_at = datetime.now(UTC)

        elapsed = 0.0
        if self._session_controller:
            self._session_controller.log_event(
                self.session_id,
                SessionLifecycleEvent.RESTARTING,
                self.context.get_stats().utilization_percent,
                elapsed,
                self.turns,
                f"Restarted from {old_session_id}",
            )

        print(f"   🧠 New session started: {self.session_id}")
        print("   📥 Restored context from previous session.\n")

    # ------------------------------------------------------------------
    # Model call
    # ------------------------------------------------------------------

    def _call_model(self, tools: list[dict] | None = None) -> Any | None:
        """Call the Ollama model with current messages and optional tools."""
        try:
            request = CompletionRequest(
                prompt="",  # Not used when messages provided
                messages=self.context.get_messages(),
                model=self.model,
                temperature=0.7,
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            start = time.time()
            response = self.provider.complete(request)
            latency_ms = (time.time() - start) * 1000
            self.total_tokens += response.tokens_used
            self._record_telemetry(
                model=response.model or self.model,
                latency_ms=latency_ms,
                tokens=response.tokens_used,
                fallback=False,
            )
            return response
        except Exception as exc:
            logger.exception("Model call failed")
            print(f"   [ERROR: Model call failed: {exc}]")
            return None

    def _record_telemetry(
        self,
        model: str,
        latency_ms: float,
        tokens: int,
        fallback: bool = False,
    ) -> None:
        """Append a single observation to the per-model telemetry store."""
        entry = self._model_telemetry.setdefault(
            model, {"calls": 0, "latency_ms": 0.0, "tokens": 0, "fallbacks": 0}
        )
        entry["calls"] += 1
        entry["latency_ms"] += latency_ms
        entry["tokens"] += tokens
        if fallback:
            entry["fallbacks"] += 1

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _show_model_info(self) -> None:
        """Display current model, installed models, and hardware profile."""
        print(f"\n   Current model: {self.model}")

        # Show pin status
        from animus_kernel.providers.model_pin import ModelPinStore

        pin_store = ModelPinStore()
        pin = pin_store.get_pin(self.model)
        if pin:
            print(f"   🔒 Pinned: {pin}")
        else:
            print("   🔓 Not pinned")

        # List installed models
        try:
            installed = self.provider.list_models()
            if installed:
                print(f"\n   Installed models ({len(installed)}):")
                for m in installed:
                    marker = "  ← current" if m == self.model else ""
                    pin_marker = " 🔒" if pin_store.get_pin(m) else ""
                    print(f"   • {m}{marker}{pin_marker}")
            else:
                print("\n   No models reported by Ollama.")

            # Show currently loaded models
            try:
                running = self.provider.running_models()
                if running:
                    print(f"\n   Running in VRAM ({len(running)}):")
                    for m in running:
                        size_mb = m.get("size_vram", 0) / (1024 * 1024)
                        print(f"   • {m['name']} ({size_mb:.0f} MB)")
                else:
                    print("\n   No models currently loaded in VRAM.")
            except Exception:
                pass  # Non-critical
        except Exception as exc:
            print(f"\n   Could not list models: {exc}")

        # Hardware profile
        from animus_kernel.providers.hardware import detect_hardware

        hw = detect_hardware()
        print("\n   Hardware profile:")
        print(f"   • Platform: {hw.platform_id}")
        print(f"   • GPU: {hw.gpu_name or 'None detected'}")
        if hw.gpu_vram_gb:
            print(f"   • GPU VRAM: {hw.gpu_vram_gb} GB")
        print(f"   • Total memory: {hw.total_memory_gb} GB")
        print(f"   • Available memory: {hw.available_memory_gb} GB")
        print(f"   • Recommended tier: {hw.recommended_tier}")
        if hw.recommended_models:
            print(f"   • Recommended models: {', '.join(hw.recommended_models)}")
        if hw.warnings:
            for w in hw.warnings:
                print(f"   ⚠ {w}")
        print()

    def _recommend_model(self) -> None:
        """Recommend the best installed model not currently in use."""
        from animus_kernel.providers.hardware import detect_hardware

        try:
            installed = self.provider.list_models()
        except Exception as exc:
            print(f"   Error listing models: {exc}")
            return

        hw = detect_hardware()
        recs = hw.recommended_models

        # Filter to installed models
        installed_set = {m.split(":")[0] for m in installed}
        installed_full = set(installed)
        candidates: list[str] = []
        for r in recs:
            # Exact match first
            if r in installed_full:
                candidates.append(r)
                continue
            # Base match (e.g. "qwen2.5" matches "qwen2.5:14b")
            base = r.split(":")[0]
            for m in installed:
                if m.split(":")[0] == base:
                    candidates.append(m)
                    break

        # Remove current model
        current_base = self.model.split(":")[0]
        candidates = [c for c in candidates if c.split(":")[0] != current_base]

        if not candidates:
            print("   No alternative recommendations.")
            print(f"   Current model ({self.model}) is the best installed option.")
            return

        print(f"\n   Recommended models (tier: {hw.recommended_tier}):")
        for i, c in enumerate(candidates[:3], 1):
            marker = "  ← already running" if c == self.model else ""
            print(f"   {i}. {c}{marker}")
        print("\n   Use /model <name> to swap.")
        print()

    def _show_model_stats(self) -> None:
        """Display per-model performance telemetry for this session."""
        if not self._model_telemetry:
            print("\n   No model calls recorded yet.")
            return

        print("\n   Model performance this session:")
        print(f"   {'Model':<25} {'Calls':>6} {'Avg ms':>10} {'Tokens/sec':>12} {'Fallbacks':>10}")
        print(f"   {'-' * 25} {'-' * 6} {'-' * 10} {'-' * 12} {'-' * 10}")
        for model, data in sorted(
            self._model_telemetry.items(),
            key=lambda x: x[1]["latency_ms"] / max(x[1]["calls"], 1),
        ):
            calls = data["calls"]
            avg_lat = data["latency_ms"] / calls
            tokens = data["tokens"]
            tps = (
                round((tokens / (data["latency_ms"] / 1000)), 1) if data["latency_ms"] > 0 else 0.0
            )
            fallbacks = data.get("fallbacks", 0)
            print(f"   {model:<25} {calls:>6} {avg_lat:>10.1f} {tps:>12} {fallbacks:>10}")
        print()

    def _pin_model(self, model: str) -> None:
        """Pin a model's digest for tamper detection."""
        from animus_kernel.providers.model_pin import ModelPinStore, fetch_ollama_digest

        store = ModelPinStore()
        digest = fetch_ollama_digest(model, base_url=self.provider.base_url)
        if digest is None:
            print(f"   Could not fetch digest for '{model}'. Is Ollama running?")
            return
        store.pin_model(model, digest)
        print(f"   🔒 Pinned {model} → {digest}")

    def _unpin_model(self, model: str) -> None:
        """Remove a model pin."""
        from animus_kernel.providers.model_pin import ModelPinStore

        store = ModelPinStore()
        store.unpin_model(model)
        print(f"   🔓 Unpinned {model}")

    def _list_pins(self) -> None:
        """Show all pinned models."""
        from animus_kernel.providers.model_pin import ModelPinStore

        store = ModelPinStore()
        pins = store.list_pins()
        if not pins:
            print("   No pinned models.")
            return
        print(f"\n   Pinned models ({len(pins)}):")
        for model, digest in pins.items():
            print(f"   • {model} → {digest}")
        print()

    def _swap_model(self, model: str, warm: bool = False) -> None:
        """Swap to a different Ollama model mid-session, preserving conversation state.

        Args:
            model: Target Ollama model name
            warm: If True, send a tiny warmup prompt to preload the model into VRAM
        """
        # Validate model is installed
        try:
            installed = self.provider.list_models()
        except Exception as exc:
            print(f"   Error listing models: {exc}")
            return

        # Normalize: handle bare names by checking if any installed model contains it
        exact_match = model in installed
        if not exact_match:
            matches = [m for m in installed if m.startswith(model) or model in m]
            if len(matches) == 1:
                model = matches[0]
                exact_match = True
            elif len(matches) > 1:
                print(f"   Ambiguous model '{model}' matches: {', '.join(matches)}")
                return

        if not exact_match:
            print(f"   Model '{model}' is not installed.")
            print(f"   Run: ollama pull {model}")
            return

        # Hardware sanity check
        from animus_kernel.providers.hardware import detect_hardware

        hw = detect_hardware()
        size_hint = self._extract_model_size(model)
        warning = ""
        if hw.gpu_vram_gb and size_hint:
            est_vram = size_hint * 0.7  # rough Q4 estimate
            if est_vram > hw.gpu_vram_gb * 0.9:
                warning = (
                    f" (Warning: may exceed available VRAM — "
                    f"{est_vram:.0f}GB estimated vs {hw.gpu_vram_gb}GB available)"
                )

        # Preserve conversation state
        old_messages = self.context._messages.copy()
        old_summary = self.context._summary
        old_summary_tokens = self.context._summary_tokens
        old_dropped = self.context.dropped_messages

        # Rebuild provider and context manager — preserve custom host/base_url
        try:
            new_provider = OllamaProvider(
                model=model,
                host=self.provider.base_url,
            )
            if not new_provider.is_configured():
                print(f"   Ollama reports model '{model}' is not available.")
                return
        except Exception as exc:
            print(f"   Failed to initialize provider for '{model}': {exc}")
            return

        self.model = model
        self.provider = new_provider
        self.context = HeadContextManager(model=model)

        # Restore conversation state (private access is intentional: preserves
        # message history across model swaps without triggering re-pruning)
        self.context._messages = old_messages
        self.context._summary = old_summary
        self.context._summary_tokens = old_summary_tokens
        self.context.dropped_messages = old_dropped

        # Context-budget guard: proactively prune if the new window is smaller
        stats = self.context.get_stats()
        if stats.utilization_percent > 100:
            print(
                f"   Warning: conversation ({stats.total_tokens:,} tokens) exceeds "
                f"new context window ({self.context.max_tokens:,} tokens)."
            )
            self.context._prune_if_needed()
            post_stats = self.context.get_stats()
            print(
                f"   Pruned to {post_stats.total_tokens:,} tokens "
                f"({post_stats.dropped_messages} messages dropped)."
            )

        print(f"   Model swapped to: {self.model}{warning}")
        print(f"   Context window: {self.context.max_tokens:,} tokens")

        if warm:
            print("   Warming up model...", end="", flush=True)
            try:
                from animus_kernel.providers.base import CompletionRequest

                warmup_req = CompletionRequest(
                    prompt="",
                    messages=[{"role": "user", "content": "hi"}],
                    model=self.model,
                    temperature=0.1,
                )
                self.provider.complete(warmup_req)
                print(" done.")
            except Exception:
                print(" failed (non-critical).")

    @staticmethod
    def _extract_model_size(model_name: str) -> int | None:
        """Extract parameter size hint from model name, e.g. 'qwen2.5:32b' -> 32."""
        match = re.search(r"(\d+)(?:\.\d+)?[bm]", model_name.lower())
        if match:
            val = float(match.group(1))
            if "b" in model_name.lower():
                return int(val)
            if "m" in model_name.lower():
                return int(val / 1000)
        return None

    def _show_tools(self) -> None:
        """Display available tools."""
        print("\n   Available tools:")
        for tool in self.tools.list_tools():
            fn = tool.get("function", {})
            print(f"   • {fn.get('name', '?')} — {fn.get('description', '')[:60]}...")
        print()

    def _handle_slash(self, command: str) -> None:
        """Handle slash commands."""
        parts = command[1:].split(None, 1)
        if not parts:
            return
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "model":
            if arg == "recommend":
                self._recommend_model()
            elif arg == "stats":
                self._show_model_stats()
            elif arg.startswith("pin "):
                self._pin_model(arg[4:].strip())
            elif arg.startswith("unpin "):
                self._unpin_model(arg[6:].strip())
            elif arg == "pins":
                self._list_pins()
            elif arg:
                warm = arg.endswith(" --warm")
                model_arg = arg.replace(" --warm", "").strip() if warm else arg
                self._swap_model(model_arg, warm=warm)
            else:
                self._show_model_info()
        elif cmd == "hardware":
            self._show_model_info()
        elif cmd == "project":
            print(f"   Project root: {self.project_root}")
        elif cmd == "session":
            stats = self.context.get_stats()
            print(f"   Session: {self.session_id}")
            print(f"   Turns: {self.turns}")
            print(f"   Total tokens (provider): {self.total_tokens:,}")
            print(f"   Context window: {stats.max_tokens:,} tokens")
            print(f"   Utilization: {stats.utilization_percent}%")
            print(
                f"   Messages: {stats.message_count} ({stats.user_messages} user, {stats.assistant_messages} assistant, {stats.tool_messages} tool)"
            )
            print(f"   Available: {stats.available_tokens:,} tokens")
            if stats.dropped_messages:
                print(f"   Pruned: {stats.dropped_messages} messages")
            fb = self._fallback.status
            if fb.enabled:
                print(
                    f"   Fallback: {fb.fallbacks_this_session}/{fb.max_fallbacks} used ({fb.provider_name})"
                )
        elif cmd == "context":
            stats = self.context.get_stats()
            print(f"   Context window: {stats.max_tokens:,} tokens")
            print(f"   Reserve: {stats.reserve_tokens:,} tokens")
            print(f"   Used: {stats.total_tokens:,} tokens")
            print(f"   Available: {stats.available_tokens:,} tokens")
            print(f"   Utilization: {stats.utilization_percent}%")
            print(f"   Messages: {stats.message_count}")
            if self.context._summary:
                print(f"   Summary: {self.context._summary[:120]}...")
        elif cmd == "mode":
            if arg == "local":
                self._fallback.enabled = False
                print("   Mode: local-only. Cloud fallback disabled.")
            elif arg == "hybrid":
                self._fallback.enabled = True
                configured = self._fallback.is_configured()
                if configured:
                    print(
                        f"   Mode: hybrid. Cloud fallback enabled ({self._fallback.fallback_provider})."
                    )
                else:
                    print(
                        f"   Mode: hybrid requested, but {self._fallback.fallback_provider} is not configured."
                    )
                    print("   Set your ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
            elif arg == "cloud":
                self._fallback.enabled = True
                print(
                    "   Mode: cloud-preferred. (Note: Head is local-first; cloud-preferred is for testing.)"
                )
            elif arg:
                print(f"   Unknown mode: {arg}. Use local, hybrid, or cloud.")
            else:
                mode = "hybrid" if self._fallback.enabled else "local"
                print(f"   Current mode: {mode}")
                fb = self._fallback.status
                print(f"   Fallback provider: {fb.provider_name}")
                print(f"   Configured: {fb.configured}")
        elif cmd == "auto":
            if arg == "on":
                self.auto_execute_direct = True
                print("   Auto-execute direct commands: on")
            elif arg == "off":
                self.auto_execute_direct = False
                print("   Auto-execute direct commands: off")
            else:
                print(
                    f"   Auto-execute direct commands: {'on' if self.auto_execute_direct else 'off'}"
                )
        elif cmd == "remember":
            if arg:
                result = self.tools.execute("remember", {"content": arg, "tags": ["manual"]})
                print(f"   {result}")
            else:
                print("   Usage: /remember <content>")
        elif cmd == "recall":
            query = arg or self.project_root.name
            result = self.tools.execute("recall", {"query": query, "limit": 5})
            print(f"   {result}")
        elif cmd == "checkpoint":
            self._checkpoint()
            print("   Checkpoint saved.")
        elif cmd == "restart":
            self._restart_session()
            print("   Session restarted from checkpoint.")
        elif cmd == "clear":
            # Keep system message, reset rest
            system = self.context._system_message()
            self.context.clear()
            if system:
                self.context.add_message(system)
            self.turns = 0
            self._quality_gate.reset()
            self._fallback.reset()
            print("   Context cleared.")
        else:
            print(f"   Unknown command: /{cmd}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save session checkpoint."""
        checkpoint = HeadCheckpoint(
            session_id=self.session_id,
            started_at=datetime.now(UTC),
            last_active_at=datetime.now(UTC),
            project_root=str(self.project_root),
            model=self.model,
            messages=self._filter_messages(self.context.get_messages()),
            summary=self.context._summary,
            total_tokens=self.total_tokens,
            turns=self.turns,
        )
        self.checkpoint_store.save(checkpoint)

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a short session ID."""
        return f"head-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _filter_messages(messages: list[dict]) -> list[dict]:
        """Filter messages to serializable subset."""
        # Drop any keys that aren't standard for the chat API
        allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
        cleaned = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed}
            cleaned.append(clean)
        return cleaned
