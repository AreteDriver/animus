"""Animus Head REPL — local-first agentic conversation loop.

Runs an iterative turn-by-turn loop with a local Ollama model,
auto-loading context, executing tools, and persisting state.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore
from animus_kernel.head.context_manager import HeadContextManager
from animus_kernel.head.fallback_controller import HeadFallbackController
from animus_kernel.head.intent_parser import HeadIntentParser, IntentType
from animus_kernel.head.planner import HeadPlanner
from animus_kernel.head.quality_gate import HeadQualityGate
from animus_kernel.head.session_bootstrap import SessionBootstrap
from animus_kernel.head.synthesizer import HeadSynthesizer
from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.head.tool_validator import RetryableToolExecutor
from animus_kernel.providers.base import CompletionRequest, ToolCall
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Bootstrap session context and initialize message history.

        Called automatically by `start()`, but can be invoked directly
        for testing or programmatic use.
        """
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
        print(f"\n🧠 Animus Head — local-first agentic loop")
        print(f"   Model: {self.model}")
        print(f"   Project: {self.project_root}")
        print(f"   Session: {self.session_id}")
        print(f"   Context window: {self.context.max_tokens:,} tokens")
        if self._fallback.enabled:
            fb_status = "enabled" if self._fallback.is_configured() else "enabled (not configured)"
            print(f"   Fallback: {fb_status} ({self._fallback.fallback_provider})")
        else:
            print(f"   Fallback: disabled (local-only)")
        print(f"   Auto-execute direct: {'on' if self.auto_execute_direct else 'off'}")
        print(f"   Type 'exit', 'quit', or Ctrl+D to leave.")
        print(f"   Type '!!' to see available tools.")
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
            print(f"\n   💾 Checkpoint saved. Goodbye.\n")

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
        print(f"   ⚡ Executing plan ({len(plan.steps)} step{'s' if len(plan.steps) > 1 else ''})...")

        results: list[tuple[str, dict, str]] = []
        for i, step in enumerate(plan.steps, 1):
            print(f"   → Step {i}: {step.tool_name}")
            result = self.tools.execute(step.tool_name, step.arguments)
            results.append((step.tool_name, step.arguments, result))

            # Add to context as tool result (simulating assistant + tool exchange)
            tool_call_id = f"fastpath-{i}"
            self.context.add_message({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": step.tool_name, "arguments": json.dumps(step.arguments)},
                }],
            })
            self.context.add_message({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": step.tool_name,
                "content": result,
            })

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
                score = self._quality_gate.evaluate(
                    user_input, response, [], invalid_calls
                )
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

        # Final assistant response
        self.context.add_message({
            "role": "assistant",
            "content": response.content or "",
        })
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
        finally:
            sys.stdout = old_stdout

        fb_after = self._fallback.status.fallbacks_this_session
        return {
            "response": buf.getvalue(),
            "tokens_used": self.total_tokens,
            "fallback_used": fb_after > fb_before,
            "turns": self.turns,
        }

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
            response = self.provider.complete(request)
            self.total_tokens += response.tokens_used
            return response
        except Exception as exc:
            logger.exception("Model call failed")
            print(f"   [ERROR: Model call failed: {exc}]")
            return None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

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
            print(f"   Current model: {self.model}")
        elif cmd == "project":
            print(f"   Project root: {self.project_root}")
        elif cmd == "session":
            stats = self.context.get_stats()
            print(f"   Session: {self.session_id}")
            print(f"   Turns: {self.turns}")
            print(f"   Total tokens (provider): {self.total_tokens:,}")
            print(f"   Context window: {stats.max_tokens:,} tokens")
            print(f"   Utilization: {stats.utilization_percent}%")
            print(f"   Messages: {stats.message_count} ({stats.user_messages} user, {stats.assistant_messages} assistant, {stats.tool_messages} tool)")
            print(f"   Available: {stats.available_tokens:,} tokens")
            if stats.dropped_messages:
                print(f"   Pruned: {stats.dropped_messages} messages")
            fb = self._fallback.status
            if fb.enabled:
                print(f"   Fallback: {fb.fallbacks_this_session}/{fb.max_fallbacks} used ({fb.provider_name})")
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
                    print(f"   Mode: hybrid. Cloud fallback enabled ({self._fallback.fallback_provider}).")
                else:
                    print(f"   Mode: hybrid requested, but {self._fallback.fallback_provider} is not configured.")
                    print("   Set your ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
            elif arg == "cloud":
                self._fallback.enabled = True
                print("   Mode: cloud-preferred. (Note: Head is local-first; cloud-preferred is for testing.)")
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
                print(f"   Auto-execute direct commands: {'on' if self.auto_execute_direct else 'off'}")
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
