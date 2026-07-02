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
from animus_kernel.head.session_bootstrap import SessionBootstrap
from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.head.tool_validator import RetryableToolExecutor
from animus_kernel.providers.base import CompletionRequest, ToolCall
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
    ) -> None:
        self.model = model
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.session_id = self._generate_session_id()
        self.max_turns = max_turns
        self.checkpoint_every = checkpoint_every

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

        # Load or build system prompt
        if system_prompt is None:
            prompt_path = Path(__file__).parent / "prompts" / "system_repl.md"
            if prompt_path.exists():
                system_prompt = prompt_path.read_text()
            else:
                system_prompt = "You are Animus Head — a local-first agentic assistant."
        self.system_prompt = system_prompt

        # Conversation state
        self.messages: list[dict] = []
        self.turns = 0
        self.total_tokens = 0
        self.summary = ""

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
            self.messages = self._filter_messages(prev["messages"])
            self.turns = prev.get("turns", 0)
            self.total_tokens = prev.get("total_tokens", 0)
            self.summary = prev.get("summary", "")
            # Replace the first system message with fresh context
            if self.messages and self.messages[0].get("role") == "system":
                self.messages[0] = {"role": "system", "content": full_system}
            else:
                self.messages.insert(0, {"role": "system", "content": full_system})
        else:
            self.messages = [{"role": "system", "content": full_system}]

    def start(self) -> None:
        """Bootstrap and enter the REPL loop."""
        print(f"\n🧠 Animus Head — local-first agentic loop")
        print(f"   Model: {self.model}")
        print(f"   Project: {self.project_root}")
        print(f"   Session: {self.session_id}")
        print(f"   Type 'exit', 'quit', or Ctrl+D to leave.")
        print(f"   Type '!!' to see available tools.")
        print()

        self.bootstrap()
        if len(self.messages) > 1 or any(
            m.get("role") != "system" for m in self.messages
        ):
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
        """Process one user turn with validation and retry."""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Get available tools
        tools = self.tools.list_tools()

        # Call model
        response = self._call_model(tools=tools if tools else None)
        if not response:
            print("   [Model returned no response]")
            return

        # Handle tool_calls loop
        max_tool_rounds = 10
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

            # If any invalid, inform model and retry once per round
            if invalid_calls:
                logger.warning(
                    "Invalid tool calls detected: %s",
                    [c.tool_name for c in invalid_calls],
                )
                retry_prompt = self._retryable.validator.build_retry_prompt(invalid_calls)
                self.messages.append({"role": "user", "content": retry_prompt})

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
            self.messages.append(assistant_msg)

            # Execute each validated tool
            for tc in valid_calls:
                result = self.tools.execute(tc.name, tc.arguments)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                }
                self.messages.append(tool_msg)

            # Call model again with tool results
            response = self._call_model(tools=tools if tools else None)
            if not response:
                print("   [Model returned no response after tool execution]")
                return

        # Final assistant response
        self.messages.append({
            "role": "assistant",
            "content": response.content or "",
        })
        print(f"\n{response.content}\n")

    def _call_model(self, tools: list[dict] | None = None) -> Any | None:
        """Call the Ollama model with current messages and optional tools."""
        try:
            request = CompletionRequest(
                prompt="",  # Not used when messages provided
                messages=self.messages,
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
            print(f"   Session: {self.session_id}")
            print(f"   Turns: {self.turns}")
            print(f"   Tokens: {self.total_tokens}")
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
            if self.messages:
                system = self.messages[0]
                self.messages = [system]
            self.turns = 0
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
            messages=self._filter_messages(self.messages),
            summary=self.summary,
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
