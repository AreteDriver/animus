# Spike — Provider.complete() Ollama tool-call support

**Spike ID:** Pre-v0-implementation #1 (per `cc-loop-port-plan.md` §15 step 3)
**Question:** Does Forge's existing `Provider.complete()` support passing tool schemas to Ollama and parsing tool-call responses, or do we need a new `provider_wrapper` adapter? (cc-loop-port-plan OQ8)
**Time:** ~30 min (vs ~1h budget)
**Date:** 2026-05-23
**Verdict:** **RESOLVED — integration is feasible WITHOUT a new adapter.** Two 1-line patches close the remaining gaps.

---

## Evidence

### CompletionRequest schema already supports tools

`packages/forge/src/animus_forge/providers/base.py:69-91`:

```python
@dataclass
class CompletionRequest:
    # ...
    # Tool use support
    tools: list[dict] | None = None  # Anthropic/OpenAI tool definitions
    tool_choice: str | None = None  # "auto", "any", "none", or specific tool name
```

`base.py:94-100` + `:117`:

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class CompletionResponse:
    # ...
    tool_calls: list[ToolCall] = field(default_factory=list)
```

Schema is provider-agnostic and already designed for tool-use. No changes needed.

### OllamaProvider wires both sides

**Request side** — `providers/ollama_provider.py:192-193` passes tools through to Ollama's `/api/chat` payload:

```python
if request.tools:
    payload["tools"] = request.tools
```

**Response side** — `providers/ollama_provider.py:304-325` extracts tool calls from Ollama's response:

```python
@staticmethod
def _extract_tool_calls(message: dict) -> list[ToolCall]:
    raw_calls = message.get("tool_calls", [])
    tool_calls: list[ToolCall] = []
    for i, call in enumerate(raw_calls):
        fn = call.get("function", {})
        tool_calls.append(
            ToolCall(
                id=f"ollama_tool_{i}",
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
            )
        )
    return tool_calls
```

Called from both sync `complete()` (line 229) and async `complete_async()` (line 283). Both paths construct `CompletionResponse(..., tool_calls=tool_calls)`.

### Token counting matches spec needs

`ollama_provider.py:232-241` (and async equivalent):

```python
prompt_eval_count = data.get("prompt_eval_count", 0)
eval_count = data.get("eval_count", 0)
# ...
tokens_used=prompt_eval_count + eval_count,
input_tokens=prompt_eval_count,
output_tokens=eval_count,
```

This satisfies `spec.md` R3 (BudgetManager routing) + RD1 (token tracking for $0 local calls). Also resolves cc-loop-port-plan **OQ3** ("Token-counting mechanism for local Ollama") — trust Ollama's counts; no separate tokenizer pass needed.

### Real tool-use already exercised in tests

`packages/forge/tests/test_smoke_ollama.py:195-274` runs `TaskRunner` integration tests with Ollama + tool registry:

```python
async def test_task_runner_with_tools(self, agent_provider):
    """TaskRunner executes a builder task with tool access."""
    # ...
    tool_registry = ForgeToolRegistry(enable_shell=True)
    # ...
    task="List the files in the current directory using the list_files tool.",
    use_tools=True,
```

This is not theoretical — Forge already runs Ollama-backed agents with tool use today.

---

## Gaps Found (small)

### Gap 1: `qwen3:32b` not in default tier model list

`ollama_provider.py:30-57` enumerates `DEFAULT_TIER_MODELS`. The "reasoning" tier lists `qwen2.5:32b`, `deepseek-r1:32b`, etc. — but NOT `qwen3:32b` (v0's default per spec R4).

**Impact:** Cosmetic. If v0 uses `--model qwen3:32b` explicitly (spec R4 default precedence: CLI > env > config > `qwen3:32b`), the tier list is bypassed entirely. Tier selection only kicks in for the router's automatic tier-based routing, which v0 doesn't use.

**Fix at v0 implementation kickoff (optional):** Add `"qwen3:32b"` to `DEFAULT_TIER_MODELS["reasoning"]` ordered first. 1-line patch.

### Gap 2: `tool_choice` field is dropped before reaching Ollama

`_build_request()` at `ollama_provider.py:165-195` passes `request.tools` through but does NOT pass `request.tool_choice`. Other providers (e.g. Anthropic, OpenAI) honor it.

**Impact:** v0 doesn't need to force specific tool selection ("auto" is fine). Only matters if v0 ever wants to constrain to a specific tool. Spec.md does NOT name `tool_choice` as a v0 requirement.

**Fix at v0 implementation kickoff (optional, when actually needed):** Add `if request.tool_choice: payload["tool_choice"] = request.tool_choice` to `_build_request`. 1-line patch.

### Non-gap: tool_call IDs are synthesized

`_extract_tool_calls` synthesizes IDs as `f"ollama_tool_{i}"` since Ollama doesn't return them. cc-loop-port-plan §3 already specs that Animus wraps with its own `call.id` at hook-chain entry, so this is by design.

---

## Implications for v0 Implementation

| Spec element | Status |
|---|---|
| R3 (BudgetManager routing) | **No change needed** — existing `Provider.complete()` + token counts route through unchanged |
| R4 (Ollama + Qwen3-32B default) | **Already works** — `--model qwen3:32b` is sufficient. Optional tier-list addition (Gap 1) is polish, not load-bearing |
| `spec.md §4.7 Skill Loader` interaction | **Independent** — skill loader doesn't touch Provider layer |
| cc-loop-port-plan §3 tool execution protocol | **No adapter needed** — protocol maps cleanly to existing `CompletionRequest(tools=...)` / `CompletionResponse(tool_calls=[...])` shapes |
| cc-loop-port-plan OQ3 (token counting) | **RESOLVED** — `prompt_eval_count` + `eval_count` are populated; trust them |
| cc-loop-port-plan OQ8 (provider_wrapper adapter need) | **RESOLVED — no adapter needed** |
| Hook chain → audit log via BudgetManager | **Path clear** — BudgetManager wraps `Provider.complete()` calls; agent's hook chain emits audit entries adjacent (per hook-system-port-plan §1 row 12) |

---

## Recommended Actions

1. **Mark cc-loop-port-plan OQ3 + OQ8 resolved.** Both are answered by this spike. (Edit cc-loop-port-plan.md to add Resolved-Decisions note for these two OQs, or treat this spike doc as the resolution and reference it.)
2. **At v0 implementation kickoff:** add 1-line `qwen3:32b` entry to `DEFAULT_TIER_MODELS["reasoning"]` if you want tier-routing to know about it. Otherwise just pass `--model qwen3:32b` and skip the patch.
3. **At v0 implementation kickoff (if/when needed):** add 1-line `tool_choice` passthrough in `_build_request`. Not v0-blocking.
4. **No spec.md amendment needed** — existing requirements R3/R4 are satisfied by the current OllamaProvider implementation.

---

## Time-to-First-Working-Agent Implications

Before this spike, OQ8 carried the highest "is this feasible without a new adapter" risk. Concrete answer: **the Provider abstraction + OllamaProvider implementation already do everything v0 needs.** v0 implementation work can start at Quorum v2 wk5 gate without any pre-v0 provider work.

This shifts the smallest-meaningful-first-step from "verify Provider integration" to "write `animus-agent run` CLI skeleton + wire to existing OllamaProvider with smolagents subclass (per hook-system-port-plan §6)." Likely a 2-4 hour first sprint produces a working `--dry-run` end-to-end.

---

End of spike.
