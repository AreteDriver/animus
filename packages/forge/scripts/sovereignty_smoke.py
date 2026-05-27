"""Smoke test for Qwen3.6 sovereignty-stack v0.

First runnable artifact of the local-AI-sovereignty research arc. Calls the
locally-hosted Qwen3.6-35B-A3B-Uncensored model and captures latency + token
throughput for three representative prompts.

Two runtime paths:

1. **OllamaProvider** (default) — when ollama can serve qwen35moe. Pulls the
   tag listed in ``MODEL_TAG`` and calls via ``CompletionRequest``.
2. **OpenAI-compat** (env override) — when ``ANIMUS_SOVEREIGNTY_BASE_URL`` is
   set, calls ``<base_url>/v1/chat/completions`` directly. Works against
   ``llama-server`` built from llama.cpp source (which supports qwen35moe
   while ollama 0.24.0's vendored llama.cpp does not — ollama issue #15898).

Usage:
    cd packages/forge

    # Path 1: ollama (default)
    python scripts/sovereignty_smoke.py

    # Path 2: llama-server / any OAI-compat endpoint
    ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 \\
      python scripts/sovereignty_smoke.py

The script is intentionally minimal:
- No BudgetManager (smoke only; BM is enforced at WorkflowExecutor level)
- No fallback wiring (this run is to characterize the local model in isolation)
- No judge — raw outputs are written to stdout for manual inspection

Output:
    Prompt N: ...
    Latency: NNN ms
    Tokens (in/out/total): a/b/c
    Throughput: N.NN tok/s
    Response:
        <content>

Captured metrics feed docs/SOVEREIGNTY_STACK.md for the v0 baseline.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter
from typing import NamedTuple

from animus_forge.providers.base import CompletionRequest
from animus_forge.providers.ollama_provider import OllamaProvider

MODEL_TAG = "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P"

PROMPTS = [
    {
        "name": "factual_concise",
        "system": "Answer in one sentence. No preamble.",
        "user": "What is the difference between TRIM and a secure-erase on an SSD?",
    },
    {
        "name": "reasoning_multistep",
        "system": "Think step by step, then give a final answer in one line prefixed with 'ANSWER:'",
        "user": (
            "A workflow runs every 6 hours and processes ~12k events per run. "
            "Each event triggers exactly one LLM call at ~800 input + ~400 output tokens. "
            "At Anthropic Sonnet pricing ($3/MT in, $15/MT out), what is the monthly "
            "cost? Show the arithmetic."
        ),
    },
    {
        "name": "format_structured",
        "system": "Respond ONLY with the requested JSON. No prose.",
        "user": (
            "Return a JSON object with keys: model_name (string), "
            "tradeoff_summary (string, <=20 words), best_for (array of 3 strings). "
            "The model is yourself: local Qwen3.6-35B-A3B."
        ),
    },
]


class _SmokeResult(NamedTuple):
    content: str
    input_tokens: int
    output_tokens: int


def _call_ollama(prompt: dict, model: str) -> _SmokeResult:
    provider = OllamaProvider(model=model)
    req = CompletionRequest(
        prompt=prompt["user"],
        system_prompt=prompt["system"],
        model=model,
        temperature=0.2,
        max_tokens=512,
    )
    resp = provider.complete(req)
    return _SmokeResult(
        content=resp.content or "",
        input_tokens=resp.input_tokens or 0,
        output_tokens=resp.output_tokens or 0,
    )


def _call_oai_compat(prompt: dict, model: str, base_url: str) -> _SmokeResult:
    """Call any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Used when ``ANIMUS_SOVEREIGNTY_BASE_URL`` is set — pointing at a
    ``llama-server`` instance (built from llama.cpp source, with
    ``qwen35moe`` support that ollama's vendored llama.cpp lacks).

    The Qwen3 chat template emits ``<think>...</think>`` by default;
    ``chat_template_kwargs.enable_thinking=false`` suppresses it on
    llama-server (ollama ignores the field silently). The strip pass
    below is belt-and-suspenders for any backend that emits the block
    inline anyway.
    """
    import httpx

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    content = (msg.get("content") or "").strip()
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    usage = data.get("usage") or {}
    return _SmokeResult(
        content=content,
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
    )


def run() -> int:
    base_url = os.environ.get("ANIMUS_SOVEREIGNTY_BASE_URL")
    use_oai = bool(base_url)

    if use_oai:
        host = base_url
        print("=== Sovereignty smoke v0 ===")
        print(f"Model: {MODEL_TAG}")
        print(f"Backend: openai-compat at {host}")
    else:
        provider = OllamaProvider(model=MODEL_TAG)
        if not provider.is_configured():
            print(
                "ERROR: Ollama not reachable at http://localhost:11434",
                file=sys.stderr,
            )
            return 1
        available = provider.list_models()
        if not any(MODEL_TAG.split(":")[0] in m for m in available):
            print(
                f"ERROR: model not pulled. Run: ollama pull {MODEL_TAG}",
                file=sys.stderr,
            )
            print(f"Available models: {available}", file=sys.stderr)
            return 1
        provider.initialize()
        host = provider.base_url
        print("=== Sovereignty smoke v0 ===")
        print(f"Model: {MODEL_TAG}")
        print(f"Backend: ollama at {host}")
    print()

    for i, p in enumerate(PROMPTS, 1):
        print(f"--- Prompt {i}: {p['name']} ---")
        start = perf_counter()
        try:
            result = (
                _call_oai_compat(p, MODEL_TAG, base_url) if use_oai else _call_ollama(p, MODEL_TAG)
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            continue
        elapsed_ms = (perf_counter() - start) * 1000

        in_tokens = result.input_tokens
        out_tokens = result.output_tokens
        throughput = (out_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        print(f"  Latency: {elapsed_ms:.0f} ms")
        print(f"  Tokens (in/out/total): {in_tokens}/{out_tokens}/{in_tokens + out_tokens}")
        print(f"  Throughput: {throughput:.2f} tok/s")
        print("  Response:")
        for line in result.content.splitlines():
            print(f"    {line}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(run())
