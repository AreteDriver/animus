"""Agent adapter for the sovereignty-stack-v0 eval suite.

Routes each case's ``input.query`` to the model under test and returns
the raw text response. The model under test is parameterized so the
same suite runs across providers without YAML edits.

Environment:
    MODEL_UNDER_TEST: Model name passed to the provider (or to
        ``llama-server`` as the ``model`` field — llama-server ignores
        it since only one model is loaded, but it's preserved for
        provenance in logs). Default: ``claude-sonnet-4-6``.
    ANIMUS_SOVEREIGNTY_BASE_URL: If set, the adapter bypasses the
        Forge provider stack and calls ``<base_url>/v1/chat/completions``
        directly via httpx. This is the path that works against a
        ``llama-server`` built from llama.cpp source (which supports
        ``qwen35moe``; ollama 0.24.0's vendored llama.cpp does not —
        ollama issue #15898). Without this env var, the adapter routes
        through ``get_provider()`` and uses ``MODEL_UNDER_TEST`` as the
        default model.

Run patterns:
    # Cloud baseline (Claude Sonnet UUT, Haiku judge)
    MODEL_UNDER_TEST=claude-sonnet-4-6 \\
      animus-forge eval run sovereignty-stack-v0 \\
        --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \\
        --rubric personal-quality \\
        --model claude-haiku-4-5 \\
        --prompt-version v0-cloud-baseline

    # Local UUT via llama-server (Haiku judge — cost-asymmetric, cheap)
    MODEL_UNDER_TEST=hauhaucs \\
      ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 \\
      animus-forge eval run sovereignty-stack-v0 \\
        --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \\
        --rubric personal-quality \\
        --model claude-haiku-4-5 \\
        --prompt-version v0-local-uut

The reciprocal pattern (local UUT + local judge) needs a llama-server
provider in the Forge provider stack. Not wired yet; tracked as
follow-up.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"


def _call_oai_compat(query: str, model: str, base_url: str, max_tokens: int) -> str:
    """Call an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Disables Qwen3 thinking via ``chat_template_kwargs`` (llama-server
    honors; ollama ignores silently) and strips any inline
    ``<think>...</think>`` block defensively.
    """
    import httpx

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    content = (msg.get("content") or "").strip()
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    return content


def _call_provider(query: str, model: str, max_tokens: int) -> str:
    """Call the configured Forge provider with ``model`` as default."""
    from animus_forge.providers import get_provider
    from animus_forge.providers.base import CompletionRequest

    provider = get_provider()
    provider.config.default_model = model
    req = CompletionRequest(
        prompt=query,
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    resp = provider.complete(req)
    return (resp.content or "").strip()


def run_query(case_input: str | dict[str, Any]) -> str:
    """Eval adapter — routes ``case.input`` to the model under test.

    Args:
        case_input: Eval case input. Either a dict with ``query`` and
            optional ``max_tokens``, or a plain string treated as the
            query with ``max_tokens=512``.

    Returns:
        Plain-text response from the model under test. Empty string is
        possible (some prompts produce empty output); the rubric judge
        will score that low rather than the adapter raising.
    """
    if isinstance(case_input, str):
        query = case_input
        max_tokens = 512
    else:
        query = case_input["query"]
        max_tokens = int(case_input.get("max_tokens") or 512)

    model = os.environ.get("MODEL_UNDER_TEST", DEFAULT_MODEL)
    base_url = os.environ.get("ANIMUS_SOVEREIGNTY_BASE_URL")

    if base_url:
        return _call_oai_compat(query, model, base_url, max_tokens)
    return _call_provider(query, model, max_tokens)


__all__ = ["run_query"]
