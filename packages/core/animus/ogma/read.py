"""``/ogma read`` — synthesize one cached lugh item into an ``OgmaOutput``.

Pipeline: ``SourceItem`` → ``verify_animus_gap`` against the animus repo →
assemble system + user prompts → ``ModelInterface.generate(prompt, system)``
→ ``OgmaOutput.from_markdown`` → ``.write()``.

Default provider is Ollama (``ModelConfig.ollama()``). Per
``[[ADL-20260511-001]] #2``, a 401/403 on a configured provider MUST NOT
silently fall back to Ollama — exceptions propagate as
``OgmaSynthesisError``. Pass an explicit ``model=CognitiveLayer(...)`` to
opt into a different provider.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from animus.cognitive import CognitiveLayer, ModelConfig, ModelInterface, create_model
from animus.lugh.sources.base import SourceItem
from animus.ogma.grounding import GapResult, verify_animus_gap
from animus.ogma.models import OgmaOutput, OgmaParseError

logger = logging.getLogger(__name__)

DEFAULT_MIN_RELEVANCE = 0.5
DEFAULT_FAILURE_DIR = Path("~/projects/notes/ogma/.failures").expanduser()

PERSONA_SYSTEM_PROMPT = """You are Ogma — the reverse-engineering synthesis persona for the Animus exocortex project. Companion to Lugh. Ethos: figure out how it works, then build it better.

Your output MUST follow this exact markdown contract, with every section non-empty and in this order:

# <Title of the source work>

**Source:** <item_id>  •  **Date:** <YYYY-MM-DD>
**Cited from:** <source_id>

## Concept
<One paragraph. What is this, really. No hedging. No "this paper proposes to possibly explore.">

## Novelty
<What's actually new here vs prior art. Cite predecessors where known. If reheated, say so.>

## Animus gap
**Status:** <NONE | PARTIAL | FULL>
<If PARTIAL/FULL: the exact file(s) + function(s) in the animus repo that implement the overlapping concept. Use ONLY the file paths listed in the grounding context below — never invent paths.>

## Weaknesses in the source
<What's hand-wavy, unreproducible, missing ablations, or load-bearing on bad assumptions.>

## Proposal — how we build it better
<Concrete. Name the module (existing or new) in the animus namespace. Sketch the change. Explicitly call out how this version improves on the source (sharper contract, better Forge/Quorum composition, smaller deps, reproducibility test, etc.). If Animus gap is NONE, greenfield-propose — never refuse.>

## ROI
**Value:** <one line — what this unlocks>
**Effort:** <trivial | moderate | substantial>
**Priority:** <why now / why later>

## Risks
<Reproducibility, maturity, licensing, perf, scope creep, coupling.>

## Confidence
<0.00–1.00> — <one-line justification>

## Sources cited
- <source URL or id>
- <every animus file:line ref you cited in Animus gap or Proposal — must come from the grounding context>

NON-NEGOTIABLES:
- Every required section above MUST be present and non-empty.
- ROI Effort MUST be exactly one of: trivial, moderate, substantial.
- Confidence MUST be a number in [0.00, 1.00] formatted to 2 decimals.
- Animus file paths in your output MUST come from the grounding context — never invent.
- No preamble, no closing remarks, no markdown other than the contract above.
"""


class OgmaSynthesisError(RuntimeError):
    """Raised when the LLM call or response parse fails."""


def synthesize(
    item: SourceItem,
    *,
    model: ModelInterface | CognitiveLayer | None = None,
    repo_root: Path | None = None,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
    relevance_score: float | None = None,
    concept_hint: str | None = None,
    output_dir: Path | None = None,
) -> OgmaOutput | None:
    """Synthesize one ``SourceItem`` → ``OgmaOutput`` (written to disk).

    Args:
        item: the cached lugh item to read.
        model: a ``ModelInterface`` (provider) or ``CognitiveLayer``. When
            None, defaults to ``create_model(ModelConfig.ollama())`` — Ollama
            with no Anthropic key required.
        repo_root: animus repo path passed to ``verify_animus_gap``; defaults
            to ``~/projects/animus``.
        min_relevance: items scoring below this are skipped (returns None).
        relevance_score: caller-supplied score; if None, the gate passes.
        concept_hint: pin the gap-check concept; if None, derive from title.
        output_dir: override the default ``~/projects/notes/ogma/``.

    Returns:
        The written ``OgmaOutput``, or ``None`` if the item was gated out.

    Raises:
        ValueError: item has no usable text (raw_text/summary/title all empty).
        OgmaSynthesisError: the provider call raised, or the response could
            not be parsed into the contract. The raw response is dumped to
            ``~/projects/notes/ogma/.failures/`` for diagnosis.
    """
    if relevance_score is not None and relevance_score < min_relevance:
        logger.info(
            "ogma.read: skipping %s (score %.3f < %.3f)",
            item.fingerprint[:12],
            relevance_score,
            min_relevance,
        )
        return None

    source_text = item.raw_text or item.summary or item.title
    if not source_text or not source_text.strip():
        raise ValueError("read: source has no text to synthesize")

    provider = _resolve_provider(model)
    concept = concept_hint or item.title or item.summary[:120] or item.item_id
    gap = verify_animus_gap(concept, repo_root=repo_root)
    prompt = _assemble_prompt(item, source_text, gap)

    try:
        response = provider.generate(prompt, system=PERSONA_SYSTEM_PROMPT)
    except Exception as e:
        raise OgmaSynthesisError(f"provider.generate raised: {e}") from e

    try:
        output = OgmaOutput.from_markdown(response, source_id=item.source_id, item_id=item.item_id)
    except OgmaParseError as e:
        dump = _dump_failure(item, response)
        raise OgmaSynthesisError(
            f"could not parse synthesis response: {e}; raw response at {dump}"
        ) from e

    output.write(output_dir=output_dir)
    return output


def _resolve_provider(
    model: ModelInterface | CognitiveLayer | None,
) -> ModelInterface:
    """Default to Ollama; unwrap a CognitiveLayer's primary provider."""
    if model is None:
        return create_model(ModelConfig.ollama())
    if isinstance(model, CognitiveLayer):
        return model.primary
    return model


def _assemble_prompt(item: SourceItem, source_text: str, gap: GapResult) -> str:
    """Build the user-message text that pairs with the persona system prompt.

    Local instruction-following models (llama3, qwen2.5) lose track of long
    system prompts when the user message is large. We repeat the format
    contract at the *tail* of the user prompt — where the model's attention
    is freshest before generation — so the first response token is more
    likely to be the required ``# <Title>`` heading.
    """
    grounding_block = _format_grounding(gap)
    return (
        f"# Source\n"
        f"- title: {item.title}\n"
        f"- source_id: {item.source_id}\n"
        f"- item_id: {item.item_id}\n"
        f"- url: {item.url}\n"
        f"\n"
        f"# Grounding (the ONLY animus paths you may cite)\n"
        f"{grounding_block}\n"
        f"\n"
        f"# Source text\n"
        f"{source_text}\n"
        f"\n"
        f"---\n"
        f"\n"
        f"Now produce the Ogma synthesis for this source. Your response MUST\n"
        f"begin with the line `# {item.title}` (the source's title as a level-1\n"
        f"markdown heading) — no preamble, no acknowledgement, no '<think>'\n"
        f"block. After that line, emit each of the nine `## ` sections in the\n"
        f"order specified by the system prompt: Concept, Novelty, Animus gap,\n"
        f"Weaknesses in the source, Proposal — how we build it better, ROI,\n"
        f"Risks, Confidence, Sources cited. Every section MUST be non-empty.\n"
        f"For the ROI block, **Effort:** MUST be one of: trivial, moderate,\n"
        f"substantial. For Confidence, format as `0.NN — <one-line justification>`.\n"
    )


def _format_grounding(gap: GapResult) -> str:
    if gap.status == "NONE":
        return (
            "Status: NONE — animus does not currently implement this concept. "
            "Greenfield-propose: name the new module path you would add, the "
            "abstractions it composes with, and the public API shape. Do NOT "
            "refuse to propose."
        )
    paths = "\n".join(f"- {p}" for p in gap.paths_read) or "(none)"
    return f"Status: {gap.status}\nNotes: {gap.notes}\nPaths read:\n{paths}"


def _dump_failure(item: SourceItem, response: str) -> Path:
    DEFAULT_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    safe_id = item.item_id.replace("/", "_").replace(":", "_")[:60]
    path = DEFAULT_FAILURE_DIR / f"{ts}-{safe_id}.txt"
    path.write_text(response, encoding="utf-8")
    return path
