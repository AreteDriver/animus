"""Reflection proactive check — periodic self-reflection that updates LEARNED.md."""

from __future__ import annotations

import logging

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck

logger = logging.getLogger(__name__)

# Module-level references, wired at runtime via set_reflection_deps()
_identity_manager = None
_feedback_store = None
_memory_manager = None
_cognitive_backend = None
_config = None


def set_reflection_deps(
    identity_manager=None,  # noqa: ANN001
    feedback_store=None,  # noqa: ANN001
    memory_manager=None,  # noqa: ANN001
    cognitive_backend=None,  # noqa: ANN001
    config=None,  # noqa: ANN001
) -> None:
    """Wire runtime dependencies into the reflection check."""
    global _identity_manager, _feedback_store, _memory_manager  # noqa: PLW0603
    global _cognitive_backend, _config  # noqa: PLW0603
    _identity_manager = identity_manager
    _feedback_store = feedback_store
    _memory_manager = memory_manager
    _cognitive_backend = cognitive_backend
    _config = config


async def _run_reflection() -> str | None:
    """Run a reflection cycle — summarize interactions and update LEARNED.md.

    Steps:
        1. Check minimum interaction threshold from config
        2. Gather recent feedback signals
        3. Read current LEARNED.md to avoid duplication
        4. Build reflection prompt
        5. If cognitive backend available, get AI-generated insights
        6. Otherwise, generate summary from feedback patterns
        7. Append new findings to LEARNED.md
        8. Return nudge text or None
    """
    if _identity_manager is None:
        logger.debug("Reflection skipped — no identity manager")
        return None

    # Gather feedback if available
    feedback_summary = ""
    if _feedback_store is not None:
        stats = _feedback_store.get_stats()
        if stats["total"] == 0:
            logger.debug("Reflection skipped — no feedback data")
            return None

        positive = _feedback_store.get_positive_patterns(limit=10)
        negative = _feedback_store.get_negative_patterns(limit=10)

        if positive:
            feedback_summary += "Positive feedback patterns:\n"
            for p in positive:
                feedback_summary += f"  - Q: {p['message_text'][:80]} → thumbs up\n"

        if negative:
            feedback_summary += "Negative feedback patterns:\n"
            for n in negative:
                feedback_summary += f"  - Q: {n['message_text'][:80]} → thumbs down"
                if n.get("comment"):
                    feedback_summary += f" ({n['comment']})"
                feedback_summary += "\n"

    # Read current LEARNED.md to provide context
    current_learned = _identity_manager.read("LEARNED.md")

    # Try to get AI-generated reflection
    if _cognitive_backend is not None and feedback_summary:
        user_name = "the user"
        if _config is not None:
            user_name = getattr(_config, "identity", None)
            if user_name:
                user_name = getattr(user_name, "name", "the user") or "the user"

        reflection_prompt = (
            f"You are Animus, reviewing your recent interactions with {user_name}.\n\n"
            f"Feedback from recent interactions:\n{feedback_summary}\n"
            f"Current LEARNED.md:\n{current_learned[:1000]}\n\n"
            "What new preferences, patterns, or facts did you observe? "
            "What worked well? What should you do differently? "
            "Write ONLY the new additions — not the full file. "
            "Format as bullet points. Be specific. Max 5 new items."
        )

        try:
            messages = [{"role": "user", "content": reflection_prompt}]
            response = await _cognitive_backend.generate_response(
                messages, system_prompt="You are a self-reflecting AI assistant."
            )
            # Parse bullet points from response
            new_entries = _parse_reflection_entries(response)
            if new_entries:
                for entry in new_entries:
                    _identity_manager.append_to_learned("Reflection", entry)
                logger.info("Reflection added %d entries to LEARNED.md", len(new_entries))
                return f"Reflection complete — {len(new_entries)} new insights recorded."
        except Exception:
            logger.exception("AI-powered reflection failed, using fallback")

    # Fallback: generate simple entries from feedback patterns
    entries_added = 0
    if _feedback_store is not None:
        negative = _feedback_store.get_negative_patterns(limit=5)
        for n in negative:
            if n.get("comment"):
                _identity_manager.append_to_learned(
                    "Feedback", f"User disliked response — {n['comment']}"
                )
                entries_added += 1

        stats = _feedback_store.get_stats()
        if stats["total"] >= 5:
            _identity_manager.append_to_learned(
                "Reflection",
                f"Feedback summary: {stats['positive_pct']}% positive, "
                f"{stats['negative_pct']}% negative ({stats['total']} total)",
            )
            entries_added += 1

    if entries_added:
        return f"Reflection complete — {entries_added} entries added to LEARNED.md."
    return None


def _parse_reflection_entries(response: str) -> list[str]:
    """Parse bullet point entries from an LLM reflection response."""
    entries: list[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            entry = line.lstrip("-*• ").strip()
            if entry:
                entries.append(entry)
    return entries[:10]  # Cap at 10 entries


def get_reflection_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for periodic reflection."""
    return ProactiveCheck(
        name="reflection",
        schedule="0 3 * * *",  # Default: 3 AM daily
        checker=_run_reflection,
        channels=[],  # Reflection is internal — no external nudge by default
        priority="low",
        enabled=True,
    )
