"""Prompt template evolution with A/B testing.

Evolves prompt templates based on outcome data. Variants are tested
against each other using epsilon-greedy selection, and winners are
promoted when statistically significant improvements are observed.
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.3  # Exponential moving average smoothing factor
_MIN_TRIALS = 10  # Minimum trials before declaring a winner
_MIN_IMPROVEMENT = 0.10  # 10% improvement threshold
_EPSILON = 0.10  # Exploration probability


@dataclass
class PromptVariant:
    """A single prompt variant registered for A/B testing.

    Attributes:
        variant_id: Unique identifier for this variant.
        base_template_id: The base template this variant derives from.
        user_prompt: The user-facing prompt text.
        system_prompt: Optional system prompt override.
        metadata: Arbitrary key-value metadata.
        created_at: UTC timestamp when the variant was created.
    """

    variant_id: str
    base_template_id: str
    user_prompt: str
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class VariantStats:
    """Aggregate performance statistics for a variant.

    Attributes:
        variant_id: The variant these stats belong to.
        trials: Number of recorded outcomes.
        avg_quality: Exponential moving average of quality scores.
        success_rate: Fraction of successful outcomes (0.0-1.0).
        avg_tokens: Running average of tokens used per call.
        avg_latency_ms: Running average of latency in milliseconds.
    """

    variant_id: str
    trials: int = 0
    avg_quality: float = 0.0
    success_rate: float = 0.0
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class VariantReport:
    """Comparative report across all variants of a base template.

    Attributes:
        base_template_id: The base template being compared.
        variants: Per-variant statistics.
        winner: Variant ID of the current best performer, or None.
        confidence: Confidence level: "high", "medium", "low", or
            "insufficient_data".
        improvement_pct: How much better the winner is vs second best,
            as a percentage (e.g. 15.0 means 15% better).
    """

    base_template_id: str
    variants: list[VariantStats]
    winner: str | None = None
    confidence: str = "insufficient_data"
    improvement_pct: float = 0.0


# ---------------------------------------------------------------------------
# Internal mutable stats tracker (not exposed publicly)
# ---------------------------------------------------------------------------


@dataclass
class _RunningStats:
    """Mutable running statistics for a variant."""

    trials: int = 0
    ema_quality: float = 0.0
    total_successes: int = 0
    total_tokens: float = 0.0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# PromptEvolution
# ---------------------------------------------------------------------------


class PromptEvolution:
    """A/B testing and evolution engine for prompt templates.

    Manages prompt variants, selects which variant to use via
    epsilon-greedy exploration, records outcomes, and promotes
    winners when statistically justified.

    All state is held in-memory and is thread-safe.

    Example::

        evo = PromptEvolution()
        evo.register_variant("review_tmpl", "v1", "Review this code: {code}")
        evo.register_variant("review_tmpl", "v2", "Carefully review: {code}")
        chosen = evo.select_variant("review_tmpl", agent_role="reviewer")
        evo.record_variant_outcome(chosen.variant_id, 0.9, True, 500, 1200)
        report = evo.get_variant_report("review_tmpl")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # base_template_id -> list[PromptVariant]
        self._variants: dict[str, list[PromptVariant]] = {}
        # variant_id -> _RunningStats
        self._stats: dict[str, _RunningStats] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_variant(
        self,
        base_template_id: str,
        variant_id: str,
        user_prompt: str,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptVariant:
        """Register a new prompt variant for A/B testing.

        Args:
            base_template_id: The base template this variant belongs to.
            variant_id: Unique identifier for the variant.
            user_prompt: The user prompt text.
            system_prompt: Optional system prompt override.
            metadata: Arbitrary metadata dict.

        Returns:
            The created ``PromptVariant``.

        Raises:
            ValueError: If a variant with the same ID already exists.
        """
        variant = PromptVariant(
            variant_id=variant_id,
            base_template_id=base_template_id,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        with self._lock:
            existing = self._variants.setdefault(base_template_id, [])
            for v in existing:
                if v.variant_id == variant_id:
                    raise ValueError(
                        f"Variant '{variant_id}' already registered for "
                        f"base template '{base_template_id}'"
                    )
            existing.append(variant)
            self._stats.setdefault(variant_id, _RunningStats())

        logger.info(
            "Registered variant '%s' for base template '%s'",
            variant_id,
            base_template_id,
        )
        return variant

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_variant(
        self,
        base_template_id: str,
        agent_role: str | None = None,
    ) -> PromptVariant:
        """Select a prompt variant using epsilon-greedy strategy.

        If no outcome data exists for any variant, a random variant is
        chosen (pure exploration). Otherwise, the best-performing
        variant is selected 90% of the time (exploitation) and a random
        variant 10% of the time (exploration).

        Args:
            base_template_id: The base template whose variants to choose from.
            agent_role: Optional agent role for logging context.

        Returns:
            The selected ``PromptVariant``.

        Raises:
            KeyError: If no variants are registered for the base template.
        """
        with self._lock:
            variants = self._variants.get(base_template_id)
            if not variants:
                raise KeyError(f"No variants registered for base template '{base_template_id}'")

            # Check if any variant has outcome data
            has_data = any(
                self._stats.get(v.variant_id, _RunningStats()).trials > 0 for v in variants
            )

            if not has_data:
                chosen = random.choice(variants)
                logger.debug(
                    "No outcome data for '%s'; randomly selected variant '%s'",
                    base_template_id,
                    chosen.variant_id,
                )
                return chosen

            # Epsilon-greedy: exploit best 90%, explore 10%
            if random.random() < _EPSILON:
                chosen = random.choice(variants)
                logger.debug(
                    "Exploration: randomly selected variant '%s' for '%s'",
                    chosen.variant_id,
                    base_template_id,
                )
                return chosen

            # Pick the variant with the highest EMA quality
            best_variant = max(
                variants,
                key=lambda v: self._stats.get(v.variant_id, _RunningStats()).ema_quality,
            )
            logger.debug(
                "Exploitation: selected best variant '%s' (quality=%.3f) for '%s'",
                best_variant.variant_id,
                self._stats[best_variant.variant_id].ema_quality,
                base_template_id,
            )
            return best_variant

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_variant_outcome(
        self,
        variant_id: str,
        quality_score: float,
        success: bool,
        tokens_used: int,
        latency_ms: float,
    ) -> None:
        """Record how a variant performed in a real execution.

        Updates running statistics using exponential moving average for
        quality and simple cumulative averages for other metrics.

        Args:
            variant_id: The variant that was used.
            quality_score: Quality rating from 0.0 to 1.0.
            success: Whether the execution succeeded.
            tokens_used: Total tokens consumed.
            latency_ms: Wall-clock latency in milliseconds.

        Raises:
            KeyError: If the variant ID is not registered.
        """
        with self._lock:
            stats = self._stats.get(variant_id)
            if stats is None:
                raise KeyError(f"Unknown variant '{variant_id}'")

            n = stats.trials
            if n == 0:
                stats.ema_quality = quality_score
            else:
                stats.ema_quality = (
                    _EMA_ALPHA * quality_score + (1 - _EMA_ALPHA) * stats.ema_quality
                )

            stats.total_successes += int(success)
            stats.total_tokens += tokens_used
            stats.total_latency_ms += latency_ms
            stats.trials = n + 1

        logger.debug(
            "Recorded outcome for variant '%s': quality=%.3f success=%s",
            variant_id,
            quality_score,
            success,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _build_variant_stats(self, variant_id: str) -> VariantStats:
        """Build a ``VariantStats`` snapshot for a variant.

        Args:
            variant_id: The variant to summarise.

        Returns:
            A ``VariantStats`` dataclass. Caller must hold ``self._lock``.
        """
        rs = self._stats.get(variant_id, _RunningStats())
        trials = rs.trials
        return VariantStats(
            variant_id=variant_id,
            trials=trials,
            avg_quality=rs.ema_quality,
            success_rate=(rs.total_successes / trials) if trials > 0 else 0.0,
            avg_tokens=(rs.total_tokens / trials) if trials > 0 else 0.0,
            avg_latency_ms=(rs.total_latency_ms / trials) if trials > 0 else 0.0,
        )

    def get_variant_report(self, base_template_id: str) -> VariantReport:
        """Generate a comparative report for all variants of a base template.

        A winner is declared only when at least two variants have
        ``_MIN_TRIALS`` (10) trials each and the best variant shows
        more than ``_MIN_IMPROVEMENT`` (10%) quality improvement over
        the second best.

        Args:
            base_template_id: The base template to report on.

        Returns:
            A ``VariantReport`` with per-variant stats, winner, confidence,
            and improvement percentage.

        Raises:
            KeyError: If no variants are registered for the base template.
        """
        with self._lock:
            variants = self._variants.get(base_template_id)
            if not variants:
                raise KeyError(f"No variants registered for base template '{base_template_id}'")

            all_stats = [self._build_variant_stats(v.variant_id) for v in variants]

        # Sort by avg_quality descending
        ranked = sorted(all_stats, key=lambda s: s.avg_quality, reverse=True)

        # Determine winner and confidence
        winner: str | None = None
        confidence = "insufficient_data"
        improvement_pct = 0.0

        if len(ranked) >= 2:
            best = ranked[0]
            second = ranked[1]
            both_sufficient = best.trials >= _MIN_TRIALS and second.trials >= _MIN_TRIALS

            if both_sufficient:
                if second.avg_quality > 0:
                    improvement_pct = (
                        (best.avg_quality - second.avg_quality) / second.avg_quality
                    ) * 100.0
                elif best.avg_quality > 0:
                    improvement_pct = 100.0

                if improvement_pct > _MIN_IMPROVEMENT * 100:
                    winner = best.variant_id
                    if best.trials >= 30 and second.trials >= 30:
                        confidence = "high"
                    elif best.trials >= 20 and second.trials >= 20:
                        confidence = "medium"
                    else:
                        confidence = "low"
                else:
                    confidence = "low"
        elif len(ranked) == 1 and ranked[0].trials >= _MIN_TRIALS:
            winner = ranked[0].variant_id
            confidence = "low"

        return VariantReport(
            base_template_id=base_template_id,
            variants=all_stats,
            winner=winner,
            confidence=confidence,
            improvement_pct=round(improvement_pct, 2),
        )

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_winner(self, base_template_id: str) -> str | None:
        """Promote the winning variant as the new default.

        A variant is promoted only if it has at least ``_MIN_TRIALS``
        trials, another variant also has ``_MIN_TRIALS`` trials, and
        the winner shows more than ``_MIN_IMPROVEMENT`` quality
        improvement over the runner-up.

        When promoted, the winning variant is moved to the front of
        the variant list and all other variants' stats are reset so
        that future comparisons start fresh against the new champion.

        Args:
            base_template_id: The base template to evaluate.

        Returns:
            The winning variant ID, or ``None`` if no clear winner.

        Raises:
            KeyError: If no variants are registered for the base template.
        """
        report = self.get_variant_report(base_template_id)
        if report.winner is None:
            logger.info("No clear winner for '%s'; skipping promotion", base_template_id)
            return None

        winner_id = report.winner

        with self._lock:
            variants = self._variants.get(base_template_id, [])
            # Move the winner to the front
            winner_variant = None
            rest = []
            for v in variants:
                if v.variant_id == winner_id:
                    winner_variant = v
                else:
                    rest.append(v)
                    # Reset stats for non-winners so future tests are fair
                    self._stats[v.variant_id] = _RunningStats()

            if winner_variant is not None:
                self._variants[base_template_id] = [winner_variant] + rest

        logger.info(
            "Promoted variant '%s' as winner for '%s' (improvement=%.1f%%)",
            winner_id,
            base_template_id,
            report.improvement_pct,
        )
        return winner_id

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_prompt(
        self,
        base_template_id: str,
        agent_role: str,
        outcome_history: list[dict[str, Any]],
    ) -> str | None:
        """Auto-generate a new variant based on failure patterns.

        Analyses the provided outcome history for common failure
        patterns and appends corrective instructions to the
        best-performing variant's prompt.

        Each outcome dict is expected to have at least:
        - ``quality_score`` (float): 0.0 to 1.0
        - ``success`` (bool): whether the step passed
        - ``error`` (str, optional): error description if failed
        - ``missing_fields`` (list[str], optional): fields that were missing

        Args:
            base_template_id: The base template to evolve.
            agent_role: The agent role context (for logging).
            outcome_history: List of outcome dicts from recent executions.

        Returns:
            The new variant ID if a variant was created, or ``None`` if
            no evolution is needed.

        Raises:
            KeyError: If no variants are registered for the base template.
        """
        if not outcome_history:
            logger.debug("No outcome history provided; skipping evolution")
            return None

        # Analyse failure patterns
        total = len(outcome_history)
        failures = [o for o in outcome_history if not o.get("success", True)]
        low_quality = [o for o in outcome_history if o.get("quality_score", 1.0) < 0.5]

        failure_rate = len(failures) / total if total > 0 else 0.0
        low_quality_rate = len(low_quality) / total if total > 0 else 0.0

        # No evolution needed if things are going well
        if failure_rate < 0.2 and low_quality_rate < 0.2:
            logger.debug(
                "Outcomes for '%s' are acceptable (fail=%.0f%%, low_q=%.0f%%); no evolution needed",
                base_template_id,
                failure_rate * 100,
                low_quality_rate * 100,
            )
            return None

        # Gather corrective instructions
        addenda: list[str] = []

        # Collect missing fields across failures
        missing_fields: dict[str, int] = {}
        for o in failures + low_quality:
            for f in o.get("missing_fields", []):
                missing_fields[f] = missing_fields.get(f, 0) + 1

        if missing_fields:
            # Sort by frequency
            sorted_fields = sorted(missing_fields.items(), key=lambda kv: kv[1], reverse=True)
            field_list = ", ".join(f for f, _ in sorted_fields[:5])
            addenda.append(f"Make sure to include: {field_list}.")

        # Collect common error patterns
        error_counts: dict[str, int] = {}
        for o in failures:
            err = o.get("error", "")
            if err:
                # Normalise to first 80 chars to group similar errors
                key = err[:80].strip()
                error_counts[key] = error_counts.get(key, 0) + 1

        if error_counts:
            top_errors = sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            for err_msg, count in top_errors:
                addenda.append(f"Avoid this error ({count} occurrences): {err_msg}")

        # If high failure rate but no specific patterns, add generic guidance
        if not addenda and failure_rate >= 0.3:
            addenda.append(
                "Pay close attention to output format and completeness. "
                "Double-check all required fields before responding."
            )

        if not addenda and low_quality_rate >= 0.3:
            addenda.append(
                "Focus on producing higher-quality, more detailed output. "
                "Ensure thoroughness and accuracy."
            )

        if not addenda:
            return None

        # Build new variant from the best existing one
        with self._lock:
            variants = self._variants.get(base_template_id)
            if not variants:
                raise KeyError(f"No variants registered for base template '{base_template_id}'")

            # Pick variant with best quality as the base
            best = max(
                variants,
                key=lambda v: self._stats.get(v.variant_id, _RunningStats()).ema_quality,
            )
            base_prompt = best.user_prompt
            base_system = best.system_prompt

        # Append corrective instructions
        evolution_suffix = "\n\nIMPORTANT additional instructions:\n" + "\n".join(
            f"- {a}" for a in addenda
        )
        new_user_prompt = base_prompt + evolution_suffix

        new_variant_id = f"{base_template_id}_evolved_{uuid.uuid4().hex[:8]}"

        self.register_variant(
            base_template_id=base_template_id,
            variant_id=new_variant_id,
            user_prompt=new_user_prompt,
            system_prompt=base_system,
            metadata={
                "evolved_from": best.variant_id,
                "agent_role": agent_role,
                "failure_rate": round(failure_rate, 3),
                "low_quality_rate": round(low_quality_rate, 3),
                "addenda": addenda,
            },
        )

        logger.info(
            "Evolved variant '%s' from '%s' for role '%s' with %d corrective rules",
            new_variant_id,
            best.variant_id,
            agent_role,
            len(addenda),
        )
        return new_variant_id
