"""Media Pipeline Orchestrator.

Coordinates Lugh (harvest), Ogma (synthesize), and the Research Guild citizens
(Pattern, First-Principles, Architecture) for media sources (YouTube playlists,
channels, podcasts).

Uses ``OgmaOutput.animus_gap`` as a gate for downstream analysis depth:

- ``NONE``     → store Ogma synthesis + MechanismCards only
- ``PARTIAL``  → store + run PatternCitizen (media-tuned clustering)
- ``FULL``     → store + run full Research Guild (Pattern → FP → Architecture)

Override with ``run_research_guild=True`` to force full pipeline regardless
of gap status.

This is an orchestrator, not a citizen — it delegates to existing citizens
rather than producing ``ImprovementProposal`` s directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.citizens.abstraction import AbstractionCitizen, MechanismCard
from animus.citizens.architecture_citizen import ArchitectureCitizen, GapAnalysis
from animus.citizens.first_principles import FirstPrinciplesCitizen, PrincipleCard
from animus.citizens.proposal_queue import ProposalQueue
from animus.citizens.pattern import PatternCitizen, PatternCard
from animus.citizens.proposal import ImprovementProposal
from animus.citizens.research_guild import StageResult
from animus.logging import get_logger
from animus.lugh.sources.base import SourceItem
from animus.lugh.sources.youtube import YouTubeSource, playlist_to_source_items
from animus.ogma.models import OgmaOutput

if TYPE_CHECKING:
    from animus.cognitive import ModelInterface
    from animus.memory import MemoryLayer

logger = get_logger("citizens.media")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class MediaPipelineReport:
    """Report produced by MediaPipelineOrchestrator."""

    stages: list[StageResult] = field(default_factory=list)
    ogma_output: OgmaOutput | None = None
    mechanisms: list[MechanismCard] = field(default_factory=list)
    patterns: list[PatternCard] = field(default_factory=list)
    principles: list[PrincipleCard] = field(default_factory=list)
    gaps: list[GapAnalysis] = field(default_factory=list)
    final_proposal: ImprovementProposal | None = None
    gap_status: str = "NONE"
    forced_rg: bool = False
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        parts = [
            f"Media Pipeline: gap={self.gap_status} forced_rg={self.forced_rg}",
            f"  Stages: {len(self.stages)}",
            f"  Mechanisms: {len(self.mechanisms)}",
            f"  Patterns: {len(self.patterns)}",
            f"  Principles: {len(self.principles)}",
            f"  Gaps: {len(self.gaps)}",
        ]
        if self.final_proposal:
            parts.append(f"  Final proposal: {self.final_proposal.id}")
        parts.append(f"  Duration: {self.duration_seconds:.1f}s")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Media Harvester — thin wrapper over Lugh
# ═══════════════════════════════════════════════════════════════════


class MediaHarvester:
    """Thin wrapper over Lugh source adapters."""

    def ingest_playlist(
        self,
        playlist_url: str,
        fetch_captions: bool = True,
        list_limit: int = 25,
        tags: list[str] | None = None,
    ) -> list[SourceItem]:
        """Harvest a YouTube playlist. Delegates to YouTubeSource."""
        logger.info("MediaHarvester: ingesting playlist %s", playlist_url)
        return playlist_to_source_items(
            playlist_url=playlist_url,
            fetch_captions=fetch_captions,
            list_limit=list_limit,
            tags=tags,
        )

    def ingest_channel(
        self,
        channel: str,
        fetch_captions: bool = True,
        list_limit: int = 25,
        tags: list[str] | None = None,
    ) -> list[SourceItem]:
        """Harvest a YouTube channel. Delegates to YouTubeSource."""
        logger.info("MediaHarvester: ingesting channel %s", channel)
        src = YouTubeSource(
            channel=channel,
            fetch_captions=fetch_captions,
            list_limit=list_limit,
            tags=list(tags) if tags else [],
        )
        return list(src.fetch(limit=list_limit))

    def ingest_text(self, text: str, identifier: str, title: str) -> SourceItem:
        """Harvest raw text as a SourceItem."""
        return SourceItem(
            source_id=f"text:{identifier}",
            item_id=identifier,
            title=title,
            url="",
            published=datetime.now(),
            summary=text[:500],
            raw_text=text,
            tags=["media", "text"],
        )


# ═══════════════════════════════════════════════════════════════════
# Media Synthesizer — thin wrapper over Ogma
# ═══════════════════════════════════════════════════════════════════


class MediaSynthesizer:
    """Thin wrapper over Ogma's synthesize()."""

    # Variant of Ogma's persona with a "## Cross-cutting themes" section
    # and media-aware framing. Kept as class constant — does NOT modify
    # Ogma's built-in prompt.
    PERSONA_SYSTEM_PROMPT_MEDIA = """You are Ogma — the reverse-engineering synthesis persona for the Animus exocortex project.

Your output MUST follow this exact markdown contract, with every section non-empty and in this order.

# <Title of the source work>

**Source:** <item_id>  •  **Date:** <YYYY-MM-DD>
**Cited from:** <source_id>

## Concept
<One paragraph. What is this, really.>

## Novelty
<What's actually new here vs prior art.>

## Animus gap
**Status:** <NONE | PARTIAL | FULL>
<If PARTIAL/FULL: the exact file(s) + function(s) in the Animus codebase that implement the overlapping concept. If NONE: one sentence saying Animus does not currently implement this concept.>

## Weaknesses in the source
<What's hand-wavy, unreproducible, or load-bearing on bad assumptions.>

## Proposal — how we build it better
<Concrete. Name the module (existing or new) in the Animus codebase namespace. Sketch the change.>

## ROI
**Value:** <one line>
**Effort:** <trivial | moderate | substantial>
**Priority:** <why now / why later>

## Risks
<Reproducibility, maturity, licensing, perf, scope creep, coupling.>

## Confidence
<0.00–1.00> — <one-line justification>

## Sources cited
- <source URL or id>

NON-NEGOTIABLES:
1. EVERY required section above MUST be present and non-empty.
2. "## ROI" MUST contain exactly three lines starting with "**Value:**", "**Effort:**", "**Priority:**".
3. "**Effort:**" MUST be exactly one of: trivial, moderate, substantial.
4. "## Confidence" MUST be a number in [0.00, 1.00] followed by " — " and a one-line justification.
5. "## Sources cited" MUST be a bullet list using "- " prefix.
6. "## Animus gap" MUST begin with "**Status:** NONE", "**Status:** PARTIAL", or "**Status:** FULL".
7. No preamble, no closing remarks.
"""

    def synthesize_corpus(
        self,
        items: list[SourceItem],
        model: ModelInterface | None = None,
        repo_root: Path | None = None,
    ) -> OgmaOutput:
        """Synthesize a corpus of media items into an OgmaOutput.

        Assembles combined transcript + metadata into a SourceItem,
        calls Ogma.synthesize(), returns OgmaOutput.
        """
        from animus.ogma.read import synthesize

        if not items:
            raise ValueError("synthesize_corpus: no items provided")

        # Build a combined corpus item
        combined_text_parts: list[str] = []
        for item in items:
            header = f"## {item.title}"
            meta = f"- ID: {item.item_id}"
            if item.url:
                meta += f" | URL: {item.url}"
            body = item.raw_text or item.summary or ""
            combined_text_parts.append(f"{header}\n{meta}\n\n{body}")

        combined_text = "\n\n---\n\n".join(combined_text_parts)

        corpus_item = SourceItem(
            source_id=items[0].source_id,
            item_id=f"corpus:{len(items)}",
            title=f"Media corpus ({len(items)} items)",
            url=items[0].url,
            published=items[0].published,
            summary=combined_text[:500],
            raw_text=combined_text,
            tags=list(set(t for item in items for t in item.tags)),
        )

        return synthesize(corpus_item, model=model, repo_root=repo_root)

    def synthesize_single(
        self,
        item: SourceItem,
        model: ModelInterface | None = None,
        repo_root: Path | None = None,
    ) -> OgmaOutput:
        """Synthesize a single media item. Delegates to Ogma."""
        from animus.ogma.read import synthesize

        return synthesize(item, model=model, repo_root=repo_root)


# ═══════════════════════════════════════════════════════════════════
# Media Abstraction Adapter — OgmaOutput → MechanismCard
# ═══════════════════════════════════════════════════════════════════


class MediaAbstractionAdapter:
    """Map OgmaOutput → MechanismCard(s) for the Research Guild."""

    def from_ogma_output(
        self,
        ogma: OgmaOutput,
        source_ids: list[str],
    ) -> list[MechanismCard]:
        """Extract MechanismCards from an OgmaOutput.

        Heuristic mapping:
        - ogma.concept → MechanismCard.description
        - First sentence of ogma.proposal → MechanismCard.name
        - ogma.confidence → MechanismCard.confidence
        - source_ids → MechanismCard.source_provenance
        - Infer category from tags/content

        If ogma.proposal contains multiple distinct ideas, splits into
        multiple cards (lightweight extraction).
        """
        cards: list[MechanismCard] = []

        # Primary card from the concept/proposal
        name = self._extract_name(ogma.proposal)
        category = self._infer_category(ogma.concept + " " + ogma.proposal)

        cards.append(
            MechanismCard(
                name=name,
                description=ogma.concept,
                source_provenance=list(source_ids),
                confidence=ogma.confidence,
                implementation_stripped=ogma.proposal[:500],
                category=category,
                tags=[category, "media", "ogma"],
            )
        )

        # Attempt to split proposal into multiple distinct ideas
        ideas = self._split_proposal(ogma.proposal)
        if len(ideas) > 1:
            for idea in ideas[1:]:
                idea_name = self._extract_name(idea)
                idea_cat = self._infer_category(idea)
                cards.append(
                    MechanismCard(
                        name=idea_name,
                        description=idea[:300],
                        source_provenance=list(source_ids),
                        confidence=max(ogma.confidence - 0.1, 0.3),
                        implementation_stripped=idea[:500],
                        category=idea_cat,
                        tags=[idea_cat, "media", "ogma"],
                    )
                )

        return cards

    def store_mechanisms(
        self,
        cards: list[MechanismCard],
        memory: MemoryLayer | None,
    ) -> list[str]:
        """Store mechanisms in Animus memory. Returns memory IDs."""
        if memory is None:
            logger.warning("Memory layer not available — mechanisms not persisted")
            return []

        ids: list[str] = []
        for card in cards:
            try:
                from animus.memory import MemoryType

                mem = memory.remember(
                    content=f"{card.name}: {card.description}",
                    memory_type=MemoryType.SEMANTIC,
                    tags=["abstraction", "research_guild", "mechanism", card.category, "media"] + card.tags,
                    metadata=card.to_dict(),
                )
                ids.append(str(getattr(mem, "id", "")))
                logger.info("Media mechanism '%s' stored in memory", card.name)
            except Exception as e:
                logger.error("Failed to store mechanism: %s", e)
        return ids

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_name(proposal_text: str) -> str:
        """Extract a short name from the first sentence of a proposal."""
        first_sentence = proposal_text.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        return first_sentence or "Media-derived mechanism"

    @staticmethod
    def _infer_category(text: str) -> str:
        """Infer a mechanism category from text content."""
        text_lower = text.lower()
        category_map: dict[str, list[str]] = {
            "performance": ["cache", "speed", "latency", "throughput", "optimize", "scal"],
            "reliability": ["retry", "fault", "resilien", "graceful", "timeout", "backoff"],
            "security": ["auth", "encrypt", "permission", "identity", "rbac", "secret"],
            "operations": ["observ", "metric", "trace", "log", "monitor", "telemetry"],
            "architecture": ["modular", "boundary", "layer", "interface", "abstraction", "coupling"],
            "quality": ["test", "mock", "contract", "validation", "assert", "verif"],
            "deployment": ["feature flag", "canary", "rollout", "toggle"],
            "ai-engineering": ["model", "llm", "prompt", "inference", "embedding", "agent"],
            "ux": ["interface", "user", "experience", "workflow", "interaction"],
            "documentation": ["doc", "readme", "guide", "tutorial", "reference"],
        }
        for category, keywords in category_map.items():
            for kw in keywords:
                if kw in text_lower:
                    return category
        return "general"

    @staticmethod
    def _split_proposal(proposal_text: str) -> list[str]:
        """Split a proposal into distinct ideas.

        Tries multiple strategies: paragraphs, bullet points, then sentence
        boundaries that introduce new actions (Add/Create/Build/Implement).
        """
        import re

        # Strategy 1: double newlines (paragraphs)
        raw = proposal_text.split("\n\n")
        ideas = [r.strip() for r in raw if len(r.strip()) > 30]

        # Strategy 2: bullet markers
        if len(ideas) <= 1:
            bullet_split = re.split(r"\n\s*[-*]\s+", proposal_text)
            if len(bullet_split) > 1:
                ideas = [r.strip() for r in bullet_split if len(r.strip()) > 30]

        # Strategy 3: sentences starting with action verbs that introduce new modules
        if len(ideas) <= 1:
            sentence_split = re.split(
                r"(?i)(?<=\.\s)(?=Add\s|Create\s|Build\s|Implement\s|Introduce\s|Extract\s|Refactor\s)",
                proposal_text,
            )
            if len(sentence_split) > 1:
                ideas = [r.strip() for r in sentence_split if len(r.strip()) > 30]

        return ideas[:3]  # Cap at 3 to avoid over-splitting


# ═══════════════════════════════════════════════════════════════════
# Media Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════


class MediaPipelineOrchestrator:
    """Orchestrate the full media pipeline with gap gating."""

    def __init__(
        self,
        memory_layer: MemoryLayer | None = None,
        codebase_path: Path | str = ".",
        proposal_queue: Any | None = None,
    ):
        self.memory = memory_layer
        self.codebase_path = Path(codebase_path).expanduser()
        self.proposal_queue = proposal_queue
        self.harvester = MediaHarvester()
        self.synthesizer = MediaSynthesizer()
        self.adapter = MediaAbstractionAdapter()

    def run(
        self,
        url: str,
        source_type: str = "auto",
        run_research_guild: bool = False,
        store_outputs: bool = True,
        model: ModelInterface | None = None,
        list_limit: int = 25,
    ) -> MediaPipelineReport:
        """Run the full media pipeline.

        Args:
            url: Media URL (YouTube playlist, channel, podcast feed).
            source_type: "auto" | "youtube_playlist" | "youtube_channel" | "podcast".
            run_research_guild: If True, force full RG downstream regardless of gap.
            store_outputs: If True, store all outputs in memory.
            model: Optional model override for Ogma synthesis.
            list_limit: Max items to harvest from the source.

        Returns:
            MediaPipelineReport with stage results and final artifact.
        """
        start = time.time()
        report = MediaPipelineReport()
        stages: list[StageResult] = []

        # ── Step 1: Harvest ─────────────────────────────────────────────
        stage_start = time.time()
        items = self._harstep(url, source_type, list_limit)
        stages.append(
            StageResult(
                citizen_name="MediaHarvester",
                outputs_count=len(items),
                stored_count=0,
                duration_seconds=time.time() - stage_start,
            )
        )

        if not items:
            logger.warning("MediaPipeline: no items harvested from %s", url)
            report.stages = stages
            report.duration_seconds = time.time() - start
            return report

        # ── Step 2: Synthesize → OgmaOutput ────────────────────────────
        stage_start = time.time()
        ogma_output = self._synthstep(items, model)
        stages.append(
            StageResult(
                citizen_name="MediaSynthesizer",
                outputs_count=1 if ogma_output else 0,
                stored_count=0,
                duration_seconds=time.time() - stage_start,
            )
        )

        if ogma_output is None:
            logger.warning("MediaPipeline: Ogma synthesis failed for %s", url)
            report.stages = stages
            report.duration_seconds = time.time() - start
            return report

        report.ogma_output = ogma_output
        report.gap_status = ogma_output.animus_gap

        # ── Step 3: Extract MechanismCards ──────────────────────────────
        stage_start = time.time()
        source_ids = [item.item_id for item in items]
        mechanisms = self.adapter.from_ogma_output(ogma_output, source_ids)
        report.mechanisms = mechanisms
        stages.append(
            StageResult(
                citizen_name="MediaAbstractionAdapter",
                outputs_count=len(mechanisms),
                stored_count=0,
                duration_seconds=time.time() - stage_start,
            )
        )

        # ── Step 4: Store in memory ────────────────────────────────────
        if store_outputs and self.memory is not None:
            self._store_ogma(ogma_output)
            stored_ids = self.adapter.store_mechanisms(mechanisms, self.memory)
            stages[-1].stored_count = len(stored_ids)

        # ── Step 5: Gate on ogma_output.animus_gap ─────────────────────
        gap = ogma_output.animus_gap
        forced = run_research_guild

        if gap == "NONE" and not forced:
            logger.info("MediaPipeline: gap=NONE, no downstream analysis")
            report.stages = stages
            report.duration_seconds = time.time() - start
            return report

        # For PARTIAL or FULL (or forced), run PatternCitizen
        stage_start = time.time()
        patterns = self._patternstep(mechanisms)
        report.patterns = patterns
        stages.append(
            StageResult(
                citizen_name="PatternCitizen",
                outputs_count=len(patterns),
                stored_count=0,
                duration_seconds=time.time() - stage_start,
            )
        )

        if store_outputs and self.memory is not None:
            pc = PatternCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
            for pattern in patterns:
                pc.store_pattern(pattern)
            stages[-1].stored_count = len(patterns)

        # For FULL or forced, run full Research Guild (Pattern → FP → Architecture)
        if gap == "FULL" or forced:
            report.forced_rg = forced and gap != "FULL"

            # First-Principles
            stage_start = time.time()
            principles = self._fpstep(patterns)
            report.principles = principles
            stages.append(
                StageResult(
                    citizen_name="FirstPrinciplesCitizen",
                    outputs_count=len(principles),
                    stored_count=0,
                    duration_seconds=time.time() - stage_start,
                )
            )

            if store_outputs and self.memory is not None:
                fpc = FirstPrinciplesCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
                for principle in principles:
                    fpc.store_principle(principle)
                stages[-1].stored_count = len(principles)

            # Architecture
            stage_start = time.time()
            gaps = self._archstep(principles)
            report.gaps = gaps
            stages.append(
                StageResult(
                    citizen_name="ArchitectureCitizen",
                    outputs_count=len(gaps),
                    stored_count=0,
                    duration_seconds=time.time() - stage_start,
                )
            )

            if store_outputs and self.memory is not None:
                ac = ArchitectureCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
                for gap_analysis in gaps:
                    ac.store_gap(gap_analysis)
                stages[-1].stored_count = len(gaps)

            # Final proposal from Architecture citizen
            if gaps:
                ac = ArchitectureCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
                proposal = ac.generate_proposal(gaps)
                report.final_proposal = proposal
                if store_outputs and self.memory is not None and proposal:
                    ac.store_proposal(proposal)
                # Submit to ProposalQueue for approval → commission → Forge pipeline
                if self.proposal_queue is not None and proposal:
                    try:
                        self.proposal_queue.submit(
                            proposal,
                            priority=3 if report.gap_status == "FULL" else 5,
                            tags=["media", "research_guild", f"gap:{report.gap_status.lower()}"],
                        )
                        logger.info(
                            "Proposal %s submitted to queue (priority=%d, tags=%s)",
                            proposal.id,
                            3 if report.gap_status == "FULL" else 5,
                            ["media", "research_guild", f"gap:{report.gap_status.lower()}"],
                        )
                    except Exception as e:
                        logger.error("Failed to submit proposal %s to queue: %s", proposal.id, e)

        report.stages = stages
        report.duration_seconds = time.time() - start
        logger.info("MediaPipeline complete: %s", report.summary())
        return report

    # -- step helpers ------------------------------------------------------

    def _harstep(self, url: str, source_type: str, list_limit: int) -> list[SourceItem]:
        """Harvest items from the given URL."""
        st = source_type.lower()
        if st == "auto":
            if "playlist?list=" in url:
                st = "youtube_playlist"
            elif "youtube.com/" in url or "youtu.be/" in url:
                st = "youtube_channel"
            else:
                st = "text"

        if st == "youtube_playlist":
            return self.harvester.ingest_playlist(url, list_limit=list_limit)
        elif st == "youtube_channel":
            # Extract channel handle from URL or use as-is
            return self.harvester.ingest_channel(url, list_limit=list_limit)
        elif st == "text":
            return [self.harvester.ingest_text(url, identifier="manual", title="Manual input")]
        else:
            logger.warning("Unsupported source_type: %s", source_type)
            return []

    def _synthstep(
        self,
        items: list[SourceItem],
        model: ModelInterface | None,
    ) -> OgmaOutput | None:
        """Synthesize harvested items into an OgmaOutput."""
        try:
            return self.synthesizer.synthesize_corpus(items, model=model, repo_root=self.codebase_path)
        except Exception as e:
            logger.warning("MediaPipeline synthesis failed: %s", e)
            return None

    def _patternstep(self, mechanisms: list[MechanismCard]) -> list[PatternCard]:
        """Run PatternCitizen on media-derived mechanisms."""
        pc = PatternCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
        mech_dicts = [m.to_dict() for m in mechanisms]
        return pc.discover_patterns(mech_dicts)

    def _fpstep(self, patterns: list[PatternCard]) -> list[PrincipleCard]:
        """Run FirstPrinciplesCitizen on media-derived patterns."""
        fpc = FirstPrinciplesCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
        pattern_dicts = [p.to_dict() for p in patterns]
        return fpc.reduce_to_principles(pattern_dicts)

    def _archstep(self, principles: list[PrincipleCard]) -> list[GapAnalysis]:
        """Run ArchitectureCitizen on media-derived principles."""
        ac = ArchitectureCitizen(memory_layer=self.memory, codebase_path=self.codebase_path)
        principle_dicts = [p.to_dict() for p in principles]
        return ac.analyze_gaps(principle_dicts)

    # -- persistence helpers -----------------------------------------------

    def _store_ogma(self, ogma: OgmaOutput) -> bool:
        """Store OgmaOutput markdown in memory."""
        if self.memory is None:
            return False
        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=ogma.to_markdown(),
                memory_type=MemoryType.SEMANTIC,
                tags=["ogma", "media", f"playlist:{ogma.item_id}"],
                metadata={
                    "title": ogma.title,
                    "source_id": ogma.source_id,
                    "item_id": ogma.item_id,
                    "animus_gap": ogma.animus_gap,
                    "confidence": ogma.confidence,
                },
            )
            logger.info("OgmaOutput stored in memory: %s", ogma.title)
            return True
        except Exception as e:
            logger.error("Failed to store OgmaOutput: %s", e)
            return False

    # -- scheduling --------------------------------------------------------

    @staticmethod
    def schedule_scan(
        scheduler: Any,
        url: str,
        source_type: str = "auto",
        cron_expression: str = "0 9 * * 1",  # Mondays at 9am
        run_research_guild: bool = False,
        list_limit: int = 25,
        priority: str = "normal",
    ) -> Any:
        """Schedule a recurring media pipeline scan via the daemon TaskScheduler.

        Args:
            scheduler: ``TaskScheduler`` instance.
            url: Media URL to scan.
            source_type: Source type override.
            cron_expression: Cron schedule (default: weekly Monday 9am).
            run_research_guild: Force full RG regardless of gap.
            list_limit: Max items to harvest per run.
            priority: Task priority (normal, high, critical).

        Returns:
            ``ScheduledTask`` object.
        """
        task = scheduler.schedule_cron(
            description=f"Media pipeline scan: {url}",
            cron_expression=cron_expression,
            priority=priority,
            metadata={
                "task_type": "media_pipeline",
                "url": url,
                "source_type": source_type,
                "run_research_guild": run_research_guild,
                "list_limit": list_limit,
            },
        )
        logger.info(
            "Scheduled media scan %s for %s (%s)", task.task_id, url, cron_expression
        )
        return task
