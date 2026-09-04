"""Durable, resumable Markdown artifacts for harvested media items."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from animus.lugh.sources.base import SourceItem
from animus.ogma.models import OgmaOutput


@dataclass(frozen=True)
class MediaItemArtifact:
    """Persistence result for one harvested media item."""

    ordinal: int
    item_id: str
    title: str
    url: str
    path: Path
    analysis_status: str
    has_captions: bool
    transcript_chars: int
    error: str = ""
    skipped: bool = False


@dataclass
class MediaArtifactCollectionReport:
    """Report for a full individual-artifact collection run."""

    source_url: str
    artifact_dir: Path
    items_discovered: int = 0
    artifacts: list[MediaItemArtifact] = field(default_factory=list)
    index_path: Path | None = None
    duration_seconds: float = 0.0

    @property
    def artifacts_written(self) -> int:
        return sum(not artifact.skipped for artifact in self.artifacts)

    @property
    def artifacts_skipped(self) -> int:
        return sum(artifact.skipped for artifact in self.artifacts)

    @property
    def syntheses_succeeded(self) -> int:
        return sum(artifact.analysis_status == "synthesized" for artifact in self.artifacts)

    @property
    def curated(self) -> int:
        return sum(artifact.analysis_status == "curated" for artifact in self.artifacts)

    @property
    def captions_missing(self) -> int:
        return sum(not artifact.has_captions for artifact in self.artifacts)

    @property
    def failures(self) -> int:
        return sum(artifact.analysis_status == "synthesis-failed" for artifact in self.artifacts)

    def summary(self) -> str:
        return (
            f"Media artifacts: {self.items_discovered} discovered; "
            f"{self.artifacts_written} written; {self.artifacts_skipped} resumed; "
            f"{self.syntheses_succeeded} synthesized; {self.curated} curated; "
            f"{self.captions_missing} without captions; {self.failures} failed"
        )


class MediaArtifactWriter:
    """Write one provenance-preserving Markdown file per media item."""

    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.items_dir = self.artifact_dir / "items"
        self.items_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, item: SourceItem) -> Path:
        """Return a stable artifact path keyed by the source item id."""
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", item.item_id).strip("_")
        return self.items_dir / f"{safe_id or item.fingerprint[:16]}.md"

    def is_complete(self, item: SourceItem) -> bool:
        """Return whether an existing artifact contains a completed synthesis."""
        return self._existing_status(item) in {"synthesized", "curated"}

    def _existing_status(self, item: SourceItem) -> str:
        """Read the analysis status from an existing artifact front matter."""
        path = self.path_for(item)
        if not path.exists():
            return ""
        try:
            header = path.read_text(encoding="utf-8")[:1200]
        except OSError:
            return ""
        match = re.search(r"^analysis_status: ([A-Za-z-]+)$", header, re.MULTILINE)
        return match.group(1) if match else ""

    def describe_existing(self, item: SourceItem, ordinal: int) -> MediaItemArtifact:
        """Describe a completed artifact without rewriting it."""
        return MediaItemArtifact(
            ordinal=ordinal,
            item_id=item.item_id,
            title=item.title,
            url=item.url,
            path=self.path_for(item),
            analysis_status=self._existing_status(item) or "metadata-only",
            has_captions=bool(item.raw_text),
            transcript_chars=len(item.raw_text or ""),
            skipped=True,
        )

    def write_item(
        self,
        item: SourceItem,
        ordinal: int,
        synthesis: OgmaOutput | None,
        error: str = "",
    ) -> MediaItemArtifact:
        """Write one item artifact atomically enough for resumable batch runs."""
        path = self.path_for(item)
        has_captions = bool(item.raw_text)
        if synthesis is not None:
            status = "synthesized"
        elif error:
            status = "synthesis-failed"
        else:
            status = "metadata-only"

        published = item.published.isoformat() if item.published else ""
        lines = [
            "---",
            f"item_id: {json.dumps(item.item_id)}",
            f"source_id: {json.dumps(item.source_id)}",
            f"title: {json.dumps(item.title)}",
            f"url: {json.dumps(item.url)}",
            f"published: {json.dumps(published)}",
            f"analysis_status: {status}",
            f"has_captions: {str(has_captions).lower()}",
            f"transcript_chars: {len(item.raw_text or '')}",
            f"harvested_at: {json.dumps(datetime.now().astimezone().isoformat())}",
            "---",
            "",
            f"# {item.title}",
            "",
            "## Source",
            "",
            f"- Video: [{item.url}]({item.url})" if item.url else "- Video URL unavailable",
            f"- Source ID: `{item.source_id}`",
            f"- Item ID: `{item.item_id}`",
            f"- Published: {published or 'unknown'}",
            f"- Captions: {'available' if has_captions else 'unavailable'}",
            f"- Transcript characters: {len(item.raw_text or '')}",
            "",
            "## Evidence preview",
            "",
            item.summary.strip() if item.summary else "No transcript or description was available.",
            "",
        ]

        if synthesis is not None:
            lines.extend(["## Harvest analysis", "", synthesis.to_markdown().strip(), ""])
        else:
            lines.extend(
                [
                    "## Harvest analysis",
                    "",
                    "Analysis remains pending. This artifact preserves the source record so a "
                    "later resumable run can synthesize it without losing playlist coverage.",
                    "",
                ]
            )
            if error:
                lines.extend(["### Processing note", "", error, ""])

        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return MediaItemArtifact(
            ordinal=ordinal,
            item_id=item.item_id,
            title=item.title,
            url=item.url,
            path=path,
            analysis_status=status,
            has_captions=has_captions,
            transcript_chars=len(item.raw_text or ""),
            error=error,
        )

    def write_index(self, report: MediaArtifactCollectionReport) -> Path:
        """Write the collection manifest after each completed batch."""
        path = self.artifact_dir / "INDEX.md"
        lines = [
            "# Media Harvest Index",
            "",
            f"Source: [{report.source_url}]({report.source_url})",
            "",
            report.summary(),
            "",
            "| # | Video | Captions | Analysis | Artifact |",
            "|---:|---|:---:|---|---|",
        ]
        for artifact in report.artifacts:
            title = artifact.title.replace("|", "\\|")
            relative = artifact.path.relative_to(self.artifact_dir)
            captions = "yes" if artifact.has_captions else "no"
            lines.append(
                f"| {artifact.ordinal} | [{title}]({artifact.url}) | {captions} | "
                f"{artifact.analysis_status} | [{artifact.item_id}]({relative.as_posix()}) |"
            )
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path
