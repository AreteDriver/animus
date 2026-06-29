"""``animus.workflows.ingest`` — one-shot content ingestion pipeline.

Composes Lugh → Ogma → Memory tagging into a single callable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from animus.cognitive import ModelInterface
from animus.lugh.sources.base import SourceCache, SourceItem
from animus.memory import MemoryLayer
from animus.memory.types import MemoryType
from animus.ogma.models import OgmaOutput
from animus.ogma.read import OgmaSynthesisError
from animus.ogma.read import synthesize as ogma_synthesize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowError:
    """One recoverable or fatal failure inside the ingestion pipeline."""

    stage: Literal["lugh", "ogma", "memory"]
    error_type: str
    message: str


@dataclass
class IngestResult:
    """Outcome of a single ``ingest()`` call."""

    item: SourceItem | None = None
    synthesis: OgmaOutput | None = None
    memory_tags: list[str] | None = None
    errors: list[WorkflowError] = field(default_factory=list)
    success: bool = False


# ---------------------------------------------------------------------------
# Cache wrapper (spec surface)
# ---------------------------------------------------------------------------


class Cache:
    """Thin wrapper around ``SourceCache`` that also writes raw text to disk."""

    def __init__(
        self,
        *,
        raw_base: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._source_cache = SourceCache(path=cache_path)
        self.raw_base = raw_base or Path("~/.animus/lugh_raw").expanduser()

    def store(self, item: SourceItem) -> None:
        """Persist *item* to SQLite cache and ``~/.animus/lugh_raw/<source_type>/``."""
        self._source_cache.put(item)
        source_type = item.source_id.split(":")[0] if ":" in item.source_id else "unknown"
        raw_dir = self.raw_base / source_type
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item.item_id)[:80]
        path = raw_dir / f"{safe_id}.txt"
        path.write_text(item.raw_text or item.summary or item.title, encoding="utf-8")

    def find_by_url(self, url: str) -> SourceItem | None:
        """Return a cached ``SourceItem`` whose ``url`` matches, or ``None``."""
        for cached in self._source_cache.recent(limit=500):
            if cached.url == url:
                return cached
        return None


# ---------------------------------------------------------------------------
# Source fetcher protocol + implementations
# ---------------------------------------------------------------------------


class SourceFetcher(Protocol):
    """Anything that can fetch a single URL into a ``SourceItem``."""

    def fetch(self, url: str) -> SourceItem: ...


class _YouTubeFetcher:
    def fetch(self, url: str) -> SourceItem:
        if not _yt_dlp_available():
            raise RuntimeError("yt-dlp not installed")
        video_id = _extract_youtube_id(url)
        if not video_id:
            raise ValueError(f"cannot parse YouTube video id from {url}")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--print",
            "%(id)s\t%(title)s\t%(upload_date)s\t%(description)s",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        out = _run_yt_dlp(cmd)
        if not out:
            raise RuntimeError(f"yt-dlp failed for {url}")
        parts = out.strip().split("\t")
        if len(parts) < 2:
            raise RuntimeError(f"unexpected yt-dlp output for {url}")
        title = parts[1]
        published = _parse_upload_date(parts[2]) if len(parts) > 2 else None
        description = parts[3] if len(parts) > 3 else ""
        transcript = _fetch_youtube_captions(video_id)
        return SourceItem(
            source_id=f"youtube:{video_id}",
            item_id=video_id,
            title=title,
            url=f"https://www.youtube.com/watch?v={video_id}",
            published=published,
            summary=_truncate(description or title, 500),
            author=None,
            tags=["youtube"],
            raw_text=transcript or description,
            metadata={"video_id": video_id, "has_captions": bool(transcript)},
        )


class _ArxivFetcher:
    def fetch(self, url: str) -> SourceItem:
        arxiv_id = _extract_arxiv_id(url)
        if not arxiv_id:
            raise ValueError(f"cannot parse arxiv id from {url}")
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "animus-lugh/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"arxiv API HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"arxiv API error: {e}") from e
        title_match = re.search(r"<title>([^<]+)</title>", xml)
        summary_match = re.search(r"<summary>([^<]+)</summary>", xml, re.DOTALL)
        published_match = re.search(r"<published>([^<]+)</published>", xml)
        title = title_match.group(1).strip() if title_match else arxiv_id
        summary = summary_match.group(1).strip() if summary_match else ""
        published: datetime | None = None
        if published_match:
            try:
                published = datetime.fromisoformat(
                    published_match.group(1).replace("Z", "+00:00")
                )
            except ValueError:
                pass
        return SourceItem(
            source_id=f"arxiv:{arxiv_id}",
            item_id=arxiv_id,
            title=title,
            url=url,
            published=published,
            summary=_truncate(summary, 500),
            author=None,
            tags=["arxiv"],
            raw_text=summary,
            metadata={"arxiv_id": arxiv_id},
        )


class _WebFetcher:
    def fetch(self, url: str) -> SourceItem:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "animus-lugh/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"fetch error: {e}") from e
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else url
        return SourceItem(
            source_id="web:direct",
            item_id=hashlib.sha256(url.encode()).hexdigest()[:16],
            title=title,
            url=url,
            published=datetime.now(timezone.utc),
            summary=_truncate(text, 500),
            author=None,
            tags=["web"],
            raw_text=text,
            metadata={"url": url},
        )


def resolve_source(url: str) -> SourceFetcher:
    """Return a fetcher capable of handling *url*."""
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return _YouTubeFetcher()
    if "arxiv.org" in u:
        return _ArxivFetcher()
    if u.startswith(("http://", "https://")):
        return _WebFetcher()
    raise ValueError(f"unresolvable URL scheme: {url}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yt_dlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def _run_yt_dlp(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/embed/|/v/|/watch\?v=|/youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _fetch_youtube_captions(video_id: str) -> str:
    """Download and clean English auto-captions. Returns '' on failure."""
    from animus.lugh.sources.youtube import clean_vtt

    raw_dir = Path("~/.animus/lugh_raw/youtube").expanduser()
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(raw_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-lang",
        "en",
        "--sub-format",
        "vtt",
        "-o",
        out_tmpl,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    _run_yt_dlp(cmd)
    for vtt in sorted(raw_dir.glob(f"{video_id}.*.vtt")):
        try:
            raw = vtt.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cleaned = clean_vtt(raw)
        if cleaned:
            return cleaned
    return ""


def _extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+|[a-z-]+/\d+)", url)
    if m:
        return m.group(1)
    return None


def _parse_upload_date(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s or s == "NA":
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _tag_concepts(
    synthesis: OgmaOutput,
    memory_layer: MemoryLayer | None = None,
) -> list[str]:
    """Store synthesis concepts as semantic memories. Returns memory IDs."""
    if memory_layer is None:
        from animus.config import AnimusConfig

        config = AnimusConfig.load()
        memory_layer = MemoryLayer(config.data_dir, backend=config.memory.backend)

    tags: list[str] = []
    mem = memory_layer.remember(
        content=synthesis.to_markdown(),
        memory_type=MemoryType.SEMANTIC,
        tags=["ogma", "ingest", synthesis.source_id],
        source="ogma-ingest",
        confidence=synthesis.confidence,
        subtype="synthesis",
    )
    tags.append(mem.id)
    return tags


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def ingest(
    url: str,
    *,
    synthesize: bool = False,
    tag: bool = False,
    model: ModelInterface | None = None,
    min_relevance: float = 0.5,
) -> IngestResult:
    """Ingest *url* through Lugh → Ogma → Memory tagging.

    Args:
        url: the source URL to fetch.
        synthesize: whether to run Ogma synthesis on the fetched item.
        tag: whether to push structured concepts to semantic memory.
        model: optional ``ModelInterface`` forwarded to Ogma.
        min_relevance: relevance gate forwarded to Ogma.

    Returns:
        ``IngestResult`` populated according to which stages ran.
    """
    item: SourceItem | None = None
    errors: list[WorkflowError] = []
    cache = Cache()

    # ---- Lugh fetch -------------------------------------------------------
    try:
        fetcher = resolve_source(url)
    except ValueError as e:
        errors.append(WorkflowError("lugh", "ValueError", str(e)))
        return IngestResult(item=None, errors=errors, success=False)
    except Exception as e:
        errors.append(WorkflowError("lugh", type(e).__name__, str(e)))
        return IngestResult(item=None, errors=errors, success=False)

    # Check cache before network fetch
    cached = cache.find_by_url(url)
    if cached is not None:
        item = cached
    else:
        try:
            item = fetcher.fetch(url)
        except Exception as e:
            errors.append(WorkflowError("lugh", type(e).__name__, str(e)))
            return IngestResult(item=None, errors=errors, success=False)

    # ---- Cache ------------------------------------------------------------
    if cached is None:
        try:
            cache.store(item)
        except Exception as e:
            errors.append(
                WorkflowError("lugh", type(e).__name__, f"cache.store failed: {e}")
            )
            return IngestResult(item=None, errors=errors, success=False)

    # ---- Ogma synthesis ---------------------------------------------------
    synthesis: OgmaOutput | None = None
    if synthesize and item is not None:
        try:
            synthesis = ogma_synthesize(
                item,
                model=model,
                min_relevance=min_relevance,
            )
        except OgmaSynthesisError as e:
            errors.append(WorkflowError("ogma", "OgmaSynthesisError", str(e)))
        except Exception as e:
            errors.append(WorkflowError("ogma", type(e).__name__, str(e)))

    # ---- Memory tagging -----------------------------------------------------
    memory_tags: list[str] | None = None
    if tag and synthesis is not None:
        try:
            memory_tags = _tag_concepts(synthesis)
        except Exception as e:
            errors.append(WorkflowError("memory", type(e).__name__, str(e)))

    return IngestResult(
        item=item,
        synthesis=synthesis,
        memory_tags=memory_tags,
        errors=errors,
        success=True,
    )
