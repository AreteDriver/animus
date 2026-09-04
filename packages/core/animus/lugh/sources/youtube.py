"""
YouTube source — channel subscriber that harvests auto-caption transcripts.

One ``YouTubeSource`` per configured channel. We shell out to the system
``yt-dlp`` (a runtime tool, not a Python dependency) to (a) list a channel's
most recent video ids + titles and (b) download each new video's English
auto-captions as a WebVTT file, which we strip down to readable prose.

The cleaned transcript goes in ``SourceItem.raw_text``; the raw ``.vtt`` is
cached under a gitignored directory and never redistributed (creator-content
discipline — the same posture as Aurora Arcology's "does not redistribute
creator transcripts"). Videos without captions still produce an item carrying
title + description only, with ``metadata["has_captions"] is False``.

Fail-open: a missing ``yt-dlp``, a network error, a timeout, or a caption-less
video never raises — the caller gets fewer items, not an exception.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.sources.base import SourceItem

logger = logging.getLogger(__name__)

YT_DLP_BIN = "yt-dlp"
LIST_TIMEOUT_SECONDS = 60
CAPTION_TIMEOUT_SECONDS = 120
DEFAULT_LIST_LIMIT = 25
DEFAULT_RAW_DIR = Path("~/.animus/lugh_raw/youtube").expanduser()
SUMMARY_CHARS = 500

# (handle, show_name, fetch_captions, tags). Core-tier feeds harvest captions;
# Watch-tier are list-only by default to keep the recurring scan cheap — flip
# fetch_captions per channel as wanted. Handles verified 2026-05-11; rationale
# in notes/ideas/podcast-harvest-2026-q2.md §"Tracked sources".
DEFAULT_CHANNELS: list[tuple[str, str, bool, list[str]]] = [
    ("@AIDailyBrief", "The AI Daily Brief", True, ["ai-news", "core"]),
    ("@LatentSpacePod", "Latent Space", True, ["ai-engineering", "core"]),
    ("@NoPriorsPodcast", "No Priors", True, ["ai-vc", "core"]),
    ("@a16z", "a16z Podcast", True, ["ai-vc", "core"]),
    ("@allin", "All-In Podcast", False, ["tech-business", "watch"]),
    ("@PeterDiamandis", "Moonshots", False, ["futurism", "watch"]),
    ("@DwarkeshPatel", "Dwarkesh Patel", False, ["ai-research", "watch"]),
    ("@CognitiveRevolutionPodcast", "The Cognitive Revolution", False, ["ai-builder", "watch"]),
    ("@LennysPodcast", "Lenny's Podcast", False, ["product-gtm", "watch"]),
    # EVE Online economy/news roster — feeds the "EVE market intel" synthesis
    # board (notes/ideas/eve-market-intel.md). Ashterothi's "EVE Universe Show"
    # and OZ_Eve's "Oz Report" both narrate the Monthly Economic Report, patch
    # passes, and war/event news; Loru is meta/fits context (Watch tier).
    # Handles verified 2026-05-12.
    ("@Ashterothi", "Ashterothi — EVE Universe Show", True, ["eve-economy", "eve", "core"]),
    ("@OZeve", "OZ_Eve — The Oz Report", True, ["eve-economy", "eve", "core"]),
    ("@lorumerthgaming", "Loru Gaming", False, ["eve-meta", "eve", "watch"]),
]


def clean_vtt(text: str) -> str:
    """Strip a WebVTT caption file down to deduplicated spoken prose.

    YouTube auto-captions repeat each line across several overlapping cues and
    embed inline ``<00:00:00.000>`` word timings. This drops the header,
    ``Kind:``/``Language:`` lines, cue timestamps, ``align:`` positioning, and
    inline tags, then collapses consecutive duplicate lines into one running
    paragraph.

    Args:
        text: raw ``.vtt`` file contents.

    Returns:
        Single-line cleaned transcript, or ``""`` if nothing usable remains.
    """
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "WEBVTT" or s.startswith(("Kind:", "Language:", "NOTE ", "STYLE ", "REGION ")):
            continue
        if "-->" in s or s.startswith("align:"):
            continue
        s = re.sub(r"<[^>]+>", "", s).strip()
        if not s:
            continue
        if out and out[-1] == s:
            continue
        out.append(s)
    return " ".join(out)


@dataclass
class YouTubeSource:
    """One configured YouTube channel or playlist.

    Attributes:
        channel: channel handle including ``@`` (e.g. ``"@AIDailyBrief"``) or a
            full ``/channel/UC...`` / ``/c/Name`` path fragment. Mutually
            exclusive with ``playlist_url``.
        playlist_url: full YouTube playlist URL (e.g.
            ``"https://www.youtube.com/playlist?list=PLabc"``). When set,
            ``channel`` is ignored for listing.
        show_name: human-readable show name, used for metadata + author.
        fetch_captions: if True, download + clean English auto-captions per
            video; if False, items carry title + description only.
        list_limit: how many of the channel's most recent videos to consider
            per ``fetch`` when no explicit limit is passed.
        raw_dir: directory for cached raw ``.vtt`` files (gitignored; never
            redistributed). Defaults to ``~/.animus/lugh_raw/youtube``.
        tags: default tags applied to every item from this channel.
    """

    channel: str = ""
    playlist_url: str = ""
    show_name: str = ""
    fetch_captions: bool = True
    list_limit: int = DEFAULT_LIST_LIMIT
    raw_dir: Path = field(default_factory=lambda: DEFAULT_RAW_DIR)
    tags: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        if self.playlist_url:
            # Derive a stable source_id from the playlist URL
            match = re.search(r"[?&]list=([A-Za-z0-9_-]+)", self.playlist_url)
            return (
                f"youtube:playlist:{match.group(1)}"
                if match
                else f"youtube:playlist:{self.playlist_url}"
            )
        return f"youtube:{self.channel}"

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        if not _yt_dlp_available():
            logger.warning(
                "yt-dlp not found on PATH — skipping %s (install: pipx install yt-dlp)",
                self.source_id,
            )
            return
        n = limit if limit is not None else self.list_limit
        for video_id, title, published in self._list_videos(n):
            item = self._to_item(video_id, title, published)
            if item is not None:
                yield item

    # -- internals -----------------------------------------------------------

    def _list_url(self) -> str:
        """Resolve the listing URL: playlist takes precedence over channel."""
        if self.playlist_url:
            return self.playlist_url
        return f"https://www.youtube.com/{self.channel}/videos"

    def _list_videos(self, limit: int) -> list[tuple[str, str, datetime | None]]:
        """Return ``(video_id, title, published)`` for the most recent videos.

        Resolves each video's metadata (not ``--flat-playlist``) so we can
        emit ``upload_date`` alongside id/title. yt-dlp prints
        ``%(upload_date)s`` as ``YYYYMMDD`` for normal uploads and ``NA`` for
        videos without a fixed upload timestamp (live, premiere, members-only).

        When ``playlist_url`` is set, uses the playlist URL instead of the
        channel ``/videos`` tab.
        """
        url = self._list_url()
        cmd = [
            YT_DLP_BIN,
            "--playlist-end",
            str(max(1, limit)),
            "--skip-download",
            "--print",
            "%(id)s\t%(title)s\t%(upload_date)s",
            url,
        ]
        out = _run(cmd, timeout=LIST_TIMEOUT_SECONDS)
        rows: list[tuple[str, str, datetime | None]] = []
        for line in (out or "").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            vid = parts[0].strip()
            if not vid:
                continue
            title = parts[1].strip()
            published = _parse_upload_date(parts[2]) if len(parts) > 2 else None
            rows.append((vid, title, published))
        return rows

    def _to_item(self, video_id: str, title: str, published: datetime | None) -> SourceItem | None:
        if not video_id:
            return None
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        transcript = self._fetch_captions(video_id) if self.fetch_captions else ""
        has_captions = bool(transcript)
        body_for_summary = transcript if transcript else title
        return SourceItem(
            source_id=self.source_id,
            item_id=video_id,
            title=title or video_id,
            url=watch_url,
            published=published,
            summary=_truncate(body_for_summary, SUMMARY_CHARS),
            author=self.show_name or None,
            tags=list(self.tags),
            raw_text=transcript,
            metadata={
                "video_id": video_id,
                "channel": self.channel,
                "show_name": self.show_name or self.channel,
                "has_captions": has_captions,
            },
        )

    def _fetch_captions(self, video_id: str) -> str:
        """Download + clean a video's English auto-captions. ``""`` if none."""
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("cannot create raw dir %s: %s", self.raw_dir, e)
            return ""
        out_tmpl = str(self.raw_dir / "%(id)s.%(ext)s")
        cmd = [
            YT_DLP_BIN,
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
        _run(cmd, timeout=CAPTION_TIMEOUT_SECONDS)  # side-effect: writes <id>.*.vtt
        # yt-dlp names it <id>.en.vtt (or .en-orig.vtt / .en-US.vtt depending on
        # what tracks the video exposes). Take the first that cleans to text.
        for vtt in sorted(self.raw_dir.glob(f"{video_id}.*.vtt")):
            try:
                raw = vtt.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("cannot read %s: %s", vtt, e)
                continue
            cleaned = clean_vtt(raw)
            if cleaned:
                return cleaned
        return ""


def default_youtube_sources() -> list[YouTubeSource]:
    """Seed channel list — verified handles (2026-05-11).

    Core-tier (AIDB / Latent Space / No Priors / a16z) fetch captions on each
    scan; Watch-tier are list-only by default. See
    ``notes/ideas/podcast-harvest-2026-q2.md`` §"Tracked sources" for the
    rationale and the wider (RSS / blog) feed list.
    """
    return [
        YouTubeSource(channel=h, show_name=name, fetch_captions=fc, tags=list(tags))
        for (h, name, fc, tags) in DEFAULT_CHANNELS
    ]


def probe_channel(channel: str) -> dict:
    """Quick introspection of a channel handle — for registry/CLI use.

    Returns ``{ok, video_count, sample_titles, error}``. Does not raise.
    Intended for ``lugh sources add-youtube`` to sanity-check before adding.
    """
    if not _yt_dlp_available():
        return {"ok": False, "video_count": 0, "sample_titles": [], "error": "yt-dlp not installed"}
    src = YouTubeSource(channel=channel, fetch_captions=False, list_limit=3)
    rows = src._list_videos(3)
    return {
        "ok": bool(rows),
        "video_count": len(rows),
        "sample_titles": [t for _, t, _ in rows],
        "error": "" if rows else "channel not found or has no videos tab",
    }


def probe_playlist(playlist_url: str) -> dict:
    """Quick introspection of a YouTube playlist URL — for registry/CLI use.

    Returns ``{ok, video_count, sample_titles, error}``. Does not raise.
    """
    if not playlist_url or "list=" not in playlist_url:
        return {
            "ok": False,
            "video_count": 0,
            "sample_titles": [],
            "error": "not a valid playlist URL",
        }
    if not _yt_dlp_available():
        return {"ok": False, "video_count": 0, "sample_titles": [], "error": "yt-dlp not installed"}
    src = YouTubeSource(playlist_url=playlist_url, fetch_captions=False, list_limit=3)
    rows = src._list_videos(3)
    return {
        "ok": bool(rows),
        "video_count": len(rows),
        "sample_titles": [t for _, t, _ in rows],
        "error": "" if rows else "playlist not found or empty",
    }


def playlist_to_source_items(
    playlist_url: str,
    fetch_captions: bool = True,
    list_limit: int = DEFAULT_LIST_LIMIT,
    tags: list[str] | None = None,
) -> list[SourceItem]:
    """Harvest all videos from a YouTube playlist as a list of SourceItems.

    Convenience wrapper over ``YouTubeSource`` with ``playlist_url`` set.
    """
    src = YouTubeSource(
        playlist_url=playlist_url,
        fetch_captions=fetch_captions,
        list_limit=list_limit,
        tags=list(tags) if tags else [],
    )
    return list(src.fetch(limit=list_limit))


# -- module helpers ---------------------------------------------------------


def _yt_dlp_available() -> bool:
    return shutil.which(YT_DLP_BIN) is not None


def _run(cmd: list[str], *, timeout: int) -> str | None:
    """Run a subprocess; return stdout (always, even on nonzero exit) or None.

    yt-dlp writes a lot to stderr even when the run is fine (the "no JS
    runtime" notice, impersonation warnings). We never raise: a nonzero exit
    or a transport error just yields ``None`` / partial stdout, and the caller
    (which globs the output dir for captions, or parses lines for the listing)
    degrades gracefully.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        logger.warning("command not found: %s", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("timeout after %ds: %s", timeout, " ".join(cmd[:3]))
        return None
    except OSError as e:
        logger.warning("subprocess error on %s: %s", cmd[0], e)
        return None
    if proc.returncode != 0:
        logger.debug("%s exit %d: %s", cmd[0], proc.returncode, (proc.stderr or "")[-300:])
    return proc.stdout


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _parse_upload_date(raw: str) -> datetime | None:
    """Parse yt-dlp's ``%(upload_date)s`` (``YYYYMMDD`` or ``NA``) to UTC datetime."""
    s = (raw or "").strip()
    if not s or s == "NA":
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
