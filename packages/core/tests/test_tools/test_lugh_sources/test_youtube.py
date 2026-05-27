"""Tests for animus.lugh.sources.youtube — yt-dlp channel subscriber."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.sources import registry as reg
from animus.lugh.sources.youtube import (
    DEFAULT_CHANNELS,
    YouTubeSource,
    _parse_upload_date,
    clean_vtt,
    default_youtube_sources,
    probe_channel,
)

# A realistic-ish YouTube auto-caption snippet: each line repeats across
# overlapping cues, with inline word-timing tags.
_RAW_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.480 --> 00:00:02.550 align:start position:0%

Today<00:00:00.680><c> on</c><00:00:00.760><c> the</c><00:00:00.880><c> AI</c><00:00:01.040><c> Daily</c><00:00:01.240><c> Brief,</c>

00:00:02.550 --> 00:00:02.560 align:start position:0%
Today on the AI Daily Brief,


00:00:02.560 --> 00:00:04.550 align:start position:0%
Today on the AI Daily Brief,
the<00:00:02.960><c> memory</c><00:00:03.200><c> gap.</c>

00:00:04.550 --> 00:00:04.560 align:start position:0%
the memory gap.

"""


# -- clean_vtt --------------------------------------------------------------


class TestCleanVtt:
    def test_strips_metadata_timestamps_and_inline_tags(self):
        out = clean_vtt(_RAW_VTT)
        assert "WEBVTT" not in out
        assert "Kind:" not in out and "Language:" not in out
        assert "-->" not in out
        assert "align:" not in out
        assert "<c>" not in out and "<00:" not in out
        assert "Today on the AI Daily Brief," in out
        assert "the memory gap." in out

    def test_deduplicates_consecutive_lines(self):
        # The line "Today on the AI Daily Brief," appears 3x in the raw cues.
        out = clean_vtt(_RAW_VTT)
        assert out.count("Today on the AI Daily Brief,") == 1

    def test_empty_or_header_only_yields_empty(self):
        assert clean_vtt("") == ""
        assert clean_vtt("WEBVTT\nKind: captions\nLanguage: en\n") == ""

    def test_drops_note_style_region_blocks(self):
        vtt = "WEBVTT\n\nNOTE a note\nSTYLE ::cue { }\nREGION id=foo\n\n00:00:00.000 --> 00:00:01.000\nReal text.\n"
        out = clean_vtt(vtt)
        assert out == "Real text."


# -- YouTubeSource ----------------------------------------------------------


class TestYouTubeSource:
    def test_source_id(self):
        assert YouTubeSource(channel="@AIDailyBrief").source_id == "youtube:@AIDailyBrief"

    def test_fetch_no_yt_dlp_yields_nothing(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: None)
        src = YouTubeSource(channel="@x")
        assert list(src.fetch()) == []

    def test_fetch_lists_videos_without_captions(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: (
                "abc123\tFirst Video\t20260527\nxyz789\tSecond Video\t20260526\n"
            ),
        )
        src = YouTubeSource(channel="@x", show_name="Show X", fetch_captions=False, tags=["t1"])
        items = list(src.fetch())
        assert len(items) == 2
        first = items[0]
        assert first.source_id == "youtube:@x"
        assert first.item_id == "abc123"
        assert first.title == "First Video"
        assert first.url == "https://www.youtube.com/watch?v=abc123"
        assert first.author == "Show X"
        assert first.tags == ["t1"]
        assert first.raw_text == ""
        assert first.published == datetime(2026, 5, 27, tzinfo=timezone.utc)
        assert items[1].published == datetime(2026, 5, 26, tzinfo=timezone.utc)
        assert first.metadata == {
            "video_id": "abc123",
            "channel": "@x",
            "show_name": "Show X",
            "has_captions": False,
        }
        # summary falls back to the title when there's no transcript
        assert first.summary == "First Video"
        # fingerprint is stable + distinct per item
        assert first.fingerprint != items[1].fingerprint

    def test_fetch_with_captions_populates_raw_text(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: "vid1\tThe Episode\t20260520\n",
        )
        monkeypatch.setattr(
            YouTubeSource, "_fetch_captions", lambda self, vid: "cleaned transcript text here"
        )
        src = YouTubeSource(channel="@x", fetch_captions=True)
        (item,) = list(src.fetch())
        assert item.raw_text == "cleaned transcript text here"
        assert item.summary == "cleaned transcript text here"
        assert item.metadata["has_captions"] is True
        assert item.published == datetime(2026, 5, 20, tzinfo=timezone.utc)

    def test_fetch_respects_limit(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        seen_cmds: list[list[str]] = []

        def fake_run(cmd, *, timeout):
            seen_cmds.append(cmd)
            return "a\tA\t20260101\nb\tB\t20260102\nc\tC\t20260103\n"

        monkeypatch.setattr("animus.lugh.sources.youtube._run", fake_run)
        src = YouTubeSource(channel="@x", fetch_captions=False)
        list(src.fetch(limit=2))
        # the limit is passed through to yt-dlp's --playlist-end
        assert "--playlist-end" in seen_cmds[0]
        assert seen_cmds[0][seen_cmds[0].index("--playlist-end") + 1] == "2"
        # --flat-playlist is intentionally NOT used (drops upload_date) — animus#68
        assert "--flat-playlist" not in seen_cmds[0]
        # the print template must include upload_date so SourceItem.published populates
        idx = seen_cmds[0].index("--print")
        assert "%(upload_date)s" in seen_cmds[0][idx + 1]

    def test_fetch_published_none_when_yt_dlp_emits_na(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: "live1\tLive Premiere\tNA\n",
        )
        src = YouTubeSource(channel="@x", fetch_captions=False)
        (item,) = list(src.fetch())
        assert item.published is None

    def test_fetch_published_none_when_upload_date_column_missing(self, monkeypatch):
        # Defensive: a registry/test that emits the legacy 2-col format must still parse.
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: "old1\tOld Format\n",
        )
        src = YouTubeSource(channel="@x", fetch_captions=False)
        (item,) = list(src.fetch())
        assert item.item_id == "old1"
        assert item.published is None

    def test_run_returning_none_yields_no_items(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr("animus.lugh.sources.youtube._run", lambda cmd, *, timeout: None)
        src = YouTubeSource(channel="@x", fetch_captions=False)
        assert list(src.fetch()) == []

    def test_blank_video_ids_are_skipped(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: (
                "\tno id line\t20260101\ngood1\tGood One\t20260102\n  \tblank\t20260103\n"
            ),
        )
        src = YouTubeSource(channel="@x", fetch_captions=False)
        items = list(src.fetch())
        assert [i.item_id for i in items] == ["good1"]


class TestParseUploadDate:
    def test_yyyymmdd_parses_to_utc_datetime(self):
        assert _parse_upload_date("20260527") == datetime(2026, 5, 27, tzinfo=timezone.utc)

    def test_na_yields_none(self):
        assert _parse_upload_date("NA") is None

    def test_empty_yields_none(self):
        assert _parse_upload_date("") is None
        assert _parse_upload_date("   ") is None

    def test_garbage_yields_none(self):
        assert _parse_upload_date("2026-05-27") is None
        assert _parse_upload_date("yesterday") is None
        assert _parse_upload_date("20261332") is None


class TestFetchCaptions:
    def test_reads_and_cleans_vtt_written_by_yt_dlp(self, tmp_path: Path, monkeypatch):
        # Simulate yt-dlp having written the .vtt file; _run is a no-op.
        (tmp_path / "vid1.en.vtt").write_text(_RAW_VTT, encoding="utf-8")
        monkeypatch.setattr("animus.lugh.sources.youtube._run", lambda cmd, *, timeout: None)
        src = YouTubeSource(channel="@x", raw_dir=tmp_path)
        out = src._fetch_captions("vid1")
        assert "Today on the AI Daily Brief," in out
        assert "<c>" not in out

    def test_returns_empty_when_no_vtt_appears(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube._run", lambda cmd, *, timeout: None)
        src = YouTubeSource(channel="@x", raw_dir=tmp_path)
        assert src._fetch_captions("missing") == ""

    def test_prefers_a_vtt_that_cleans_to_text(self, tmp_path: Path, monkeypatch):
        # An empty-ish track and a real one — the real one wins.
        (tmp_path / "vid1.en-orig.vtt").write_text("WEBVTT\nKind: captions\n", encoding="utf-8")
        (tmp_path / "vid1.en.vtt").write_text(_RAW_VTT, encoding="utf-8")
        monkeypatch.setattr("animus.lugh.sources.youtube._run", lambda cmd, *, timeout: None)
        src = YouTubeSource(channel="@x", raw_dir=tmp_path)
        out = src._fetch_captions("vid1")
        assert "memory gap." in out


# -- defaults + probe -------------------------------------------------------


class TestDefaults:
    def test_default_youtube_sources_nonempty_and_includes_aidb(self):
        sources = default_youtube_sources()
        assert len(sources) == len(DEFAULT_CHANNELS)
        handles = {s.channel for s in sources}
        assert "@AIDailyBrief" in handles
        assert "@LatentSpacePod" in handles
        # core-tier fetches captions; watch-tier doesn't
        aidb = next(s for s in sources if s.channel == "@AIDailyBrief")
        allin = next(s for s in sources if s.channel == "@allin")
        assert aidb.fetch_captions is True
        assert allin.fetch_captions is False
        assert "core" in aidb.tags

    def test_probe_channel_without_yt_dlp(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: None)
        result = probe_channel("@x")
        assert result["ok"] is False
        assert "yt-dlp" in result["error"]

    def test_probe_channel_with_videos(self, monkeypatch):
        monkeypatch.setattr("animus.lugh.sources.youtube.shutil.which", lambda _: "/usr/bin/yt-dlp")
        monkeypatch.setattr(
            "animus.lugh.sources.youtube._run",
            lambda cmd, *, timeout: "a\tAlpha\t20260101\nb\tBeta\tNA\nc\tGamma\t20260103\n",
        )
        result = probe_channel("@x")
        assert result["ok"] is True
        assert result["video_count"] == 3
        assert result["sample_titles"] == ["Alpha", "Beta", "Gamma"]


# -- registry integration ---------------------------------------------------


class TestRegistryIntegration:
    def test_default_registry_includes_youtube_entries(self):
        entries = reg.default_registry()
        yt = [e for e in entries if e.get("kind") == "youtube"]
        assert len(yt) == len(DEFAULT_CHANNELS)
        assert any(e["channel"] == "@AIDailyBrief" for e in yt)
        sample = next(e for e in yt if e["channel"] == "@AIDailyBrief")
        assert sample["fetch_captions"] is True
        assert isinstance(sample["tags"], list)

    def test_add_and_instantiate_youtube(self, tmp_path: Path):
        path = tmp_path / "lugh_sources.json"
        # add_youtube on a fresh registry (seeds defaults, then appends)
        assert reg.add_youtube("@MyChannel", show_name="Mine", tags=["x", "y"], path=path) is True
        # duplicate add is a no-op
        assert reg.add_youtube("@MyChannel", path=path) is False
        entries = reg.load_registry(path=path)
        mine = next(e for e in entries if e.get("channel") == "@MyChannel")
        assert mine["show_name"] == "Mine"
        assert mine["tags"] == ["x", "y"]
        sources = reg.instantiate(entries)
        my_src = next(s for s in sources if getattr(s, "channel", None) == "@MyChannel")
        assert isinstance(my_src, YouTubeSource)
        assert my_src.show_name == "Mine"
        assert my_src.tags == ["x", "y"]

    def test_remove_youtube_source(self, tmp_path: Path):
        path = tmp_path / "lugh_sources.json"
        reg.add_youtube("@Gone", path=path)
        assert reg.remove_source("youtube:@Gone", path=path) is True
        assert reg.remove_source("youtube:@Gone", path=path) is False
        entries = reg.load_registry(path=path)
        assert not any(e.get("channel") == "@Gone" for e in entries)

    def test_instantiate_tolerates_comma_string_tags(self, tmp_path: Path):
        # Older/hand-edited entries may carry tags as a comma string.
        sources = reg.instantiate([{"kind": "youtube", "channel": "@c", "tags": "a, b ,c"}])
        assert len(sources) == 1
        assert sources[0].tags == ["a", "b", "c"]

    def test_instantiate_skips_youtube_entry_missing_channel(self, tmp_path: Path):
        sources = reg.instantiate([{"kind": "youtube", "show_name": "no channel"}])
        assert sources == []
