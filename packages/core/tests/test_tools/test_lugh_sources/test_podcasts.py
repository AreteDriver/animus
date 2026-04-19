"""Tests for animus.lugh.sources.podcasts."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from animus.lugh.sources.podcasts import (
    PodcastSource,
    _flatten_json_transcript,
    _strip_cue_lines,
    probe_feed,
)
from animus.lugh.sources.rss import FeedEntry


def _entry(**kw) -> FeedEntry:
    defaults = dict(
        id="episode-42",
        title="Episode 42 — Learned Memory",
        link="https://example.com/e42",
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        author="Host",
        summary="<p>Show notes for <b>42</b>.</p>",
        content="<p>Full show notes with markup.</p>",
        tags=["tech"],
        enclosure_url="https://example.com/e42.mp3",
        enclosure_type="audio/mpeg",
        extra={
            "itunes_duration": "45:00",
            "itunes_episode": "42",
        },
    )
    defaults.update(kw)
    return FeedEntry(**defaults)


class TestToItem:
    def test_basic_mapping(self):
        src = PodcastSource(show_id="all-in", feed_url="https://x", show_name="All-In")
        item = src._to_item(_entry())
        assert item is not None
        assert item.source_id == "podcast:all-in"
        assert item.item_id == "episode-42"
        assert item.title == "Episode 42 — Learned Memory"
        assert item.metadata["show_id"] == "all-in"
        assert item.metadata["show_name"] == "All-In"
        assert item.metadata["audio_url"] == "https://example.com/e42.mp3"
        assert item.metadata["duration"] == "45:00"
        assert item.metadata["episode_number"] == "42"
        assert item.metadata["has_transcript"] is False

    def test_html_stripped_from_summary(self):
        # When entry.content is present it wins over summary (richer body).
        src = PodcastSource(show_id="x", feed_url="https://x")
        item = src._to_item(_entry())
        assert "<p>" not in item.summary
        assert "Full show notes" in item.summary  # from entry.content

    def test_falls_back_to_summary_when_content_empty(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        item = src._to_item(_entry(content=""))
        assert "<p>" not in item.summary
        assert "Show notes for" in item.summary  # from entry.summary

    def test_missing_title_returns_none(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        assert src._to_item(_entry(title="")) is None

    def test_missing_id_returns_none(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        assert src._to_item(_entry(id="", link="")) is None

    def test_falls_back_to_link_as_id(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        item = src._to_item(_entry(id=""))
        assert item is not None
        assert item.item_id == "https://example.com/e42"

    def test_transcripts_off_keeps_show_notes(self):
        src = PodcastSource(show_id="x", feed_url="https://x", fetch_transcripts=False)
        entry = _entry(
            extra={
                "itunes_transcript_url": "https://example.com/e42.srt",
                "itunes_transcript_type": "application/srt",
            }
        )
        item = src._to_item(entry)
        assert item.metadata["has_transcript"] is False
        assert "Full show notes" in item.raw_text  # show notes, not transcript

    def test_transcripts_on_populates_raw_text(self):
        src = PodcastSource(show_id="x", feed_url="https://x", fetch_transcripts=True)
        entry = _entry(
            extra={
                "itunes_transcript_url": "https://example.com/e42.srt",
                "itunes_transcript_type": "application/srt",
            }
        )
        srt_body = b"1\n00:00:00,000 --> 00:00:02,000\nHello world.\n"

        resp = MagicMock()
        resp.read.return_value = srt_body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            item = src._to_item(entry)
        assert item.metadata["has_transcript"] is True
        assert "Hello world" in item.raw_text


class TestFetch:
    def test_fetch_maps_entries(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        with patch("animus.lugh.sources.podcasts.fetch_feed", return_value=[_entry()]):
            items = list(src.fetch())
        assert len(items) == 1
        assert items[0].source_id == "podcast:x"

    def test_fetch_respects_limit(self):
        src = PodcastSource(show_id="x", feed_url="https://x")
        entries = [_entry(id=f"e{i}", link=f"https://x/{i}") for i in range(5)]
        with patch("animus.lugh.sources.podcasts.fetch_feed", return_value=entries):
            items = list(src.fetch(limit=2))
        assert len(items) == 2


class TestProbeFeed:
    def test_probe_returns_sample_titles(self):
        with patch(
            "animus.lugh.sources.podcasts.fetch_feed",
            return_value=[_entry(), _entry(title="Ep 43", id="e43")],
        ):
            result = probe_feed("https://x")
        assert result["ok"] is True
        assert result["entry_count"] == 2
        assert len(result["sample_titles"]) == 2

    def test_probe_empty_feed(self):
        with patch("animus.lugh.sources.podcasts.fetch_feed", return_value=[]):
            result = probe_feed("https://x")
        assert result["ok"] is False
        assert result["entry_count"] == 0
        assert "empty" in result["error"]


class TestCueLineStripping:
    def test_srt_stripped(self):
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Hello world.\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:04,000\n"
            "How are you?\n"
        )
        cleaned = _strip_cue_lines(srt)
        assert "00:00:00" not in cleaned
        assert "Hello world." in cleaned
        assert "How are you?" in cleaned

    def test_vtt_stripped(self):
        vtt = "WEBVTT\n\nNOTE This is a note.\n\n00:00:00.000 --> 00:00:02.000\nWelcome.\n"
        cleaned = _strip_cue_lines(vtt)
        assert "WEBVTT" not in cleaned
        assert "NOTE" not in cleaned
        assert "Welcome." in cleaned


class TestJsonTranscript:
    def test_flatten_segments(self):
        payload = '{"segments": [{"body": "Hi."}, {"body": "There."}]}'
        assert _flatten_json_transcript(payload) == "Hi. There."

    def test_flatten_top_level_list(self):
        payload = '[{"text": "One."}, {"text": "Two."}]'
        assert _flatten_json_transcript(payload) == "One. Two."

    def test_flatten_bad_json_falls_through(self):
        assert _flatten_json_transcript("not json") == "not json"
