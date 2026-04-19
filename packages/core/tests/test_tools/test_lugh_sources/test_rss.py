"""Tests for animus.lugh.sources.rss — stdlib RSS/Atom parser."""

from __future__ import annotations

from animus.lugh.sources.rss import fetch_feed, parse_feed

RSS_SAMPLE = b"""<?xml version="1.0"?>
<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     version="2.0">
  <channel>
    <title>Example Show</title>
    <item>
      <title>Episode 42</title>
      <link>https://example.com/e42</link>
      <guid isPermaLink="false">episode-42</guid>
      <pubDate>Fri, 18 Apr 2026 10:00:00 +0000</pubDate>
      <dc:creator>Jane Doe</dc:creator>
      <description>Show notes for 42.</description>
      <content:encoded>Full show notes with <b>markup</b>.</content:encoded>
      <category>tech</category>
      <category>ai</category>
      <enclosure url="https://example.com/e42.mp3" type="audio/mpeg" length="1234"/>
      <itunes:duration>45:00</itunes:duration>
      <itunes:episode>42</itunes:episode>
      <itunes:transcript url="https://example.com/e42.srt" type="application/srt"/>
    </item>
  </channel>
</rss>
"""

ATOM_SAMPLE = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv cs.LG</title>
  <entry>
    <id>http://arxiv.org/abs/2604.14176</id>
    <title>The Devil Is in Gradient Entanglement</title>
    <link rel="alternate" href="http://arxiv.org/abs/2604.14176"/>
    <published>2026-04-18T10:00:00Z</published>
    <updated>2026-04-18T11:00:00Z</updated>
    <author><name>A. Researcher</name></author>
    <summary>Short summary of the paper.</summary>
    <content type="html">Longer content body.</content>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
</feed>
"""


class TestRSSParse:
    def test_parse_rss_basic(self):
        [entry] = parse_feed(RSS_SAMPLE)
        assert entry.id == "episode-42"
        assert entry.title == "Episode 42"
        assert entry.link == "https://example.com/e42"
        assert entry.author == "Jane Doe"
        assert entry.summary.startswith("Show notes")
        # _text() concatenates text + child tails, so "<b>markup</b>" yields "markup"
        assert "markup" in entry.content

    def test_parse_rss_pubdate(self):
        [entry] = parse_feed(RSS_SAMPLE)
        assert entry.published is not None
        assert entry.published.year == 2026
        assert entry.published.month == 4
        assert entry.published.day == 18

    def test_parse_rss_tags(self):
        [entry] = parse_feed(RSS_SAMPLE)
        assert entry.tags == ["tech", "ai"]

    def test_parse_rss_enclosure(self):
        [entry] = parse_feed(RSS_SAMPLE)
        assert entry.enclosure_url == "https://example.com/e42.mp3"
        assert entry.enclosure_type == "audio/mpeg"

    def test_parse_rss_itunes_extras(self):
        [entry] = parse_feed(RSS_SAMPLE)
        assert entry.extra["itunes_duration"] == "45:00"
        assert entry.extra["itunes_episode"] == "42"
        assert entry.extra["itunes_transcript_url"] == "https://example.com/e42.srt"
        assert entry.extra["itunes_transcript_type"] == "application/srt"


class TestAtomParse:
    def test_parse_atom_basic(self):
        [entry] = parse_feed(ATOM_SAMPLE)
        assert entry.id == "http://arxiv.org/abs/2604.14176"
        assert entry.title == "The Devil Is in Gradient Entanglement"
        assert entry.link == "http://arxiv.org/abs/2604.14176"
        assert entry.author == "A. Researcher"

    def test_parse_atom_published(self):
        [entry] = parse_feed(ATOM_SAMPLE)
        assert entry.published is not None
        assert entry.published.year == 2026

    def test_parse_atom_tags(self):
        [entry] = parse_feed(ATOM_SAMPLE)
        assert entry.tags == ["cs.LG", "cs.AI"]


class TestFailOpen:
    def test_unrecognized_root_returns_empty(self):
        assert parse_feed(b"<html><body>not a feed</body></html>") == []

    def test_malformed_xml_returns_empty(self):
        assert parse_feed(b"<rss><channel><item><title>oops") == []

    def test_fetch_feed_http_failure_returns_empty(self, monkeypatch):
        import urllib.error

        def fail(*a, **kw):
            raise urllib.error.URLError("no network")

        monkeypatch.setattr("urllib.request.urlopen", fail)
        assert fetch_feed("http://example.com/feed") == []
