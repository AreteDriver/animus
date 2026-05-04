"""Coverage for scattered small-file gaps on 2c8cde2.

Narrow set — only edge helpers that can be tested without heavy fixtures.
Leaves larger webhook_manager/broadcaster branches for a follow-up.
"""

from __future__ import annotations

import sys
from datetime import datetime

sys.path.insert(0, "src")

from animus_forge.webhooks.webhook_manager import _parse_datetime


class TestParseDatetime:
    def test_parses_iso_string(self):
        """Line 26 (webhook_manager._parse_datetime): ISO string path."""
        result = _parse_datetime("2026-04-20T12:34:56")
        assert isinstance(result, datetime)
        assert result.year == 2026

    def test_passes_through_datetime(self):
        now = datetime.now()
        assert _parse_datetime(now) is now

    def test_none_is_none(self):
        assert _parse_datetime(None) is None
