"""
Drift//Monitor ingestion client for Lugh transcript signals.

Decoupled from the drift-monitor package — no imports, no shared objects.
Two sinks, chosen by env:

  LUGH_DRIFT_MONITOR_URL   — POST drift_signals payloads to this endpoint.
  LUGH_DRIFT_MONITOR_TOKEN — Optional Bearer token.
  LUGH_DRIFT_SINK_FILE     — NDJSON fallback path (default ~/.animus/lugh_drift.ndjson).

If the HTTP URL is set, each payload is POSTed. On any HTTP failure (network,
non-2xx) the client falls back to appending to the NDJSON file so no signal
is lost. If no URL is set, it writes straight to NDJSON.

`publish()` is fail-open: it never raises to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from animus.lugh.models import SessionSummary
from animus.lugh.transcripts import drift_signals

logger = logging.getLogger(__name__)

URL_ENV = "LUGH_DRIFT_MONITOR_URL"
TOKEN_ENV = "LUGH_DRIFT_MONITOR_TOKEN"
SINK_FILE_ENV = "LUGH_DRIFT_SINK_FILE"

DEFAULT_SINK_FILE = Path("~/.animus/lugh_drift.ndjson").expanduser()
HTTP_TIMEOUT_SECONDS = 10


class DriftMonitorClient:
    """Fail-open publisher for Lugh drift payloads."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        sink_file: Path | None = None,
    ) -> None:
        self.url = url if url is not None else os.environ.get(URL_ENV, "") or None
        self.token = token if token is not None else os.environ.get(TOKEN_ENV, "") or None
        self.sink_file = (
            sink_file or Path(os.environ.get(SINK_FILE_ENV, str(DEFAULT_SINK_FILE))).expanduser()
        )
        self.sink_file.parent.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def publish(self, payload: dict) -> bool:
        """Publish one payload. Returns True on successful HTTP POST."""
        if self.url:
            ok = self._post(payload)
            if ok:
                return True
            # Fall through to file sink on HTTP failure
            logger.warning("drift POST failed; falling back to sink file %s", self.sink_file)
        self._append(payload)
        return False

    def publish_session(self, summary: SessionSummary) -> bool:
        return self.publish(drift_signals(summary))

    def publish_many(self, sessions: Iterable[SessionSummary]) -> dict[str, int]:
        posted = appended = 0
        for s in sessions:
            if self.publish_session(s):
                posted += 1
            else:
                appended += 1
        return {"posted": posted, "appended": appended}

    # -- internals ----------------------------------------------------------

    def _post(self, payload: dict) -> bool:
        assert self.url is not None
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            logger.warning("drift POST HTTP %s: %s", e.code, e.reason)
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("drift POST transport error: %s", e)
            return False

    def _append(self, payload: Any) -> None:
        try:
            with self.sink_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except OSError as e:
            # Last-resort failure — log only, never raise.
            logger.error("drift sink write failed %s: %s", self.sink_file, e)
