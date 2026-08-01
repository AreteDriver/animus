"""SEC-05 pre-fix reproduction: generic HTTP tool reaches loopback.

This script starts a tiny HTTP server bound to 127.0.0.1 and asks the
built-in ``http_request`` tool to fetch it. On the SEC-02 baseline it
succeeds, demonstrating that the generic HTTP execution path has no
SSRF guard and no centralized egress gate.

Run from the repo root with the development environment active:

    python evidence/sec-05/sec05_preflight_repro.py

This is evidence, not a permanent test.
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from animus.tools import _tool_http_request


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ssrf-pre-fix-ok")

    def log_message(self, _fmt, *_args):
        pass


def _start_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/"


def _stop_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


def main() -> int:
    server, url = _start_server()
    try:
        result = _tool_http_request({"url": url, "method": "GET", "timeout": 5})
        if result.success and "ssrf-pre-fix-ok" in (result.output or ""):
            print("PRE-FIX REPRO: request to loopback succeeded (expected before SEC-05).")
            return 0
        print("PRE-FIX REPRO: unexpected result.", result, file=sys.stderr)
        return 1
    finally:
        _stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
