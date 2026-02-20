"""
OAuth2 Flow Helper

Handles OAuth2 authentication flow for Google services.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from animus.logging import get_logger

logger = get_logger("oauth")

# Check if Google auth libraries are available
GOOGLE_AUTH_AVAILABLE = False
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    Request = None  # type: ignore[assignment,misc]
    Credentials = None  # type: ignore[assignment,misc]
    Flow = None  # type: ignore[assignment,misc]


@dataclass
class OAuth2Token:
    """OAuth2 token data."""

    access_token: str
    refresh_token: str | None
    token_type: str
    expires_at: datetime | None
    scopes: list[str]

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuth2Token:
        """Create from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        return cls(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=data.get("scopes", []),
        )


class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive OAuth2 callback."""

    authorization_code: str | None = None
    error: str | None = None

    def do_GET(self):
        """Handle GET request with OAuth2 callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuth2CallbackHandler.authorization_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Authorization successful!</h1>"
                b"<p>You can close this window and return to Animus.</p></body></html>"
            )
        elif "error" in params:
            OAuth2CallbackHandler.error = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h1>Authorization failed</h1>"
                f"<p>Error: {OAuth2CallbackHandler.error}</p></body></html>".encode()
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class OAuth2Flow:
    """
    Handles OAuth2 authentication flow for Google services.

    Usage:
        flow = OAuth2Flow(
            client_id="...",
            client_secret="...",
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        token = flow.run_local_server()
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scopes: list[str],
        redirect_port: int = 8422,
    ):
        """
        Initialize OAuth2 flow.

        Args:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            scopes: List of OAuth2 scopes to request
            redirect_port: Local port for OAuth2 callback
        """
        if not GOOGLE_AUTH_AVAILABLE:
            raise ImportError(
                "Google auth libraries not installed. "
                "Install with: pip install google-auth google-auth-oauthlib"
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes
        self.redirect_port = redirect_port
        self.redirect_uri = f"http://localhost:{redirect_port}"

    def run_local_server(self, open_browser: bool = True) -> OAuth2Token | None:
        """
        Run OAuth2 flow with local server to receive callback.

        Args:
            open_browser: Whether to automatically open browser

        Returns:
            OAuth2Token if successful, None if failed
        """
        # Create flow using client config
        client_config = {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri],
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )

        # Get authorization URL
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        # Reset callback handler state
        OAuth2CallbackHandler.authorization_code = None
        OAuth2CallbackHandler.error = None

        # Start local server
        server = HTTPServer(("localhost", self.redirect_port), OAuth2CallbackHandler)
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        logger.info("Opening browser for authorization...")
        if open_browser:
            webbrowser.open(auth_url)
        else:
            logger.info(f"Please visit: {auth_url}")

        # Wait for callback
        server_thread.join(timeout=300)  # 5 minute timeout
        server.server_close()

        if OAuth2CallbackHandler.error:
            logger.error(f"OAuth2 error: {OAuth2CallbackHandler.error}")
            return None

        if not OAuth2CallbackHandler.authorization_code:
            logger.error("No authorization code received")
            return None

        # Exchange code for token
        try:
            flow.fetch_token(code=OAuth2CallbackHandler.authorization_code)
            credentials = flow.credentials

            expires_at = None
            if credentials.expiry:
                expires_at = credentials.expiry

            return OAuth2Token(
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_type="Bearer",
                expires_at=expires_at,
                scopes=list(credentials.scopes) if credentials.scopes else self.scopes,
            )
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            return None

    def refresh_token(self, token: OAuth2Token) -> OAuth2Token | None:
        """
        Refresh an expired token.

        Args:
            token: Existing OAuth2Token with refresh_token

        Returns:
            New OAuth2Token or None if refresh failed
        """
        if not token.refresh_token:
            logger.error("No refresh token available")
            return None

        try:
            credentials = Credentials(
                token=token.access_token,
                refresh_token=token.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=token.scopes,
            )

            credentials.refresh(Request())

            expires_at = None
            if credentials.expiry:
                expires_at = credentials.expiry

            return OAuth2Token(
                access_token=credentials.token,
                refresh_token=credentials.refresh_token or token.refresh_token,
                token_type="Bearer",
                expires_at=expires_at,
                scopes=token.scopes,
            )
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None


def save_token(token: OAuth2Token, path: Path) -> None:
    """Save OAuth2 token to file."""
    with open(path, "w") as f:
        json.dump(token.to_dict(), f)
    path.chmod(0o600)


def load_token(path: Path) -> OAuth2Token | None:
    """Load OAuth2 token from file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return OAuth2Token.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.error(f"Failed to load token: {e}")
        return None
