"""Token-based authentication."""

import secrets
from datetime import datetime, timedelta

from animus_forge.config import get_settings


class TokenAuth:
    """Simple token-based authentication."""

    def __init__(self):
        self.settings = get_settings()
        self._tokens = {}

    def create_token(self, user_id: str) -> str:
        """Create a new cryptographically secure access token."""
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(minutes=self.settings.access_token_expire_minutes)
        self._tokens[token] = {"user_id": user_id, "expiry": expiry}
        return token

    def verify_token(self, token: str) -> str | None:
        """Verify token and return user_id if valid."""
        if token not in self._tokens:
            return None

        token_data = self._tokens[token]
        if datetime.now() > token_data["expiry"]:
            del self._tokens[token]
            return None

        return token_data["user_id"]

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False


_auth: TokenAuth | None = None


def _get_auth() -> TokenAuth:
    """Lazy-initialize the global TokenAuth instance."""
    global _auth
    if _auth is None:
        _auth = TokenAuth()
    return _auth


def create_access_token(user_id: str) -> str:
    """Create an access token for a user."""
    return _get_auth().create_token(user_id)


def verify_token(token: str) -> str | None:
    """Verify a token and return user_id if valid."""
    return _get_auth().verify_token(token)
