"""Tests for token authentication module."""

import sys
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.auth.token_auth import (
    TokenAuth,
    create_access_token,
    verify_token,
)


class TestTokenAuth:
    """Tests for TokenAuth class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.access_token_expire_minutes = 30
        return settings

    @pytest.fixture
    def auth(self, mock_settings):
        """Create TokenAuth with mocked settings."""
        with patch("animus_forge.auth.token_auth.get_settings", return_value=mock_settings):
            return TokenAuth()

    def test_create_token_returns_string(self, auth):
        """create_token returns a string token."""
        token = auth.create_token("user123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_is_opaque(self, auth):
        """Token is opaque and does not leak user ID."""
        token = auth.create_token("user123")
        assert "user123" not in token
        assert len(token) >= 32

    def test_create_token_unique(self, auth):
        """Each token is unique."""
        token1 = auth.create_token("user123")
        time.sleep(0.01)  # Ensure different timestamp
        token2 = auth.create_token("user123")
        assert token1 != token2

    def test_create_token_stores_in_memory(self, auth):
        """Token is stored in internal dict."""
        token = auth.create_token("user123")
        assert token in auth._tokens
        assert auth._tokens[token]["user_id"] == "user123"

    def test_create_token_sets_expiry(self, auth, mock_settings):
        """Token has expiry set based on settings."""
        before = datetime.now()
        token = auth.create_token("user123")
        after = datetime.now()

        expiry = auth._tokens[token]["expiry"]
        expected_min = before + timedelta(minutes=mock_settings.access_token_expire_minutes)
        expected_max = after + timedelta(minutes=mock_settings.access_token_expire_minutes)

        assert expected_min <= expiry <= expected_max

    def test_verify_token_valid(self, auth):
        """verify_token returns user_id for valid token."""
        token = auth.create_token("user456")
        result = auth.verify_token(token)
        assert result == "user456"

    def test_verify_token_invalid(self, auth):
        """verify_token returns None for invalid token."""
        result = auth.verify_token("invalid_token_xyz")
        assert result is None

    def test_verify_token_expired(self, auth, mock_settings):
        """verify_token returns None for expired token."""
        # Set very short expiry
        mock_settings.access_token_expire_minutes = 0

        with patch("animus_forge.auth.token_auth.get_settings", return_value=mock_settings):
            short_auth = TokenAuth()

        token = short_auth.create_token("user789")

        # Manually expire the token
        short_auth._tokens[token]["expiry"] = datetime.now() - timedelta(seconds=1)

        result = short_auth.verify_token(token)
        assert result is None

    def test_verify_token_removes_expired(self, auth):
        """Expired token is removed from storage."""
        token = auth.create_token("user123")

        # Manually expire
        auth._tokens[token]["expiry"] = datetime.now() - timedelta(seconds=1)

        auth.verify_token(token)
        assert token not in auth._tokens

    def test_revoke_token_success(self, auth):
        """revoke_token returns True and removes token."""
        token = auth.create_token("user123")
        result = auth.revoke_token(token)

        assert result is True
        assert token not in auth._tokens

    def test_revoke_token_not_found(self, auth):
        """revoke_token returns False for unknown token."""
        result = auth.revoke_token("nonexistent_token")
        assert result is False

    def test_revoke_token_already_revoked(self, auth):
        """revoke_token returns False if already revoked."""
        token = auth.create_token("user123")
        auth.revoke_token(token)

        result = auth.revoke_token(token)
        assert result is False

    def test_multiple_users(self, auth):
        """Can manage tokens for multiple users."""
        token1 = auth.create_token("user1")
        token2 = auth.create_token("user2")
        token3 = auth.create_token("user3")

        assert auth.verify_token(token1) == "user1"
        assert auth.verify_token(token2) == "user2"
        assert auth.verify_token(token3) == "user3"

    def test_revoke_one_keeps_others(self, auth):
        """Revoking one token doesn't affect others."""
        token1 = auth.create_token("user1")
        token2 = auth.create_token("user2")

        auth.revoke_token(token1)

        assert auth.verify_token(token1) is None
        assert auth.verify_token(token2) == "user2"

    def test_same_user_multiple_tokens(self, auth):
        """Same user can have multiple tokens."""
        token1 = auth.create_token("user1")
        time.sleep(0.01)
        token2 = auth.create_token("user1")

        assert auth.verify_token(token1) == "user1"
        assert auth.verify_token(token2) == "user1"

        # Revoke one, other still works
        auth.revoke_token(token1)
        assert auth.verify_token(token1) is None
        assert auth.verify_token(token2) == "user1"


class TestGlobalFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_global_auth(self):
        """Reset global auth instance before each test."""
        import animus_forge.auth.token_auth as auth_module

        # Store original
        original = auth_module._auth

        # Create fresh mock settings
        mock_settings = MagicMock()
        mock_settings.access_token_expire_minutes = 30

        with patch("animus_forge.auth.token_auth.get_settings", return_value=mock_settings):
            auth_module._auth = TokenAuth()

        yield

        # Restore
        auth_module._auth = original

    def test_create_access_token(self):
        """create_access_token creates valid token."""
        token = create_access_token("test_user")
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_verify_token_function(self):
        """verify_token function works with global auth."""
        token = create_access_token("test_user")
        result = verify_token(token)
        assert result == "test_user"

    def test_verify_invalid_token_function(self):
        """verify_token returns None for invalid token."""
        result = verify_token("invalid")
        assert result is None


class TestTokenFormat:
    """Tests for token format and structure."""

    @pytest.fixture
    def auth(self):
        """Create TokenAuth with mocked settings."""
        mock_settings = MagicMock()
        mock_settings.access_token_expire_minutes = 30
        with patch("animus_forge.auth.token_auth.get_settings", return_value=mock_settings):
            return TokenAuth()

    def test_token_is_url_safe(self, auth):
        """Token is URL-safe base64."""
        token = auth.create_token("user")
        # secrets.token_urlsafe uses A-Z, a-z, 0-9, -, _
        import re

        assert re.fullmatch(r"[A-Za-z0-9_-]+", token)

    def test_token_has_sufficient_entropy(self, auth):
        """Token has at least 32 bytes of entropy (43+ chars in base64)."""
        token = auth.create_token("user")
        assert len(token) >= 43

    def test_special_characters_in_user_id(self, auth):
        """Handles special characters in user ID."""
        token = auth.create_token("user@email.com")
        result = auth.verify_token(token)
        assert result == "user@email.com"

    def test_empty_user_id(self, auth):
        """Handles empty user ID."""
        token = auth.create_token("")
        result = auth.verify_token(token)
        assert result == ""

    def test_unicode_user_id(self, auth):
        """Handles unicode in user ID."""
        token = auth.create_token("用户123")
        result = auth.verify_token(token)
        assert result == "用户123"
