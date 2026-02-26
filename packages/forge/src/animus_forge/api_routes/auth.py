"""Authentication routes — login and token verification."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, Request
from slowapi.util import get_remote_address

from animus_forge import api_state as state
from animus_forge.api_errors import responses, unauthorized
from animus_forge.api_models import LoginRequest, LoginResponse
from animus_forge.auth import create_access_token, verify_token
from animus_forge.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_auth(authorization: str | None = Header(None)) -> str:
    """Verify authentication token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthorized("Authentication required. Provide Bearer token in Authorization header.")

    parts = authorization.split(" ", maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise unauthorized("Malformed Authorization header. Expected: Bearer <token>")
    token = parts[1]
    user_id = verify_token(token)

    if not user_id:
        raise unauthorized("Invalid or expired token")

    return user_id


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    responses=responses(401, 429),
)
@state.limiter.limit("5/minute")
def login(request: Request, login_request: LoginRequest):
    """Login endpoint. Rate limited to 5 requests/minute per IP.

    Authentication methods (in priority order):
    1. Configured credentials via API_CREDENTIALS env var
    2. Demo auth (password='demo') if ALLOW_DEMO_AUTH=true (default in dev)

    Configure credentials (bcrypt preferred):
        API_CREDENTIALS='user1:$2b$12$...,user2:$2b$12$...'
        Generate hash: python -c "import bcrypt; print(bcrypt.hashpw(b'password', bcrypt.gensalt()).decode())"
    """
    settings = get_settings()

    if settings.verify_credentials(login_request.user_id, login_request.password):
        token = create_access_token(login_request.user_id)
        logger.info("User '%s' logged in successfully", login_request.user_id)
        return LoginResponse(access_token=token)

    logger.warning(
        "Failed login attempt for user '%s' from IP %s",
        login_request.user_id,
        get_remote_address(request),
    )
    raise unauthorized("Invalid credentials")
