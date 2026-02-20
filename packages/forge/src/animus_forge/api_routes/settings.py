"""Settings and API key management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header

from animus_forge import api_state as state
from animus_forge.api_errors import AUTH_RESPONSES, CRUD_RESPONSES, bad_request, not_found
from animus_forge.api_models import APIKeyCreateRequest, PreferencesUpdateRequest
from animus_forge.api_routes.auth import verify_auth

router = APIRouter()


@router.get("/settings/preferences", responses=AUTH_RESPONSES)
def get_preferences(authorization: str | None = Header(None)):
    """Get user preferences."""
    user_id = verify_auth(authorization)

    prefs = state.settings_manager.get_preferences(user_id)
    return prefs.model_dump(mode="json")


@router.post("/settings/preferences", responses=AUTH_RESPONSES)
def update_preferences(
    request: PreferencesUpdateRequest,
    authorization: str | None = Header(None),
):
    """Update user preferences."""
    user_id = verify_auth(authorization)

    from animus_forge.settings.models import NotificationSettings, UserPreferencesUpdate

    update_data = {}
    if request.theme is not None:
        if request.theme not in ("light", "dark", "system"):
            raise bad_request("Invalid theme", {"valid_values": ["light", "dark", "system"]})
        update_data["theme"] = request.theme
    if request.compact_view is not None:
        update_data["compact_view"] = request.compact_view
    if request.show_costs is not None:
        update_data["show_costs"] = request.show_costs
    if request.default_page_size is not None:
        if request.default_page_size < 10 or request.default_page_size > 100:
            raise bad_request("Invalid page size", {"valid_range": "10-100"})
        update_data["default_page_size"] = request.default_page_size
    if request.notifications is not None:
        update_data["notifications"] = NotificationSettings(**request.notifications)

    update = UserPreferencesUpdate(**update_data)
    prefs = state.settings_manager.update_preferences(user_id, update)
    return prefs.model_dump(mode="json")


@router.get("/settings/api-keys", responses=AUTH_RESPONSES)
def get_api_keys(authorization: str | None = Header(None)):
    """Get API key metadata (keys are masked)."""
    user_id = verify_auth(authorization)

    keys = state.settings_manager.get_api_keys(user_id)
    return [k.model_dump(mode="json") for k in keys]


@router.get("/settings/api-keys/status", responses=AUTH_RESPONSES)
def get_api_key_status(authorization: str | None = Header(None)):
    """Get status of which API keys are configured."""
    user_id = verify_auth(authorization)

    api_status = state.settings_manager.get_api_key_status(user_id)
    return api_status.model_dump()


@router.post("/settings/api-keys", responses=AUTH_RESPONSES)
def set_api_key(
    request: APIKeyCreateRequest,
    authorization: str | None = Header(None),
):
    """Set or update an API key."""
    user_id = verify_auth(authorization)

    if request.provider not in ("openai", "anthropic", "github"):
        raise bad_request(
            "Invalid provider",
            {"valid_providers": ["openai", "anthropic", "github"]},
        )

    if not request.key or len(request.key) < 10:
        raise bad_request("API key is too short")

    from animus_forge.settings.models import APIKeyCreate

    key_create = APIKeyCreate(provider=request.provider, key=request.key)
    key_info = state.settings_manager.set_api_key(user_id, key_create)
    return {
        "status": "success",
        "key": key_info.model_dump(mode="json"),
    }


@router.delete("/settings/api-keys/{provider}", responses=CRUD_RESPONSES)
def delete_api_key(provider: str, authorization: str | None = Header(None)):
    """Delete an API key."""
    user_id = verify_auth(authorization)

    if provider not in ("openai", "anthropic", "github"):
        raise bad_request(
            "Invalid provider",
            {"valid_providers": ["openai", "anthropic", "github"]},
        )

    if state.settings_manager.delete_api_key(user_id, provider):
        return {"status": "success"}

    raise not_found("API Key", provider)
