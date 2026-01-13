"""
Google Calendar Integration

Calendar access via Google Calendar API.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.integrations.oauth import (
    GOOGLE_AUTH_AVAILABLE,
    OAuth2Flow,
    OAuth2Token,
    load_token,
    save_token,
)
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.google.calendar")

# Check if Google API client is available
GOOGLE_API_AVAILABLE = False
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    GOOGLE_API_AVAILABLE = True
except ImportError:
    pass

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


class GoogleCalendarIntegration(BaseIntegration):
    """
    Google Calendar integration.

    Provides tools for:
    - Listing calendar events
    - Creating events
    - Checking availability
    """

    name = "google_calendar"
    display_name = "Google Calendar"
    auth_type = AuthType.OAUTH2

    def __init__(self, data_dir: Path | None = None):
        super().__init__()
        self._data_dir = data_dir or Path.home() / ".animus" / "integrations"
        self._service: Any = None
        self._token: OAuth2Token | None = None

    @property
    def _token_path(self) -> Path:
        """Path to stored OAuth token."""
        return self._data_dir / "google_calendar_token.json"

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Connect to Google Calendar.

        Credentials:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            token: Optional OAuth2Token dict if already authorized

        If no token provided, initiates OAuth2 flow.
        """
        if not GOOGLE_AUTH_AVAILABLE or not GOOGLE_API_AVAILABLE:
            self._set_error(
                "Google API libraries not installed. Install with: "
                "pip install google-api-python-client google-auth google-auth-oauthlib"
            )
            return False

        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")

        if not client_id or not client_secret:
            self._set_error("client_id and client_secret required")
            return False

        # Check for existing token
        self._token = load_token(self._token_path)

        # Check for token passed in credentials
        if token_data := credentials.get("token"):
            self._token = OAuth2Token.from_dict(token_data)

        # If no token or expired, run OAuth flow
        if not self._token or self._token.is_expired():
            if self._token and self._token.refresh_token:
                # Try to refresh
                flow = OAuth2Flow(client_id, client_secret, CALENDAR_SCOPES)
                self._token = flow.refresh_token(self._token)

            if not self._token:
                # Run full OAuth flow
                flow = OAuth2Flow(client_id, client_secret, CALENDAR_SCOPES)
                self._token = flow.run_local_server()

            if not self._token:
                self._set_error("OAuth2 authorization failed")
                return False

            # Save token for future use
            self._data_dir.mkdir(parents=True, exist_ok=True)
            save_token(self._token, self._token_path)

        # Build the Calendar service
        try:
            creds = Credentials(
                token=self._token.access_token,
                refresh_token=self._token.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
            )
            self._service = build("calendar", "v3", credentials=creds)
            self._credentials = credentials
            self._set_connected(expires_at=self._token.expires_at)
            logger.info("Connected to Google Calendar")
            return True
        except Exception as e:
            self._set_error(f"Failed to build Calendar service: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Google Calendar."""
        self._service = None
        self._token = None
        # Optionally remove stored token
        if self._token_path.exists():
            self._token_path.unlink()
        self._set_disconnected()
        logger.info("Disconnected from Google Calendar")
        return True

    async def verify(self) -> bool:
        """Verify Google Calendar connection."""
        if not self._service:
            return False
        try:
            self._service.calendarList().list(maxResults=1).execute()
            return True
        except Exception:
            self._set_expired()
            return False

    def get_tools(self) -> list[Tool]:
        """Get Google Calendar tools."""
        return [
            Tool(
                name="calendar_list_events",
                description="List upcoming calendar events",
                parameters={
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look ahead (default: 7)",
                        "required": False,
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "required": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 20)",
                        "required": False,
                    },
                },
                handler=self._tool_list_events,
            ),
            Tool(
                name="calendar_create_event",
                description="Create a new calendar event",
                parameters={
                    "summary": {
                        "type": "string",
                        "description": "Event title",
                        "required": True,
                    },
                    "start": {
                        "type": "string",
                        "description": "Start time (ISO format, e.g., '2024-01-15T10:00:00')",
                        "required": True,
                    },
                    "end": {
                        "type": "string",
                        "description": "End time (ISO format)",
                        "required": True,
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                        "required": False,
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location",
                        "required": False,
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "required": False,
                    },
                },
                handler=self._tool_create_event,
            ),
            Tool(
                name="calendar_check_availability",
                description="Check free/busy times for a date range",
                parameters={
                    "start": {
                        "type": "string",
                        "description": "Start of range (ISO format)",
                        "required": True,
                    },
                    "end": {
                        "type": "string",
                        "description": "End of range (ISO format)",
                        "required": True,
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "required": False,
                    },
                },
                handler=self._tool_check_availability,
            ),
            Tool(
                name="calendar_list_calendars",
                description="List available calendars",
                parameters={},
                handler=self._tool_list_calendars,
            ),
        ]

    async def _tool_list_events(
        self,
        days: int = 7,
        calendar_id: str = "primary",
        max_results: int = 20,
    ) -> ToolResult:
        """List upcoming events."""
        if not self._service:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error="Not connected to Google Calendar",
            )

        try:
            now = datetime.utcnow()
            time_min = now.isoformat() + "Z"
            time_max = (now + timedelta(days=days)).isoformat() + "Z"

            events_result = (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            event_list = []

            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))

                event_list.append(
                    {
                        "id": event["id"],
                        "summary": event.get("summary", "(No title)"),
                        "start": start,
                        "end": end,
                        "location": event.get("location"),
                        "description": event.get("description", "")[:200],
                        "html_link": event.get("htmlLink"),
                    }
                )

            return ToolResult(
                tool_name="calendar_tool",
                success=True,
                output={
                    "calendar_id": calendar_id,
                    "days": days,
                    "count": len(event_list),
                    "events": event_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error=f"Failed to list events: {e}",
            )

    async def _tool_create_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
    ) -> ToolResult:
        """Create a calendar event."""
        if not self._service:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error="Not connected to Google Calendar",
            )

        try:
            # Parse and format times
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

            event_body: dict[str, Any] = {
                "summary": summary,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
            }

            if description:
                event_body["description"] = description
            if location:
                event_body["location"] = location

            event = self._service.events().insert(calendarId=calendar_id, body=event_body).execute()

            return ToolResult(
                tool_name="calendar_tool",
                success=True,
                output={
                    "id": event["id"],
                    "summary": event.get("summary"),
                    "html_link": event.get("htmlLink"),
                    "start": event["start"].get("dateTime"),
                    "end": event["end"].get("dateTime"),
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error=f"Failed to create event: {e}",
            )

    async def _tool_check_availability(
        self,
        start: str,
        end: str,
        calendar_id: str = "primary",
    ) -> ToolResult:
        """Check free/busy times."""
        if not self._service:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error="Not connected to Google Calendar",
            )

        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

            body = {
                "timeMin": start_dt.isoformat(),
                "timeMax": end_dt.isoformat(),
                "items": [{"id": calendar_id}],
            }

            result = self._service.freebusy().query(body=body).execute()
            busy_times = result["calendars"][calendar_id]["busy"]

            return ToolResult(
                tool_name="calendar_tool",
                success=True,
                output={
                    "calendar_id": calendar_id,
                    "start": start,
                    "end": end,
                    "busy_periods": busy_times,
                    "busy_count": len(busy_times),
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error=f"Failed to check availability: {e}",
            )

    async def _tool_list_calendars(self) -> ToolResult:
        """List available calendars."""
        if not self._service:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error="Not connected to Google Calendar",
            )

        try:
            result = self._service.calendarList().list().execute()
            calendars = result.get("items", [])

            calendar_list = [
                {
                    "id": cal["id"],
                    "summary": cal.get("summary"),
                    "primary": cal.get("primary", False),
                    "access_role": cal.get("accessRole"),
                }
                for cal in calendars
            ]

            return ToolResult(
                tool_name="calendar_tool",
                success=True,
                output={
                    "count": len(calendar_list),
                    "calendars": calendar_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="calendar_tool",
                success=False,
                output=None,
                error=f"Failed to list calendars: {e}",
            )
