"""Google Calendar API client wrapper.

Provides read/write access to Google Calendar for Clawdbot-style operation:
- List events
- Create events
- Update events
- Delete events
- Check availability
- Manage reminders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from animus_forge.api_clients.resilience import resilient_call
from animus_forge.config import get_settings
from animus_forge.errors import MaxRetriesError
from animus_forge.utils.retry import with_retry

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """Represents a calendar event."""

    id: str | None = None
    summary: str = ""
    description: str = ""
    location: str = ""
    start: datetime | None = None
    end: datetime | None = None
    all_day: bool = False
    attendees: list[str] = field(default_factory=list)
    reminders: list[dict] = field(default_factory=list)
    recurrence: list[str] = field(default_factory=list)
    status: str = "confirmed"  # confirmed, tentative, cancelled
    html_link: str = ""
    calendar_id: str = "primary"

    def to_api_format(self) -> dict:
        """Convert to Google Calendar API format."""
        event = {
            "summary": self.summary,
            "description": self.description,
            "location": self.location,
            "status": self.status,
        }

        # Handle all-day vs timed events
        if self.all_day:
            if self.start:
                event["start"] = {"date": self.start.strftime("%Y-%m-%d")}
            if self.end:
                event["end"] = {"date": self.end.strftime("%Y-%m-%d")}
        else:
            if self.start:
                event["start"] = {
                    "dateTime": self.start.isoformat(),
                    "timeZone": "UTC",
                }
            if self.end:
                event["end"] = {
                    "dateTime": self.end.isoformat(),
                    "timeZone": "UTC",
                }

        # Add attendees
        if self.attendees:
            event["attendees"] = [{"email": email} for email in self.attendees]

        # Add reminders
        if self.reminders:
            event["reminders"] = {
                "useDefault": False,
                "overrides": self.reminders,
            }
        else:
            event["reminders"] = {"useDefault": True}

        # Add recurrence
        if self.recurrence:
            event["recurrence"] = self.recurrence

        return event

    @classmethod
    def from_api_response(cls, data: dict) -> CalendarEvent:
        """Create from Google Calendar API response."""
        start_data = data.get("start", {})
        end_data = data.get("end", {})

        # Determine if all-day event
        all_day = "date" in start_data

        # Parse start time
        start = None
        if "dateTime" in start_data:
            start = datetime.fromisoformat(start_data["dateTime"].replace("Z", "+00:00"))
        elif "date" in start_data:
            start = datetime.strptime(start_data["date"], "%Y-%m-%d").replace(tzinfo=UTC)

        # Parse end time
        end = None
        if "dateTime" in end_data:
            end = datetime.fromisoformat(end_data["dateTime"].replace("Z", "+00:00"))
        elif "date" in end_data:
            end = datetime.strptime(end_data["date"], "%Y-%m-%d").replace(tzinfo=UTC)

        # Extract attendees
        attendees = [att.get("email", "") for att in data.get("attendees", []) if att.get("email")]

        return cls(
            id=data.get("id"),
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            location=data.get("location", ""),
            start=start,
            end=end,
            all_day=all_day,
            attendees=attendees,
            reminders=data.get("reminders", {}).get("overrides", []),
            recurrence=data.get("recurrence", []),
            status=data.get("status", "confirmed"),
            html_link=data.get("htmlLink", ""),
        )


class CalendarClient:
    """Wrapper for Google Calendar API.

    Provides full CRUD operations for calendar events.

    Usage:
        client = CalendarClient()
        if client.authenticate():
            # List upcoming events
            events = client.list_events(max_results=10)

            # Create an event
            event = CalendarEvent(
                summary="Team Meeting",
                start=datetime.now() + timedelta(hours=1),
                end=datetime.now() + timedelta(hours=2),
            )
            created = client.create_event(event)

            # Check availability
            busy = client.check_availability(
                start=datetime.now(),
                end=datetime.now() + timedelta(days=1),
            )
    """

    # Scopes for full calendar access
    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    def __init__(self, credentials_path: str | None = None):
        """Initialize Calendar client.

        Args:
            credentials_path: Path to OAuth credentials JSON file.
                             Uses settings if not provided.
        """
        settings = get_settings()
        self.credentials_path = credentials_path or settings.gmail_credentials_path
        self.service = None
        self._authenticated = False

    def is_configured(self) -> bool:
        """Check if Calendar client is configured."""
        return self.credentials_path is not None

    def authenticate(self, token_path: str = "calendar_token.json") -> bool:
        """Authenticate with Google Calendar API.

        Args:
            token_path: Path to store OAuth token.

        Returns:
            True if authentication succeeded.
        """
        if not self.is_configured():
            logger.warning("Calendar credentials not configured")
            return False

        try:
            import os
            import os.path

            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            creds = None

            # Load existing token
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save token securely
                fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as token:
                    token.write(creds.to_json())

            self.service = build("calendar", "v3", credentials=creds)
            self._authenticated = True
            logger.info("Calendar authentication successful")
            return True

        except Exception as e:
            logger.exception(f"Calendar authentication failed: {e}")
            return False

    def list_events(
        self,
        calendar_id: str = "primary",
        max_results: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        query: str | None = None,
        single_events: bool = True,
        order_by: str = "startTime",
    ) -> list[CalendarEvent]:
        """List calendar events.

        Args:
            calendar_id: Calendar ID (default: primary).
            max_results: Maximum events to return.
            time_min: Start of time range (default: now).
            time_max: End of time range.
            query: Free text search query.
            single_events: Expand recurring events.
            order_by: Sort order (startTime or updated).

        Returns:
            List of CalendarEvent objects.
        """
        if not self.service:
            logger.warning("Calendar service not authenticated")
            return []

        try:
            return self._list_events_with_retry(
                calendar_id,
                max_results,
                time_min,
                time_max,
                query,
                single_events,
                order_by,
            )
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to list events: {e}")
            return []

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _list_events_with_retry(
        self,
        calendar_id: str,
        max_results: int,
        time_min: datetime | None,
        time_max: datetime | None,
        query: str | None,
        single_events: bool,
        order_by: str,
    ) -> list[CalendarEvent]:
        """List events with retry logic."""
        # Default to now if no time_min
        if time_min is None:
            time_min = datetime.now(UTC)

        params = {
            "calendarId": calendar_id,
            "maxResults": max_results,
            "timeMin": time_min.isoformat(),
            "singleEvents": single_events,
            "orderBy": order_by,
        }

        if time_max:
            params["timeMax"] = time_max.isoformat()
        if query:
            params["q"] = query

        results = self.service.events().list(**params).execute()
        items = results.get("items", [])

        return [CalendarEvent.from_api_response(item) for item in items]

    def get_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> CalendarEvent | None:
        """Get a specific event.

        Args:
            event_id: Event ID.
            calendar_id: Calendar ID.

        Returns:
            CalendarEvent or None if not found.
        """
        if not self.service:
            return None

        try:
            return self._get_event_with_retry(event_id, calendar_id)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to get event: {e}")
            return None

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _get_event_with_retry(
        self,
        event_id: str,
        calendar_id: str,
    ) -> CalendarEvent:
        """Get event with retry logic."""
        result = self.service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        return CalendarEvent.from_api_response(result)

    def create_event(
        self,
        event: CalendarEvent,
        calendar_id: str = "primary",
        send_notifications: bool = False,
    ) -> CalendarEvent | None:
        """Create a new calendar event.

        Args:
            event: CalendarEvent to create.
            calendar_id: Calendar ID.
            send_notifications: Send notifications to attendees.

        Returns:
            Created CalendarEvent with ID, or None on failure.
        """
        if not self.service:
            logger.warning("Calendar service not authenticated")
            return None

        try:
            return self._create_event_with_retry(event, calendar_id, send_notifications)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to create event: {e}")
            return None

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _create_event_with_retry(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool,
    ) -> CalendarEvent:
        """Create event with retry logic."""
        body = event.to_api_format()
        result = (
            self.service.events()
            .insert(
                calendarId=calendar_id,
                body=body,
                sendUpdates="all" if send_notifications else "none",
            )
            .execute()
        )
        logger.info(f"Created event: {result.get('id')}")
        return CalendarEvent.from_api_response(result)

    def update_event(
        self,
        event: CalendarEvent,
        calendar_id: str = "primary",
        send_notifications: bool = False,
    ) -> CalendarEvent | None:
        """Update an existing event.

        Args:
            event: CalendarEvent with ID to update.
            calendar_id: Calendar ID.
            send_notifications: Send notifications to attendees.

        Returns:
            Updated CalendarEvent, or None on failure.
        """
        if not self.service:
            return None

        if not event.id:
            logger.error("Cannot update event without ID")
            return None

        try:
            return self._update_event_with_retry(event, calendar_id, send_notifications)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to update event: {e}")
            return None

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _update_event_with_retry(
        self,
        event: CalendarEvent,
        calendar_id: str,
        send_notifications: bool,
    ) -> CalendarEvent:
        """Update event with retry logic."""
        body = event.to_api_format()
        result = (
            self.service.events()
            .update(
                calendarId=calendar_id,
                eventId=event.id,
                body=body,
                sendUpdates="all" if send_notifications else "none",
            )
            .execute()
        )
        logger.info(f"Updated event: {event.id}")
        return CalendarEvent.from_api_response(result)

    def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_notifications: bool = False,
    ) -> bool:
        """Delete an event.

        Args:
            event_id: Event ID to delete.
            calendar_id: Calendar ID.
            send_notifications: Send cancellation notifications.

        Returns:
            True if deleted successfully.
        """
        if not self.service:
            return False

        try:
            return self._delete_event_with_retry(event_id, calendar_id, send_notifications)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to delete event: {e}")
            return False

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _delete_event_with_retry(
        self,
        event_id: str,
        calendar_id: str,
        send_notifications: bool,
    ) -> bool:
        """Delete event with retry logic."""
        self.service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates="all" if send_notifications else "none",
        ).execute()
        logger.info(f"Deleted event: {event_id}")
        return True

    def check_availability(
        self,
        start: datetime,
        end: datetime,
        calendar_ids: list[str] | None = None,
    ) -> list[dict]:
        """Check calendar availability (free/busy).

        Args:
            start: Start of time range.
            end: End of time range.
            calendar_ids: List of calendar IDs (default: primary).

        Returns:
            List of busy time periods.
        """
        if not self.service:
            return []

        try:
            return self._check_availability_with_retry(start, end, calendar_ids)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to check availability: {e}")
            return []

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _check_availability_with_retry(
        self,
        start: datetime,
        end: datetime,
        calendar_ids: list[str] | None,
    ) -> list[dict]:
        """Check availability with retry logic."""
        if calendar_ids is None:
            calendar_ids = ["primary"]

        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": cal_id} for cal_id in calendar_ids],
        }

        result = self.service.freebusy().query(body=body).execute()
        calendars = result.get("calendars", {})

        busy_periods = []
        for cal_id, data in calendars.items():
            for busy in data.get("busy", []):
                busy_periods.append(
                    {
                        "calendar_id": cal_id,
                        "start": busy.get("start"),
                        "end": busy.get("end"),
                    }
                )

        return busy_periods

    def list_calendars(self) -> list[dict]:
        """List all calendars.

        Returns:
            List of calendar info dicts.
        """
        if not self.service:
            return []

        try:
            return self._list_calendars_with_retry()
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to list calendars: {e}")
            return []

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _list_calendars_with_retry(self) -> list[dict]:
        """List calendars with retry logic."""
        result = self.service.calendarList().list().execute()
        items = result.get("items", [])

        return [
            {
                "id": item.get("id"),
                "summary": item.get("summary"),
                "description": item.get("description", ""),
                "primary": item.get("primary", False),
                "access_role": item.get("accessRole"),
                "background_color": item.get("backgroundColor"),
            }
            for item in items
        ]

    def quick_add(
        self,
        text: str,
        calendar_id: str = "primary",
    ) -> CalendarEvent | None:
        """Create an event using natural language.

        Args:
            text: Natural language event description.
                  e.g., "Lunch with Bob tomorrow at noon"
            calendar_id: Calendar ID.

        Returns:
            Created CalendarEvent, or None on failure.
        """
        if not self.service:
            return None

        try:
            return self._quick_add_with_retry(text, calendar_id)
        except (MaxRetriesError, Exception) as e:
            logger.exception(f"Failed to quick add: {e}")
            return None

    @resilient_call("calendar")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _quick_add_with_retry(
        self,
        text: str,
        calendar_id: str,
    ) -> CalendarEvent:
        """Quick add with retry logic."""
        result = self.service.events().quickAdd(calendarId=calendar_id, text=text).execute()
        logger.info(f"Quick added event: {result.get('id')}")
        return CalendarEvent.from_api_response(result)

    def get_upcoming_today(self) -> list[CalendarEvent]:
        """Get today's remaining events.

        Returns:
            List of events from now until end of day.
        """
        now = datetime.now(UTC)
        end_of_day = now.replace(hour=23, minute=59, second=59)

        return self.list_events(
            time_min=now,
            time_max=end_of_day,
            max_results=50,
        )

    def get_tomorrow(self) -> list[CalendarEvent]:
        """Get tomorrow's events.

        Returns:
            List of tomorrow's events.
        """
        tomorrow = datetime.now(UTC) + timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0)
        end = tomorrow.replace(hour=23, minute=59, second=59)

        return self.list_events(
            time_min=start,
            time_max=end,
            max_results=50,
        )
