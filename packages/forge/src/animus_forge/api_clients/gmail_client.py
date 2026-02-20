"""Gmail API client wrapper."""

from animus_forge.api_clients.resilience import resilient_call
from animus_forge.config import get_settings
from animus_forge.errors import MaxRetriesError
from animus_forge.utils.retry import with_retry


class GmailClient:
    """Wrapper for Gmail API."""

    def __init__(self):
        settings = get_settings()
        self.credentials_path = settings.gmail_credentials_path
        self.service = None

    def is_configured(self) -> bool:
        """Check if Gmail client is configured."""
        return self.credentials_path is not None

    def authenticate(self) -> bool:
        """Authenticate with Gmail API."""
        if not self.is_configured():
            return False

        try:
            import os.path

            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
            creds = None

            token_path = "token.json"
            if os.path.exists(token_path):
                from google.oauth2.credentials import Credentials

                creds = Credentials.from_authorized_user_file(token_path, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)

                import os

                fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as token:
                    token.write(creds.to_json())

            self.service = build("gmail", "v1", credentials=creds)
            return True
        except Exception:
            return False

    def list_messages(self, max_results: int = 10, query: str | None = None) -> list[dict]:
        """List Gmail messages."""
        if not self.service:
            return []

        try:
            return self._list_messages_with_retry(max_results, query)
        except (MaxRetriesError, Exception):
            return []

    @resilient_call("gmail")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _list_messages_with_retry(self, max_results: int, query: str | None) -> list[dict]:
        """List messages with retry logic."""
        results = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=max_results, q=query or "")
            .execute()
        )
        return results.get("messages", [])

    def get_message(self, message_id: str) -> dict | None:
        """Get a specific message."""
        if not self.service:
            return None

        try:
            return self._get_message_with_retry(message_id)
        except (MaxRetriesError, Exception):
            return None

    @resilient_call("gmail")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _get_message_with_retry(self, message_id: str) -> dict:
        """Get message with retry logic."""
        return (
            self.service.users().messages().get(userId="me", id=message_id, format="full").execute()
        )

    def extract_email_body(self, message: dict) -> str:
        """Extract email body from message."""
        try:
            import base64

            if "parts" in message["payload"]:
                for part in message["payload"]["parts"]:
                    if part["mimeType"] == "text/plain":
                        data = part["body"].get("data", "")
                        return base64.urlsafe_b64decode(data).decode("utf-8")
            else:
                data = message["payload"]["body"].get("data", "")
                return base64.urlsafe_b64decode(data).decode("utf-8")
        except Exception:
            return ""
