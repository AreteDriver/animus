"""Email notification channel."""

from __future__ import annotations

import logging

from ..base import NotificationChannel
from ..models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class EmailChannel(NotificationChannel):
    """Send notifications via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        from_addr: str = "gorgon@localhost",
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
    ):
        """Initialize email channel.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port (default 587 for TLS)
            username: SMTP username (optional)
            password: SMTP password (optional)
            from_addr: From email address
            to_addrs: List of recipient email addresses
            use_tls: Whether to use TLS (default True)
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls

    def name(self) -> str:
        return "email"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification via email."""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        if not self.to_addrs:
            logger.warning("Email channel has no recipients configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Gorgon] {self._event_emoji(event.event_type)} {event.workflow_name}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            # Plain text version
            text_body = self._format_text(event)
            msg.attach(MIMEText(text_body, "plain"))

            # HTML version
            html_body = self._format_html(event)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    def _event_emoji(self, event_type: EventType) -> str:
        emojis = {
            EventType.WORKFLOW_STARTED: "\u25b6\ufe0f",
            EventType.WORKFLOW_COMPLETED: "\u2705",
            EventType.WORKFLOW_FAILED: "\u274c",
            EventType.STEP_COMPLETED: "\u2611\ufe0f",
            EventType.STEP_FAILED: "\u26a0\ufe0f",
            EventType.BUDGET_WARNING: "\U0001f4b0",
            EventType.BUDGET_EXCEEDED: "\U0001f6ab",
            EventType.SCHEDULE_TRIGGERED: "\u23f0",
        }
        return emojis.get(event_type, "\U0001f514")

    def _format_text(self, event: NotificationEvent) -> str:
        lines = [
            f"Workflow: {event.workflow_name}",
            f"Event: {event.event_type.value}",
            f"Message: {event.message}",
            f"Severity: {event.severity}",
            f"Time: {event.timestamp.isoformat()}",
            "",
            "Details:",
        ]
        for key, value in event.details.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _format_html(self, event: NotificationEvent) -> str:
        severity_colors = {
            "info": "#3498db",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c",
        }
        color = severity_colors.get(event.severity, "#95a5a6")

        details_html = "".join(
            f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in event.details.items()
        )

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {color}; padding-left: 16px; margin: 16px 0;">
                <h2 style="margin: 0;">{event.workflow_name}</h2>
                <p style="color: #666; margin: 8px 0;">{event.message}</p>
                <table style="border-collapse: collapse; margin-top: 16px;">
                    <tr><td><strong>Event</strong></td><td>{event.event_type.value}</td></tr>
                    <tr><td><strong>Severity</strong></td><td style="color: {color};">{event.severity.upper()}</td></tr>
                    <tr><td><strong>Time</strong></td><td>{event.timestamp.isoformat()}</td></tr>
                    {details_html}
                </table>
            </div>
            <p style="color: #999; font-size: 12px;">Sent by Gorgon Workflow Engine</p>
        </body>
        </html>
        """
