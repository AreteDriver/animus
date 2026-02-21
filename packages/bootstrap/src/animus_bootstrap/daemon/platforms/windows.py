"""Windows service management — stub implementation.

All methods raise NotImplementedError. Windows support is planned for a
future release. pywin32 is NOT imported at module level to avoid import
errors on non-Windows systems.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_MSG = (
    "Windows service management is not yet implemented. "
    "See https://github.com/Arete-Consortium/animus/issues for status."
)


class WindowsService:
    """Stub for Windows service management.

    All methods raise NotImplementedError with a clear message.
    pywin32 is intentionally NOT imported at module level.
    """

    def install_service(self) -> bool:
        """Install Animus as a Windows service."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def start(self) -> bool:
        """Start the Animus Windows service."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def stop(self) -> bool:
        """Stop the Animus Windows service."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def is_running(self) -> bool:
        """Check if the Animus Windows service is running."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def uninstall_service(self) -> bool:
        """Uninstall the Animus Windows service."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
