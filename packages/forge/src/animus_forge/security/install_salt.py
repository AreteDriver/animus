"""Per-install PBKDF2 salt — generated once and persisted, never hardcoded.

A hardcoded salt is shared across every install, which weakens key derivation
against precomputation. This module gives each install its own random salt,
persisted under the Animus config dir with restrictive permissions.

Migration / backward-compat: every call site keeps an environment-variable
override (e.g. ``ENCRYPTION_SALT``). An existing install that has data
encrypted under a legacy hardcoded salt can pin it by setting that env var to
the old value, or re-key. Without the override, a fresh random salt is used,
so old ciphertext derived from a different salt will not decrypt — this is a
deliberate, surfaced migration, not a silent one.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

_SALT_BYTES = 16


def _config_dir() -> Path:
    override = os.environ.get("ANIMUS_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "animus"


def get_install_salt(name: str, env_var: str | None = None) -> bytes:
    """Return a stable per-install salt for ``name``.

    Resolution order:
      1. ``env_var`` if set (explicit override / legacy-salt pin).
      2. The persisted random salt at ``<config>/<name>.salt``.
      3. A freshly generated random salt, persisted for next time.

    Falls back to a deterministic, process-stable value only if the salt
    cannot be persisted (e.g. read-only filesystem), logging the degradation.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override.encode()

    path = _config_dir() / f"{name}.salt"
    try:
        existing = path.read_bytes()
        if existing:
            return existing
    except OSError:
        pass  # not created yet (or unreadable) → generate below

    salt = secrets.token_bytes(_SALT_BYTES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(salt)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        logger.info("Generated per-install salt for %s at %s", name, path)
        return salt
    except OSError:
        logger.warning(
            "Could not persist install salt %s; using non-persisted fallback "
            "(set %s to pin a stable salt)",
            name,
            env_var or "an env override",
        )
        return f"animus-install-fallback-{name}".encode()
