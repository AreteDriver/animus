"""SHA-256 baseline checker for critical hardening code.

Tracks a fixed set of files that implement the boundary gates. On
daemon startup, computes their hashes and compares to a signed-on-disk
baseline. Drift = refuse to boot.

The tracked file set is intentionally small and curated. Adding files
to the set requires regenerating the baseline. This is the right
posture: explicit, auditable, manually attested.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("integrity")


class IntegrityMismatchError(RuntimeError):
    """Raised when a tracked file's hash differs from the baseline."""


class IntegrityNotInitializedError(RuntimeError):
    """Raised when the baseline file is missing and override isn't set."""


# Critical-path files — relative to the animus package root.
# Each path is resolved against ``Path(animus.__file__).parent`` at runtime
# so tests can target a synthetic tree without hardcoded absolute paths.
_TRACKED_RELATIVE_PATHS: tuple[str, ...] = (
    "memory/redaction.py",
    "network/egress.py",
    "mcp_server.py",
    "audit/egress_log.py",
    # A6 — the checker must hash itself (tampering it would otherwise defeat
    # detection) and the immutable learning guardrails.
    "integrity/checker.py",
    "learning/guardrails.py",
)

# A6 — cross-package critical-path modules. Core's ``network/egress.py`` is now
# a re-export shim; the REAL egress + DLP logic lives in ``animus_types``, so
# hashing the shim alone is useless. Resolved via importlib (they live outside
# the core package). REQUIRED ones must be present; OPTIONAL forge adopters are
# hashed only when forge is installed in the running interpreter.
_TRACKED_MODULES_REQUIRED: tuple[str, ...] = (
    "animus_types.egress",
    "animus_types.secrets",
)
_TRACKED_MODULES_OPTIONAL: tuple[str, ...] = (
    "animus_forge.providers.openrouter_provider",
    "animus_forge.security.pi_wrap",
    # C5 — the forge-side egress enforcement surface. A6 hashed the OpenRouter
    # provider but left the modules that actually gate every cloud call
    # untracked: the egress allow/deny engine, the content-aware DLP helper +
    # ``CompletionRequest.scannable_text`` (providers.base), and the
    # sensitivity-aware model router (providers.router / TierRouter). Tampering
    # any of these silently disables the boundary, so they must be in baseline.
    "animus_forge.network.egress",
    "animus_forge.providers.base",
    "animus_forge.providers.router",
)


def _module_file(mod: str) -> Path | None:
    """Resolve an importable module to its source file, or None if absent."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin == "built-in":
        return None
    return Path(spec.origin)


def tracked_module_hashes() -> dict[str, str]:
    """Hash the cross-package tracked modules. Required-missing raises."""
    out: dict[str, str] = {}
    for mod in _TRACKED_MODULES_REQUIRED:
        path = _module_file(mod)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Tracked module missing: {mod}")
        out[f"module:{mod}"] = _hash_file(path)
    for mod in _TRACKED_MODULES_OPTIONAL:
        path = _module_file(mod)
        if path is not None and path.is_file():
            out[f"module:{mod}"] = _hash_file(path)
    return out


def _animus_pkg_root() -> Path:
    """Return the on-disk root of the installed ``animus`` package."""
    import animus

    return Path(animus.__file__).parent


def tracked_files(root: Path | None = None) -> list[Path]:
    """Return the absolute paths of every tracked file (relative + modules)."""
    base = root or _animus_pkg_root()
    paths = [base / rel for rel in _TRACKED_RELATIVE_PATHS]
    for mod in (*_TRACKED_MODULES_REQUIRED, *_TRACKED_MODULES_OPTIONAL):
        mod_path = _module_file(mod)
        if mod_path is not None:
            paths.append(mod_path)
    return paths


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_current(root: Path | None = None) -> dict[str, str]:
    """Compute the current SHA-256 of every tracked file.

    Missing files raise FileNotFoundError — they're load-bearing and
    must be present.
    """
    base = root or _animus_pkg_root()
    out: dict[str, str] = {}
    for rel in _TRACKED_RELATIVE_PATHS:
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(f"Tracked file missing: {path}")
        out[rel] = _hash_file(path)
    # Cross-package modules resolve via importlib (independent of ``root``).
    out.update(tracked_module_hashes())
    return out


def default_baseline_dir() -> Path:
    """Canonical location for the integrity baseline.

    Anchored to ``~/.config/animus/`` (NOT ``AnimusConfig.data_dir``)
    because the latter can shift with env overrides (``ANIMUS_DATA_DIR``
    in the daemon's EnvironmentFile). Config dir is stable across env
    changes and lives in the daemon's writable carve-out.
    """
    return Path.home() / ".config" / "animus"


def baseline_path(baseline_dir: Path) -> Path:
    """Return the manifest file path inside ``baseline_dir``.

    Tests pass a tmp_path here. The live CLI + daemon call
    ``default_baseline_dir()`` and pass that.
    """
    return Path(baseline_dir) / "integrity-baseline.json"


def regenerate_baseline(
    baseline_dir: Path,
    root: Path | None = None,
    *,
    note: str = "",
) -> dict[str, Any]:
    """Write a fresh baseline manifest. Returns the manifest contents.

    Call this only after intentional, attested updates to a tracked
    file. The note field is for human auditability — e.g.
    "Stage 3.D landed; regenerated baseline".
    """
    current = compute_current(root)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        # Every tracked key (relative paths + ``module:*`` entries).
        "tracked_files": sorted(current.keys()),
        "hashes": current,
    }
    path = baseline_path(baseline_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Regenerated integrity baseline at %s (%d files tracked)",
        path,
        len(_TRACKED_RELATIVE_PATHS),
    )
    return manifest


def load_baseline(baseline_dir: Path) -> dict[str, Any]:
    """Load the manifest. Raises if missing."""
    path = baseline_path(baseline_dir)
    if not path.is_file():
        raise IntegrityNotInitializedError(
            f"No integrity baseline at {path}. Run "
            "`python -m animus.integrity.cli regenerate` after the next "
            "intentional update to tracked files, or set "
            "ANIMUS_INTEGRITY_OVERRIDE=1 to bypass (NOT recommended)."
        )
    return json.loads(path.read_text())


def verify_or_raise(baseline_dir: Path, root: Path | None = None) -> None:
    """Check tracked files against the on-disk baseline.

    Behavior:
      - Hashes match → return silently.
      - Mismatch → raise ``IntegrityMismatchError`` listing drifted files.
      - Baseline missing → raise ``IntegrityNotInitializedError``.
      - ``ANIMUS_INTEGRITY_OVERRIDE=1`` env → log warning, skip check.

    This is the daemon-startup hook. Refusing to boot on drift forces
    the operator to either confirm the legitimate update (by
    regenerating the baseline) or investigate the unexpected change.
    """
    if os.environ.get("ANIMUS_INTEGRITY_OVERRIDE") == "1":
        logger.warning(
            "ANIMUS_INTEGRITY_OVERRIDE=1 — tampering check bypassed. "
            "This should only be used during a legitimate update window."
        )
        return

    manifest = load_baseline(baseline_dir)
    expected = manifest.get("hashes", {})
    actual = compute_current(root)

    mismatches: list[tuple[str, str, str]] = []
    for rel, expected_hash in expected.items():
        actual_hash = actual.get(rel)
        if actual_hash != expected_hash:
            mismatches.append((rel, expected_hash, actual_hash or "MISSING"))

    extra = set(actual) - set(expected)
    for rel in extra:
        mismatches.append((rel, "(not in baseline)", actual[rel]))

    if mismatches:
        lines = ["Integrity check failed — tracked file(s) drifted from baseline:"]
        for rel, exp, act in mismatches:
            lines.append(f"  {rel}: expected={exp[:16]}… actual={act[:16]}…")
        lines.append(
            "If this drift is legitimate (intentional update), regenerate "
            "the baseline via `python -m animus.integrity.cli regenerate`. "
            "Otherwise investigate before allowing the daemon to start."
        )
        raise IntegrityMismatchError("\n".join(lines))

    logger.debug(
        "Integrity check passed: %d tracked files match baseline (generated at %s)",
        len(expected),
        manifest.get("generated_at", "unknown"),
    )
