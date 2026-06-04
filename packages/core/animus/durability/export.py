"""Full-state export + cold-rebuild (roadmap A8).

Archive layout (a single ``.zip``)::

    manifest.json          # schema below — inventory + checksums + metadata
    data/<rel>...          # verbatim copy of every file under the data dir

``manifest.json`` schema (``schema_version`` = 1)::

    {
      "schema_version": 1,
      "created_at": "<ISO-8601 UTC>",
      "tool_version": "<animus-core version>",
      "source_data_dir": "<absolute path at export time>",
      "file_count": <int>,
      "total_bytes": <int>,
      "files": { "<rel path under data/>": {"sha256": "<hex>", "bytes": <int>} },
      "config_snapshot": { ... redacted config, secrets removed ... } | null
    }

Secrets are NEVER written to the archive: the optional config snapshot is run
through ``_redact`` (any key whose name looks like a credential is masked), so a
restored machine carries structure but the operator re-enters secrets.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARCHIVE_SCHEMA_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_DATA_PREFIX = "data/"

# Field-name hints for secret values in the config snapshot. ``public`` is
# exempt; redaction is string-only so int budgets are untouched.
_SECRET_HINTS = ("key", "token", "secret", "password", "private", "credential")


class DurabilityError(RuntimeError):
    """Base error for export/rebuild failures."""


class ManifestMismatchError(DurabilityError):
    """A restored/verified file's checksum does not match the manifest."""


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("animus-core")
    except Exception:
        return "unknown"


def _looks_secret(name: str) -> bool:
    n = name.lower()
    if "public" in n:
        return False
    return any(h in n for h in _SECRET_HINTS)


def _redact(obj: Any) -> Any:
    """Recursively mask secret-looking string fields in a config dict."""
    if isinstance(obj, dict):
        return {
            k: ("***redacted***" if isinstance(v, str) and v and _looks_secret(k) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_all(
    data_dir: Path,
    out_path: Path,
    *,
    config_snapshot: dict | None = None,
) -> Path:
    """Export the entire ``data_dir`` to a portable, checksummed archive.

    Args:
        data_dir: The Animus data directory to capture.
        out_path: Archive path; ``.zip`` is appended if absent.
        config_snapshot: Optional config dict — written REDACTED into the
            manifest so a rebuild has structure without secrets.

    Returns:
        The archive path actually written.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DurabilityError(f"data dir does not exist: {data_dir}")
    out_path = Path(out_path)
    if out_path.suffix != ".zip":
        out_path = out_path.with_suffix(".zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    members: list[tuple[Path, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(data_dir).as_posix()
        size = path.stat().st_size
        files[rel] = {"sha256": _sha256(path), "bytes": size}
        total_bytes += size
        members.append((path, _DATA_PREFIX + rel))

    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "tool_version": _tool_version(),
        "source_data_dir": str(data_dir.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "config_snapshot": _redact(config_snapshot) if config_snapshot else None,
    }

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        for src, arcname in members:
            zf.write(src, arcname)
    return out_path


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        raw = zf.read(_MANIFEST_NAME)
    except KeyError as e:
        raise DurabilityError("archive has no manifest.json — not an Animus export") from e
    manifest = json.loads(raw)
    sv = manifest.get("schema_version")
    if sv != ARCHIVE_SCHEMA_VERSION:
        raise DurabilityError(
            f"unsupported archive schema_version={sv} (expected {ARCHIVE_SCHEMA_VERSION})"
        )
    return manifest


def verify_archive(archive_path: Path) -> dict[str, Any]:
    """Verify every ``data/`` member matches its manifest checksum.

    Returns the manifest on success; raises ``ManifestMismatchError`` on the
    first drift (tampered or corrupt archive).
    """
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = _read_manifest(zf)
        names = set(zf.namelist())
        for rel, meta in manifest["files"].items():
            arcname = _DATA_PREFIX + rel
            if arcname not in names:
                raise ManifestMismatchError(f"archive missing file listed in manifest: {rel}")
            h = hashlib.sha256(zf.read(arcname)).hexdigest()
            if h != meta["sha256"]:
                raise ManifestMismatchError(
                    f"checksum mismatch for {rel}: manifest={meta['sha256'][:16]}… actual={h[:16]}…"
                )
    return manifest


def rebuild(
    archive_path: Path,
    target_data_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Restore an export into ``target_data_dir`` (verifying checksums first).

    Args:
        archive_path: An archive produced by :func:`export_all`.
        target_data_dir: Destination data dir. Must be empty/absent unless
            ``overwrite`` is set.
        overwrite: Allow restoring into a non-empty target.

    Returns:
        The verified manifest.
    """
    target = Path(target_data_dir)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise DurabilityError(
            f"target data dir {target} is not empty; pass overwrite=True to restore over it"
        )
    manifest = verify_archive(archive_path)  # fail before touching the target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for rel in manifest["files"]:
            arcname = _DATA_PREFIX + rel
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(arcname))
    return manifest
