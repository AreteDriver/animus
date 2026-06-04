"""Durability — portable full-state export + tested cold-rebuild (roadmap A8).

Loss-of-machine survivability: ``export_all`` writes a single documented-schema
archive of the entire data directory (memory store, entities, tasks, scores,
logs, integrations) plus a redacted config snapshot and a ``manifest.json``
that inventories every file with its SHA-256. ``rebuild`` verifies the manifest
checksums and restores the state into a fresh data dir. A round-trip
(export → wipe → rebuild) is exercised in tests so the path is proven, not
assumed.
"""

from animus.durability.export import (
    ARCHIVE_SCHEMA_VERSION,
    DurabilityError,
    ManifestMismatchError,
    export_all,
    rebuild,
    verify_archive,
)

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DurabilityError",
    "ManifestMismatchError",
    "export_all",
    "rebuild",
    "verify_archive",
]
