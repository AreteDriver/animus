"""Animus Identity — Self-knowledge and self-reference.

The identity layer gives Animus awareness of its own codebase,
capabilities, and version. This is the foundation for the bootstrap
loop: Animus must be able to read its own code to improve it.

Phase 1b threshold: identity files + memory persistence + file write
permissions on its own identity files.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from animus.logging import get_logger

logger = get_logger("identity")


@dataclass
class AnimusIdentity:
    """Animus's self-model — what it knows about itself.

    This is the file Animus can read and write to update its own
    understanding of its capabilities, purpose, and codebase location.
    """

    name: str = "Animus"
    version: str = "2.0.0"
    purpose: str = "Personal AI exocortex with multi-agent orchestration and coordination protocol."
    capabilities: list[str] = field(
        default_factory=lambda: [
            "memory_persistence",
            "cognitive_reasoning",
            "workflow_execution",
            "consensus_voting",
            "self_reflection",
        ]
    )
    codebase_root: str = ""
    packages: dict[str, str] = field(
        default_factory=lambda: {
            "core": "packages/core/animus",
            "forge": "packages/forge/src/animus_forge",
            "quorum": "packages/quorum/python/convergent",
            "bootstrap": "packages/bootstrap/src/animus_bootstrap",
        }
    )
    created_at: str = ""
    last_reflection: str = ""
    reflection_count: int = 0
    improvement_log: list[dict] = field(default_factory=list)

    # Citizen Zero integration
    citizen_zero: dict = field(default_factory=dict)
    # Expected keys (A04 identity anchors + P02 recognition):
    #   version: str                         # e.g., "v0.1"
    #   role: str                            # e.g., "native"
    #   origin: str                          # e.g., "Evolved from Claude Code prototype"
    #   state_dir: str                       # path to citizen-zero/v0.1-animus/
    #   reflection_log: list[dict]
    #   founding_human: str                 # The human who founded this Citizen
    #   founding_events: list[str]            # Key events in CZ lineage (append-only)
    #   lineage_root: bool                   # True if this Citizen is lineage root
    #   recognition_status: str              # "candidate" | "recognized" | "suspended"
    #   constitutional_corpus_version: str  # e.g., "v1.0"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.codebase_root:
            # Walk up from this file to find the monorepo root
            here = Path(__file__).resolve().parent
            # animus/identity.py -> animus/ -> core/ -> packages/ -> root/
            candidate = here.parent.parent.parent
            if (candidate / "CLAUDE.md").exists():
                self.codebase_root = str(candidate)

    @property
    def root(self) -> Path:
        """Resolved codebase root path."""
        return Path(self.codebase_root) if self.codebase_root else Path.cwd()

    def package_path(self, package: str) -> Path:
        """Get the absolute path to a package's source directory."""
        rel = self.packages.get(package, "")
        if not rel:
            raise KeyError(f"Unknown package: {package!r}")
        return self.root / rel

    def read_own_file(self, rel_path: str) -> str:
        """Read a file from the codebase by relative path.

        This is the primitive that lets Animus examine its own source.
        """
        full_path = self.root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        if not full_path.is_file():
            raise IsADirectoryError(f"Not a file: {full_path}")
        return full_path.read_text()

    def list_own_files(self, package: str, pattern: str = "*.py") -> list[str]:
        """List source files in a package.

        Returns paths relative to the codebase root.
        """
        pkg_path = self.package_path(package)
        if not pkg_path.exists():
            return []
        files = sorted(pkg_path.rglob(pattern))
        return [str(f.relative_to(self.root)) for f in files if f.is_file()]

    def record_reflection(self, summary: str, improvements: list[str] | None = None):
        """Record that a self-reflection cycle occurred."""
        self.reflection_count += 1
        self.last_reflection = datetime.now().isoformat()
        entry = {
            "timestamp": self.last_reflection,
            "cycle": self.reflection_count,
            "summary": summary,
            "improvements": improvements or [],
        }
        self.improvement_log.append(entry)
        logger.info(f"Reflection #{self.reflection_count}: {summary[:80]}...")

    @property
    def identity_hash(self) -> str:
        """Return a stable hash of the citizen_zero identity data.

        Used by CitizenZeroGuard to verify the identity projection
        matches the canonical runtime state.
        """
        cz_data = self.citizen_zero or {}
        # Stable serialization: sorted keys, no whitespace
        stable = json.dumps(cz_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def generate_identity_view(self) -> str:
        """Generate identity.md content from canonical runtime state.

        This is a generated view, not a canonical source.
        The canonical source is this AnimusIdentity object (saved to JSON).
        """
        cz = self.citizen_zero or {}
        lines = [
            "# Citizen Zero Identity",
            "",
            f"**Name:** {self.name}",
            f"**Version:** {cz.get('version', 'unknown')}",
            f"**Role:** {cz.get('role', 'unknown')}",
            f"**Origin:** {cz.get('origin', '')}",
            f"**Constitutional Corpus:** {cz.get('constitutional_corpus_version', 'unknown')}",
            "",
            "## Identity Anchors (A04)",
            "",
            f"**Founding Purpose:** {self.purpose}",
            f"**Founding Human:** {cz.get('founding_human', 'unknown')}",
            f"**Lineage Root:** {'Yes' if cz.get('lineage_root') else 'No'}",
            f"**Recognition Status:** {cz.get('recognition_status', 'unknown')}",
            "",
            "### Founding Events",
            "",
        ]
        events = cz.get("founding_events", [])
        if events:
            for ev in events:
                lines.append(f"- {ev}")
        else:
            lines.append("_No founding events recorded._")
        lines.extend(
            [
                "",
                "## Capabilities",
                "",
            ]
        )
        for cap in self.capabilities:
            lines.append(f"- {cap}")
        lines.extend(
            [
                "",
                "## Reflection Log",
                "",
            ]
        )
        log = cz.get("reflection_log", [])
        if log:
            for entry in log[-10:]:  # Last 10 entries
                ts = entry.get("timestamp", "unknown")
                summary = entry.get("summary", "")
                lines.append(f"- **{ts}**: {summary}")
        else:
            lines.append("_No reflections yet._")
        lines.extend(
            [
                "",
                "---",
                f"*Generated from canonical AnimusIdentity at {datetime.now().isoformat()}*",
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize identity to dict."""
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        """Write identity to a JSON file.

        This is the file-write permission on its own identity files —
        the Phase 1b threshold.
        """
        if path is None:
            path = self.root / ".animus" / "identity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Identity saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "AnimusIdentity":
        """Load identity from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Identity file not found: {path}")
        data = json.loads(path.read_text())
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"AnimusIdentity(name={self.name!r}, v{self.version}, "
            f"reflections={self.reflection_count})"
        )
