"""Animus Identity — Self-knowledge and self-reference.

The identity layer gives Animus awareness of its own codebase,
capabilities, and version. This is the foundation for the bootstrap
loop: Animus must be able to read its own code to improve it.

Phase 1b threshold: identity files + memory persistence + file write
permissions on its own identity files.
"""

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
