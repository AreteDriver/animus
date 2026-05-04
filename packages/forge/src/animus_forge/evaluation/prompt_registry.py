"""Prompt registry: name + version-pinned prompts on disk.

Treats prompts as diffable artifacts rather than inline strings. Each
prompt lives at ``packages/forge/prompts/<name>/v<N>.md`` with YAML
frontmatter; the registry reads the directory and resolves ``<name>@<version>``
to a ``PromptVersion`` object with a content hash suitable for
``eval_runs.prompt_version`` tagging.

Convention:
    packages/forge/prompts/
        <name>/
            v1.md         ← first version
            v2.md         ← next iteration
            CHANGELOG.md  ← optional, free-form

Frontmatter schema (all optional except name+version):
    ---
    name: supervisor
    version: v1
    parent: null           # or "v0" etc — for lineage
    created_at: 2026-04-21
    description: ...
    tags: [role:supervisor]
    ---
    <prompt body>

Use "v<N>" (integer suffix) for ordering. Mixing semver works but the
sort will be lexicographic within non-numeric tails. Integer-suffix is
recommended for the ergonomic "latest" resolver.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

# Names that hold reference/meta docs, not versioned prompts
RESERVED_NAMES = frozenset({"modes"})

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_VERSION_FILE_RE = re.compile(r"^v(\d+)(?:\..*)?$")


@dataclass
class PromptVersion:
    """A single version of a named prompt."""

    name: str
    version: str
    content: str
    content_hash: str
    created_at: str | None = None
    parent: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    path: str | None = None

    @property
    def identifier(self) -> str:
        """Canonical identifier used for eval run tagging."""
        return f"{self.name}@{self.version}"


def _normalize_content(content: str) -> str:
    """Canonical form for hashing — strip trailing whitespace, single trailing newline."""
    return content.rstrip() + "\n"


def _hash_content(content: str) -> str:
    return hashlib.sha256(_normalize_content(content).encode()).hexdigest()[:16]


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    try:
        import yaml

        data = yaml.safe_load(match.group(1)) or {}
    except ImportError:
        data = {}
    except Exception:
        data = {}

    return data, match.group(2)


def _version_sort_key(version: str) -> tuple[int, int, str]:
    """Sort key: integer suffix first, then lexicographic tail.

    "v1" < "v2" < "v10" — the integer dominates so "v10" comes after
    "v9", not between "v1" and "v2". Non-"v<N>" versions sort after.
    """
    match = _VERSION_FILE_RE.match(version)
    if match:
        return (0, int(match.group(1)), version)
    return (1, 0, version)


class PromptRegistry:
    """Read-only registry over a prompts directory.

    For write operations (new versions), use ``register()``.
    """

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_prompts(self) -> list[dict[str, Any]]:
        """Return summary per prompt: name, versions, latest, count."""
        if not self.prompts_dir.exists():
            return []

        summaries = []
        for entry in sorted(self.prompts_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in RESERVED_NAMES:
                continue
            versions = self.versions(entry.name)
            if not versions:
                continue
            summaries.append(
                {
                    "name": entry.name,
                    "versions": versions,
                    "latest": versions[-1],
                    "count": len(versions),
                }
            )
        return summaries

    def versions(self, name: str) -> list[str]:
        """Return sorted version list for a prompt (earliest → latest)."""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.is_dir():
            return []
        versions = []
        for path in prompt_dir.glob("v*.md"):
            version = path.stem
            if _VERSION_FILE_RE.match(version):
                versions.append(version)
        return sorted(versions, key=_version_sort_key)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, name: str, version: str = "latest") -> PromptVersion:
        """Load a prompt by name and version.

        ``version`` accepts:
            - explicit version string ("v1", "v2.1", ...)
            - "latest" — resolves to highest version by sort order
        """
        available = self.versions(name)
        if not available:
            raise FileNotFoundError(
                f"No versioned prompts found for '{name}' in {self.prompts_dir}"
            )

        if version == "latest":
            version = available[-1]
        elif version not in available:
            raise FileNotFoundError(
                f"Version '{version}' not found for prompt '{name}'. Available: {available}"
            )

        path = self.prompts_dir / name / f"{version}.md"
        text = path.read_text()
        metadata, body = _parse_frontmatter(text)

        return PromptVersion(
            name=metadata.get("name", name),
            version=metadata.get("version", version),
            content=body,
            content_hash=_hash_content(body),
            created_at=str(metadata["created_at"]) if metadata.get("created_at") else None,
            parent=metadata.get("parent"),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []) or [],
            path=str(path),
        )

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, name: str, v1: str, v2: str) -> str:
        """Unified diff between two versions of a prompt."""
        a = self.load(name, v1)
        b = self.load(name, v2)
        return "".join(
            difflib.unified_diff(
                a.content.splitlines(keepends=True),
                b.content.splitlines(keepends=True),
                fromfile=f"{name}@{a.version}",
                tofile=f"{name}@{b.version}",
            )
        )

    # ------------------------------------------------------------------
    # Register (write)
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        content: str,
        *,
        version: str = "auto",
        parent: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> PromptVersion:
        """Write a new version of a prompt to disk.

        With ``version="auto"``, the next v<N+1> is allocated. Caller can
        pin an explicit version but will get FileExistsError if taken.
        """
        prompt_dir = self.prompts_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)

        existing = self.versions(name)
        if version == "auto":
            next_n = 1
            if existing:
                last = existing[-1]
                match = _VERSION_FILE_RE.match(last)
                if match:
                    next_n = int(match.group(1)) + 1
            version = f"v{next_n}"

        path = prompt_dir / f"{version}.md"
        if path.exists():
            raise FileExistsError(
                f"Prompt {name}@{version} already exists at {path}. "
                f"Use version='auto' to append a new version."
            )

        parent = parent or (existing[-1] if existing else None)
        tags = tags or []

        frontmatter_lines = [
            "---",
            f"name: {name}",
            f"version: {version}",
            f"parent: {parent if parent else 'null'}",
            f"created_at: {date.today().isoformat()}",
        ]
        if description:
            frontmatter_lines.append(f"description: {description}")
        if tags:
            tag_list = ", ".join(tags)
            frontmatter_lines.append(f"tags: [{tag_list}]")
        frontmatter_lines.append("---")
        frontmatter_lines.append("")

        full_text = "\n".join(frontmatter_lines) + content.rstrip() + "\n"
        path.write_text(full_text)

        return PromptVersion(
            name=name,
            version=version,
            content=content,
            content_hash=_hash_content(content),
            created_at=date.today().isoformat(),
            parent=parent,
            description=description,
            tags=list(tags),
            path=str(path),
        )
