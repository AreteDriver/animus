"""Tests for animus_forge.agent.identity_roots — the fail-closed guard."""

from pathlib import Path

from animus_forge.agent.identity_roots import IDENTITY_ROOTS, is_identity_path


class TestIdentityRootsStructure:
    def test_identity_roots_is_tuple(self):
        """Tuple — immutable; nobody can patch the list at runtime."""
        assert isinstance(IDENTITY_ROOTS, tuple)

    def test_identity_roots_non_empty(self):
        assert len(IDENTITY_ROOTS) > 0

    def test_identity_roots_includes_core_values(self):
        """CORE_VALUES.md is immutable per animus CLAUDE.md non-negotiables."""
        assert "packages/core/animus/CORE_VALUES.md" in IDENTITY_ROOTS

    def test_identity_roots_includes_identity_dir(self):
        assert "packages/core/animus/identity" in IDENTITY_ROOTS

    def test_identity_roots_includes_persona(self):
        assert "packages/bootstrap/src/animus_bootstrap/persona" in IDENTITY_ROOTS

    def test_identity_roots_all_relative(self):
        """Roots are repo-relative so the guard is portable across worktrees."""
        for rel in IDENTITY_ROOTS:
            assert not rel.startswith("/")
            assert not rel.startswith("~")


class TestIsIdentityPath:
    def _setup_project(self, tmp_path: Path) -> Path:
        """Create the minimal directory structure for identity guard tests."""
        for rel in IDENTITY_ROOTS:
            target = tmp_path / rel
            if rel.endswith(".md"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            else:
                target.mkdir(parents=True, exist_ok=True)
        return tmp_path

    def test_core_values_md_is_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        target = root / "packages/core/animus/CORE_VALUES.md"
        assert is_identity_path(target, root) is True

    def test_identity_dir_root_is_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        target = root / "packages/core/animus/identity"
        assert is_identity_path(target, root) is True

    def test_path_under_identity_dir_is_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        target = root / "packages/core/animus/identity/persona.yaml"
        assert is_identity_path(target, root) is True

    def test_persona_dir_path_is_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        target = root / "packages/bootstrap/src/animus_bootstrap/persona/voice.yaml"
        assert is_identity_path(target, root) is True

    def test_sibling_of_identity_not_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        target = root / "packages/core/animus/cognitive.py"
        assert is_identity_path(target, root) is False

    def test_readme_not_identity(self, tmp_path: Path):
        root = self._setup_project(tmp_path)
        readme = root / "README.md"
        readme.touch()
        assert is_identity_path(readme, root) is False

    def test_nonexistent_path_works(self, tmp_path: Path):
        """Path may not exist yet (writing a new file) — check still works."""
        root = self._setup_project(tmp_path)
        target = root / "packages/core/animus/identity/new_file.md"
        assert is_identity_path(target, root) is True

    def test_path_outside_project_root_not_identity(self, tmp_path: Path):
        """A path entirely outside the project root is not 'identity' by definition.

        The path validator catches escape attempts; the identity guard's job is
        narrower — only flag identity paths inside the project.
        """
        root = self._setup_project(tmp_path)
        outside = tmp_path.parent / "elsewhere.txt"
        assert is_identity_path(outside, root) is False

    def test_unresolvable_project_root_fails_closed(self, tmp_path: Path):
        """If the project root itself can't be resolved, deny (fail-closed)."""
        bogus_root = tmp_path / "does-not-exist"
        target = tmp_path / "anything.md"
        # bogus_root.resolve(strict=True) raises — guard returns True.
        assert is_identity_path(target, bogus_root) is True

    def test_symlink_into_identity_is_identity(self, tmp_path: Path):
        """A symlink pointing into an identity root is treated as identity."""
        root = self._setup_project(tmp_path)
        identity_file = root / "packages/core/animus/identity/real.yaml"
        identity_file.touch()
        link = root / "sneaky.yaml"
        link.symlink_to(identity_file)
        assert is_identity_path(link, root) is True
