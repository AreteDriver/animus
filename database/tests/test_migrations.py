"""Tests for database schema definitions.

These tests do NOT require a running PostgreSQL instance.
They verify that migration scripts are syntactically valid and
that table definitions match expected shapes.
"""

import ast
from pathlib import Path

import pytest

MIGRATION_DIR = Path(__file__).parent.parent / "migrations" / "versions"


def _load_migration_tree():
    """Parse the initial migration file into an AST."""
    mig_file = next(MIGRATION_DIR.glob("001_*.py"))
    return ast.parse(mig_file.read_text())


def test_initial_migration_syntax():
    """The 001 migration must be valid Python with upgrade/downgrade."""
    tree = _load_migration_tree()
    funcs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("upgrade", "downgrade")
    }
    assert funcs == {"upgrade", "downgrade"}, "Migration must define upgrade() and downgrade()"


def test_initial_migration_creates_three_tables():
    """001_initial_schema must create object_registry, event_ledger, traceability."""
    tree = _load_migration_tree()
    create_table_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_table"
    ]
    table_names = {
        node.args[0].value  # use .value, not deprecated .s
        for node in create_table_calls
        if isinstance(node.args[0], ast.Constant)
    }
    assert table_names == {"object_registry", "event_ledger", "traceability"}, (
        f"Expected exactly 3 tables, got {table_names}"
    )


def test_object_registry_has_bitemporal_columns():
    """object_registry must contain valid_from, valid_to, recorded_at, superseded_at."""
    mig_file = next(MIGRATION_DIR.glob("001_*.py"))
    source = mig_file.read_text()
    for col in ("valid_from", "valid_to", "recorded_at", "superseded_at"):
        assert col in source, f"Column {col!r} missing from object_registry migration"


def test_event_ledger_is_append_only_no_update():
    """event_ledger has no update/delete triggers in initial migration."""
    mig_file = next(MIGRATION_DIR.glob("001_*.py"))
    source = mig_file.read_text()
    assert "event_ledger" in source
    # No UPDATE/DELETE triggers should exist yet (append-only table)
    assert "CREATE TRIGGER" not in source.upper()
