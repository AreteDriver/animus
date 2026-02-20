"""Tests for database backends."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import (
    PostgresBackend,
    SQLiteBackend,
    create_backend,
)


class TestSQLiteBackend:
    """Tests for SQLiteBackend class."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            yield backend
            backend.close()

    def test_execute_create_table(self, backend):
        """Can create tables."""
        backend.executescript("""
            CREATE TABLE test (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        # Should not raise
        backend.execute("INSERT INTO test (name) VALUES (?)", ("test",))

    def test_fetchone(self, backend):
        """Can fetch one row."""
        backend.executescript("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        backend.execute("INSERT INTO test (name) VALUES (?)", ("alice",))

        row = backend.fetchone("SELECT * FROM test WHERE name = ?", ("alice",))
        assert row is not None
        assert row["name"] == "alice"

    def test_fetchone_no_result(self, backend):
        """Fetchone returns None when no match."""
        backend.executescript("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")

        row = backend.fetchone("SELECT * FROM test WHERE name = ?", ("missing",))
        assert row is None

    def test_fetchall(self, backend):
        """Can fetch all rows."""
        backend.executescript("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        backend.execute("INSERT INTO test (name) VALUES (?)", ("alice",))
        backend.execute("INSERT INTO test (name) VALUES (?)", ("bob",))

        rows = backend.fetchall("SELECT * FROM test ORDER BY name")
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[1]["name"] == "bob"

    def test_transaction_commit(self, backend):
        """Transaction commits on success."""
        backend.executescript("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")

        with backend.transaction():
            backend.execute("INSERT INTO test (name) VALUES (?)", ("test",))

        row = backend.fetchone("SELECT * FROM test")
        assert row is not None

    def test_transaction_rollback(self, backend):
        """Transaction rolls back on error."""
        backend.executescript("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")

        with pytest.raises(ValueError):
            with backend.transaction():
                backend.execute("INSERT INTO test (name) VALUES (?)", ("test",))
                raise ValueError("Simulated error")

        row = backend.fetchone("SELECT * FROM test")
        assert row is None

    def test_placeholder(self, backend):
        """SQLite uses ? placeholder."""
        assert backend.placeholder == "?"

    def test_adapt_query(self, backend):
        """SQLite doesn't change query placeholders."""
        query = "SELECT * FROM test WHERE id = ?"
        assert backend.adapt_query(query) == query


class TestCreateBackend:
    """Tests for create_backend function."""

    def test_sqlite_url(self):
        """Can create SQLite backend from URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = create_backend(f"sqlite:///{db_path}")
            assert isinstance(backend, SQLiteBackend)
            backend.close()

    def test_sqlite_default(self):
        """Creates SQLite backend by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = create_backend(db_path=db_path)
            assert isinstance(backend, SQLiteBackend)
            backend.close()

    def test_unknown_scheme_raises(self):
        """Unknown URL scheme raises ValueError."""
        with pytest.raises(ValueError) as exc:
            create_backend("mysql://localhost/test")
        assert "Unsupported" in str(exc.value)

    def test_postgres_url_requires_psycopg2(self):
        """PostgreSQL URL requires psycopg2."""
        # This will either work (if psycopg2 installed) or raise ImportError
        try:
            backend = create_backend("postgresql://user:pass@localhost/test")
            backend.close()
        except ImportError as e:
            assert "psycopg2" in str(e)

    @patch("animus_forge.state.backends.PostgresBackend")
    def test_postgres_url(self, mock_pg):
        """create_backend with postgresql:// returns PostgresBackend."""
        mock_pg.return_value = MagicMock()
        create_backend("postgresql://user:pass@localhost/testdb")
        mock_pg.assert_called_once_with(connection_string="postgresql://user:pass@localhost/testdb")

    @patch("animus_forge.state.backends.PostgresBackend")
    def test_postgres_scheme(self, mock_pg):
        """create_backend with postgres:// also works."""
        mock_pg.return_value = MagicMock()
        create_backend("postgres://user:pass@localhost/testdb")
        mock_pg.assert_called_once()

    def test_sqlite_single_slash(self):
        """create_backend with sqlite:/path strips leading slash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            # Single slash: sqlite:/path -> path without leading /
            backend = create_backend(f"sqlite:/{db_path}")
            assert isinstance(backend, SQLiteBackend)
            backend.close()


class TestPostgresBackend:
    """Tests for PostgresBackend with mocked psycopg2."""

    @pytest.fixture
    def mock_psycopg2(self):
        """Mock psycopg2 module."""
        mock_module = MagicMock()
        mock_module.extras.RealDictCursor = MagicMock()
        return mock_module

    @pytest.fixture
    def backend(self, mock_psycopg2):
        """Create a PostgresBackend with mocked psycopg2."""
        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
        ):
            b = PostgresBackend(
                host="dbhost",
                port=5433,
                database="mydb",
                user="myuser",
                password="secret",
            )
        return b

    def test_init_explicit_params(self, mock_psycopg2):
        """Init with explicit params stores them."""
        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
        ):
            b = PostgresBackend(host="h", port=1234, database="d", user="u", password="p")
        assert b._conn_params == {
            "host": "h",
            "port": 1234,
            "database": "d",
            "user": "u",
            "password": "p",
        }

    def test_init_connection_string(self, mock_psycopg2):
        """Init with connection_string parses URL."""
        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
        ):
            b = PostgresBackend(connection_string="postgresql://alice:pw@dbhost:9999/mydb")
        assert b._conn_params["host"] == "dbhost"
        assert b._conn_params["port"] == 9999
        assert b._conn_params["database"] == "mydb"
        assert b._conn_params["user"] == "alice"
        assert b._conn_params["password"] == "pw"

    def test_init_import_error(self):
        """Raises ImportError when psycopg2 not available."""
        with patch.dict("sys.modules", {"psycopg2": None}):
            with pytest.raises(ImportError, match="psycopg2"):
                PostgresBackend()

    def test_placeholder(self, backend):
        """PostgreSQL uses %s placeholder."""
        assert backend.placeholder == "%s"

    def test_adapt_query(self, backend):
        """Replaces ? with %s."""
        assert (
            backend.adapt_query("SELECT * FROM t WHERE id = ?") == "SELECT * FROM t WHERE id = %s"
        )

    def test_execute(self, backend):
        """Execute creates cursor with RealDictCursor and adapts query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        result = backend.execute("SELECT * FROM t WHERE id = ?", (1,))
        mock_conn.cursor.assert_called_once_with(cursor_factory=backend._RealDictCursor)
        mock_cursor.execute.assert_called_once_with("SELECT * FROM t WHERE id = %s", (1,))
        assert result is mock_cursor

    def test_executemany(self, backend):
        """Executemany creates cursor and calls executemany."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        backend.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        mock_cursor.executemany.assert_called_once_with("INSERT INTO t VALUES (%s)", [(1,), (2,)])

    def test_executescript(self, backend):
        """Executescript splits and executes statements."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        script = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);\n"
        backend.executescript(script)
        assert mock_cursor.execute.call_count == 2

    def test_executescript_strips_comments(self, backend):
        """Executescript removes SQL comments."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        script = "-- comment\nCREATE TABLE a (id INT);\n"
        backend.executescript(script)
        assert mock_cursor.execute.call_count == 1

    def test_adapt_schema(self, backend):
        """Replaces AUTOINCREMENT with SERIAL."""
        result = backend._adapt_schema("id INTEGER PRIMARY KEY AUTOINCREMENT")
        assert result == "id SERIAL PRIMARY KEY"

    def test_fetchone_returns_dict(self, backend):
        """Fetchone returns dict when row exists."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "test"}
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        result = backend.fetchone("SELECT * FROM t WHERE id = ?", (1,))
        assert result == {"id": 1, "name": "test"}

    def test_fetchone_returns_none(self, backend):
        """Fetchone returns None when no row."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        result = backend.fetchone("SELECT * FROM t WHERE id = ?", (99,))
        assert result is None

    def test_fetchall(self, backend):
        """Fetchall returns list of dicts."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1}, {"id": 2}]
        mock_conn.cursor.return_value = mock_cursor
        backend._local.conn = mock_conn

        result = backend.fetchall("SELECT * FROM t")
        assert result == [{"id": 1}, {"id": 2}]

    def test_transaction_commits(self, backend):
        """Transaction commits on success."""
        mock_conn = MagicMock()
        backend._local.conn = mock_conn

        with backend.transaction():
            pass

        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()

    def test_transaction_rollback(self, backend):
        """Transaction rolls back on error."""
        mock_conn = MagicMock()
        backend._local.conn = mock_conn

        with pytest.raises(ValueError):
            with backend.transaction():
                raise ValueError("boom")

        mock_conn.rollback.assert_called_once()

    def test_close(self, backend):
        """Close closes connection."""
        mock_conn = MagicMock()
        backend._local.conn = mock_conn

        backend.close()
        mock_conn.close.assert_called_once()
        assert backend._local.conn is None

    def test_close_noop(self, backend):
        """Close is no-op when no connection."""
        # No conn set - should not raise
        backend.close()

    def test_get_conn_reuses(self, backend):
        """_get_conn reuses existing connection."""
        mock_conn = MagicMock()
        backend._local.conn = mock_conn

        assert backend._get_conn() is mock_conn
        backend._psycopg2.connect.assert_not_called()

    def test_get_conn_creates(self, backend):
        """_get_conn creates new connection when none exists."""
        mock_new = MagicMock()
        backend._psycopg2.connect.return_value = mock_new

        conn = backend._get_conn()
        assert conn is mock_new
        backend._psycopg2.connect.assert_called_once_with(**backend._conn_params)
