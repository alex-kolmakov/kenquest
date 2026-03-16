"""DuckDB connection management."""

from collections.abc import Generator
from pathlib import Path

import duckdb

from backend.config import settings

_connection: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, initializing schema on first call."""
    global _connection
    if _connection is None:
        db_path = settings.database_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _connection = duckdb.connect(str(db_path))
        _init_schema(_connection)
    return _connection


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def db_dependency() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """FastAPI dependency that yields the DuckDB connection."""
    yield get_connection()
