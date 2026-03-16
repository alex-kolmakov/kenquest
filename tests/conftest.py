"""Shared test fixtures for KenQuest."""

from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Return a temporary DuckDB file path."""
    return tmp_path / "test.duckdb"


@pytest.fixture
def db_conn(temp_db_path: Path) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """In-memory DuckDB connection with schema applied."""
    conn = duckdb.connect(str(temp_db_path))
    schema_sql = (Path(__file__).parent.parent / "backend" / "db" / "schema.sql").read_text()
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    yield conn
    conn.close()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory with graph/ and chroma/ subdirs."""
    (tmp_path / "graph").mkdir()
    (tmp_path / "chroma").mkdir()
    return tmp_path
