"""DLT destination configuration — DuckDB."""

import dlt

from backend.config import settings


def make_duckdb_destination() -> dlt.destinations.duckdb:  # type: ignore[name-defined]
    """Return a DLT DuckDB destination pointed at the configured database path."""
    return dlt.destinations.duckdb(credentials=str(settings.database_path))
