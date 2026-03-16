"""FastAPI dependency injection."""

from typing import Annotated

import duckdb
from fastapi import Depends

from backend.db.duckdb_client import db_dependency

DB = Annotated[duckdb.DuckDBPyConnection, Depends(db_dependency)]
