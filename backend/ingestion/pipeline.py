"""Ingestion orchestrator — all sources run in parallel via DLT.

All sources are passed to a single pipeline.run() call. DLT's extract
stage runs each parallelized resource in its own thread concurrently:

  wikipedia  ──┐
  wikibooks  ──┤
  openstax   ──┴──► pipeline.run([...]) ──► kenquest_raw.raw_materials

Within each source, work is sequential. Wikipedia linked articles are fetched
in a loop due to DLT's constraint against multiple same-named resources in one
source. Inter-source parallelism is handled by parallelized=True per resource.

Sources dropped:
  - doab               : abstract-only — books are PDF-only on OAPEN, no HTML API
  - opentextbook       : description-only — inconsistent hosting (Pressbooks, university
                         sites, custom platforms), no single full-text API
  - arxiv              : abstracts only — too thin for curriculum use
  - roadmap.sh         : tech-only, wrong for domain-science topics
  - Semantic Scholar   : rate-limited (429) without an API key
  - GitHub general search : noise-to-signal ratio too high

Each source is optional — failures are logged and skipped without aborting.
Worker count is configured in .dlt/config.toml.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import dlt

from backend.config import settings
from backend.ingestion.sources.openstax import openstax_source
from backend.ingestion.sources.wikibooks import wikibooks_source
from backend.ingestion.sources.wikipedia import wikipedia_source

logger = logging.getLogger(__name__)

# Only allow safe characters in topic_id to prevent path traversal in DLT
# pipeline state files (.dlt/pipelines/kenquest_{topic_id}/)
_SAFE_TOPIC_ID = re.compile(r"[^a-zA-Z0-9_-]")


def _topic_query(topic_id: str) -> str:
    """Convert topic_id (kebab-case) to a human-readable query string."""
    return topic_id.replace("-", " ")


def run_ingestion(
    topic_id: str,
    *,
    progress_callback: Any | None = None,
) -> dict[str, int]:
    """Run the ingestion pipeline for *topic_id*, all sources in parallel.

    Returns {"total": rows_loaded}.
    """
    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    topic_query = _topic_query(topic_id)
    # Sanitise for DLT pipeline name (used as a filesystem directory)
    safe_id = _SAFE_TOPIC_ID.sub("", topic_id)

    if progress_callback:
        progress_callback("Fetching curriculum materials…")

    pipeline = dlt.pipeline(
        pipeline_name=f"kenquest_{safe_id}",
        destination=dlt.destinations.duckdb(credentials=str(db_path)),
        dataset_name="kenquest_raw",
    )

    sources = [
        wikipedia_source(topic=topic_query, max_linked=8),
        wikibooks_source(topic=topic_query, max_books=3, max_chapters_per_book=20),
        openstax_source(topic=topic_query),
    ]

    results: dict[str, int] = {"total": 0}
    try:
        pipeline.run(sources)
        total = count_materials(topic_id)
        results["total"] = total
        logger.info("Ingestion complete for '%s': %d rows loaded", topic_id, total)
    except Exception as e:
        logger.warning("Ingestion failed for '%s': %s", topic_id, e)

    if progress_callback:
        progress_callback("Ingestion complete")

    return results


def count_materials(topic_id: str) -> int:
    """Return number of raw_materials rows for a topic (for status reporting)."""
    import duckdb

    db_path = settings.database_path
    if not db_path.exists():
        return 0
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        result = conn.execute(
            "SELECT COUNT(*) FROM kenquest_raw.raw_materials WHERE topic_id = ?",
            [_topic_query(topic_id)],
        ).fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception:
        return 0
