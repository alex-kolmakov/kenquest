"""Full pipeline: reset → ingest → extract concepts → build graph.

Usage:
    uv run python scripts/build_topic.py ocean-conservation
    uv run python scripts/build_topic.py ocean-conservation --skip-ingest   # re-extract only
    uv run python scripts/build_topic.py ocean-conservation --skip-extract  # re-graph only
    uv run python scripts/build_topic.py ocean-conservation --keep-cache    # preserve llm_cache

Steps
-----
1. Reset  — clear concepts, edges, progress, raw_materials for the topic.
2. Ingest — fetch materials from wikipedia + wikibooks + openstax.
3. Extract — NLP candidate extraction + Wikipedia lookup + LLM batch fallback.
4. Graph  — infer prerequisite edges (LLM, tiered batches) + cycle resolution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from backend.config import settings
from backend.db.duckdb_client import _init_schema
from backend.extraction import build_graph, extract_concepts
from backend.ingestion.pipeline import _topic_query, run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_topic")


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------


def _ensure_topic(conn: duckdb.DuckDBPyConnection, topic_id: str) -> None:
    """Insert topic row if it doesn't exist yet."""
    name = topic_id.replace("-", " ").title()
    conn.execute(
        """
        INSERT INTO topics (id, name, status)
        VALUES (?, ?, 'pending')
        ON CONFLICT (id) DO NOTHING
        """,
        [topic_id, name],
    )


def reset_derived(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    *,
    clear_materials: bool = True,
    keep_cache: bool = False,
) -> None:
    """Delete all derived data for a topic, leaving topic row intact."""
    topic_query = _topic_query(topic_id)

    logger.info("Clearing quiz_sessions …")
    conn.execute(
        "DELETE FROM quiz_sessions WHERE concept_id IN "
        "(SELECT id FROM concepts WHERE topic_id = ?)",
        [topic_id],
    )
    logger.info("Clearing concept_progress …")
    conn.execute("DELETE FROM concept_progress WHERE topic_id = ?", [topic_id])

    logger.info("Clearing prerequisite_edges …")
    conn.execute(
        "DELETE FROM prerequisite_edges WHERE source_id IN "
        "(SELECT id FROM concepts WHERE topic_id = ?)",
        [topic_id],
    )
    logger.info("Clearing concepts …")
    conn.execute("DELETE FROM concepts WHERE topic_id = ?", [topic_id])

    conn.execute(
        "UPDATE topics SET status = 'pending', concept_count = 0 WHERE id = ?",
        [topic_id],
    )

    if clear_materials:
        logger.info("Clearing kenquest_raw.raw_materials …")
        try:
            conn.execute(
                "DELETE FROM kenquest_raw.raw_materials WHERE topic_id = ?",
                [topic_query],
            )
        except Exception as e:
            logger.warning("raw_materials clear skipped (schema may not exist): %s", e)

    if not keep_cache:
        logger.info("Clearing llm_cache …")
        conn.execute("DELETE FROM llm_cache")
    else:
        logger.info("Keeping llm_cache (--keep-cache).")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def _progress(label: str):
    def cb(msg: str):
        logger.info("[%s] %s", label, msg)

    return cb


def step_ingest(topic_id: str) -> int:
    logger.info("━━ STEP 1/3 — INGEST ━━")
    t0 = time.time()
    result = run_ingestion(topic_id, progress_callback=_progress("ingest"))
    total = result["total"]
    logger.info("Ingestion done in %.1fs — %d materials.", time.time() - t0, total)
    if total == 0:
        logger.error("No materials loaded — check network and source availability.")
        sys.exit(1)
    return total


def step_extract(conn: duckdb.DuckDBPyConnection, topic_id: str) -> int:
    logger.info("━━ STEP 2/3 — EXTRACT CONCEPTS ━━")
    t0 = time.time()
    concepts = extract_concepts(conn, topic_id, progress_callback=_progress("extract"))
    logger.info("Extraction done in %.1fs — %d concepts.", time.time() - t0, len(concepts))
    if not concepts:
        logger.error("No concepts extracted — check raw_materials content.")
        sys.exit(1)
    return len(concepts)


def step_graph(conn: duckdb.DuckDBPyConnection, topic_id: str) -> dict:
    logger.info("━━ STEP 3/3 — BUILD GRAPH ━━")
    t0 = time.time()
    result = build_graph(conn, topic_id, progress_callback=_progress("graph"))
    logger.info(
        "Graph done in %.1fs — %d edges added, %d cycle edges removed.",
        time.time() - t0,
        result["edges_added"],
        result["edges_removed_cycles"],
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Full KenQuest topic build pipeline.")
    parser.add_argument("topic_id", help="Topic ID in kebab-case, e.g. ocean-conservation")
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip ingestion (use existing materials)."
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Skip extraction (use existing concepts)."
    )
    parser.add_argument(
        "--keep-cache", action="store_true", help="Preserve llm_cache across reset."
    )
    args = parser.parse_args()

    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("Topic     : %s", args.topic_id)
    logger.info("DB        : %s", db_path)
    logger.info("Skip ingest  : %s", args.skip_ingest)
    logger.info("Skip extract : %s", args.skip_extract)
    logger.info("Keep cache   : %s", args.keep_cache)
    logger.info("=" * 64)

    wall = time.time()

    conn = duckdb.connect(str(db_path))
    _init_schema(conn)
    _ensure_topic(conn, args.topic_id)

    # Reset scope depends on what steps we're running
    reset_derived(
        conn,
        args.topic_id,
        clear_materials=not args.skip_ingest,
        keep_cache=args.keep_cache,
    )
    conn.close()

    # Step 1 — Ingest (runs its own DuckDB connection via DLT)
    if not args.skip_ingest:
        step_ingest(args.topic_id)
    else:
        logger.info("Skipping ingestion.")

    # Steps 2+3 share a single connection
    conn = duckdb.connect(str(db_path))

    n_concepts = 0
    if not args.skip_extract:
        n_concepts = step_extract(conn, args.topic_id)
    else:
        n_concepts = conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE topic_id = ?", [args.topic_id]
        ).fetchone()[0]
        logger.info("Skipping extraction — %d existing concepts.", n_concepts)

    graph_result = step_graph(conn, args.topic_id)
    conn.close()

    logger.info("=" * 64)
    logger.info("DONE in %.1fs", time.time() - wall)
    logger.info("  Concepts : %d", n_concepts)
    logger.info("  Edges    : %d", graph_result["edges_added"])
    logger.info("  Cycles   : %d removed", graph_result["edges_removed_cycles"])
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
