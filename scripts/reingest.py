"""Reset and re-ingest all materials for a topic.

Usage:
    uv run python scripts/reingest.py ocean-conservation
    uv run python scripts/reingest.py ocean-conservation --keep-cache

What this does:
  1. Clears all derived data for the topic: concepts, edges, progress, sessions.
  2. Clears the raw_materials in the DLT destination schema.
  3. Optionally clears the llm_cache (default: yes, pass --keep-cache to skip).
  4. Re-runs the ingestion pipeline (wikipedia + wikibooks + openstax).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from backend.config import settings
from backend.ingestion.pipeline import _topic_query, run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reingest")


def reset_topic(topic_id: str, *, keep_cache: bool = False) -> None:
    topic_query = _topic_query(topic_id)
    db_path = settings.database_path

    if not db_path.exists():
        logger.info("Database does not exist yet — nothing to reset.")
        return

    conn = duckdb.connect(str(db_path))

    try:
        # Derived data (order matters — FK constraints)
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

        logger.info("Resetting topic status …")
        conn.execute(
            "UPDATE topics SET status = 'pending', concept_count = 0 WHERE id = ?",
            [topic_id],
        )

        # Raw materials in DLT destination schema
        logger.info("Clearing kenquest_raw.raw_materials …")
        try:
            conn.execute(
                "DELETE FROM kenquest_raw.raw_materials WHERE topic_id = ?",
                [topic_query],
            )
        except Exception as e:
            logger.warning("Could not clear raw_materials (schema may not exist yet): %s", e)

        if not keep_cache:
            logger.info("Clearing llm_cache …")
            conn.execute("DELETE FROM llm_cache")
        else:
            logger.info("Keeping llm_cache (--keep-cache set).")

    finally:
        conn.close()

    logger.info("Reset complete for topic '%s'.", topic_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset and re-ingest a topic.")
    parser.add_argument("topic_id", help="Topic ID (kebab-case), e.g. ocean-conservation")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not clear llm_cache (useful to avoid re-paying for prior LLM calls).",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only reset; do not run ingestion.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Topic    : %s", args.topic_id)
    logger.info("Keep cache: %s", args.keep_cache)
    logger.info("=" * 60)

    reset_topic(args.topic_id, keep_cache=args.keep_cache)

    if args.reset_only:
        logger.info("--reset-only set, skipping ingestion.")
        return

    logger.info("Starting ingestion pipeline …")
    t0 = time.time()

    def progress(msg: str) -> None:
        logger.info("  [pipeline] %s", msg)

    result = run_ingestion(args.topic_id, progress_callback=progress)
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Ingestion complete in %.1fs — %d materials loaded.", elapsed, result["total"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
