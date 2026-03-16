"""Concept extraction from raw materials.

Pipeline:
  1. Load raw_materials rows for a topic from DuckDB (via DLT destination table)
  2. Chunk each document into ~800-token sections (paragraph-aware)
  3. Call LiteLLM → structured JSON: [{name, description, difficulty, source_refs}]
  4. Two-pass deduplication:
       a. Exact slug match (same normalised name → keep first)
       b. Embedding cosine similarity > DEDUP_THRESHOLD → keep first encountered
  5. Write deduplicated concepts to the `concepts` table
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import duckdb
import litellm
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.extraction.prompts import CONCEPT_EXTRACTION_SYSTEM, CONCEPT_EXTRACTION_USER
from backend.utils import parse_llm_json_list, safe_temperature, slugify

logger = logging.getLogger(__name__)

# Approximate token budget per chunk (1 token ≈ 4 chars for English)
_CHUNK_CHARS = 3200  # ~800 tokens
_MIN_CHUNK_CHARS = 200
_DEDUP_THRESHOLD = 0.88  # cosine similarity above this → duplicate

# Minimum/maximum concept name length (in words)
_MIN_WORDS = 2
_MAX_WORDS = 6


def _word_count(name: str) -> int:
    return len(name.strip().split())


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_text(text: str) -> list[str]:
    """Split text into paragraph-aware chunks of ~_CHUNK_CHARS characters."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > _CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c) >= _MIN_CHUNK_CHARS]


# ---------------------------------------------------------------------------
# LLM call with caching
# ---------------------------------------------------------------------------


def _content_hash(text: str, prompt: str) -> str:
    return hashlib.sha256(f"{prompt}|||{text}".encode()).hexdigest()


def _llm_extract(
    conn: duckdb.DuckDBPyConnection,
    chunk: str,
    source_id: str,
    topic: str,
    max_concepts: int,
) -> list[dict[str, Any]]:
    """Call LLM to extract concepts from a text chunk.  Results are cached in llm_cache."""
    user_msg = CONCEPT_EXTRACTION_USER.format(
        source_id=source_id,
        topic=topic,
        text=chunk,
        max_concepts=max_concepts,
    )
    cache_key = _content_hash(chunk, user_msg)

    # Cache hit
    row = conn.execute(
        "SELECT response_json FROM llm_cache WHERE content_hash = ?", [cache_key]
    ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            pass

    # LLM call
    try:
        response = litellm.completion(  # type: ignore[operator]
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            messages=[
                {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=safe_temperature(0.2),
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
    except Exception as e:
        logger.warning("LLM extraction failed for source %s: %s", source_id, e)
        return []

    # Parse — LLM may return {"concepts": [...]} or bare [...]
    parsed = parse_llm_json_list(raw, "concepts")
    if not parsed:
        logger.warning("LLM returned non-JSON or empty list for source %s", source_id)
        return []

    # Store in cache
    conn.execute(
        "INSERT OR IGNORE INTO llm_cache (content_hash, prompt_hash, response_json) VALUES (?, ?, ?)",
        [cache_key, cache_key, json.dumps(parsed)],
    )
    return parsed


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup(
    candidates: list[dict[str, Any]],
    model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """Remove duplicates by slug then by embedding cosine similarity."""
    import numpy as np

    seen_slugs: set[str] = set()
    unique: list[dict[str, Any]] = []

    # Pass 1: exact slug dedup
    for c in candidates:
        slug = slugify(c["name"])
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            c["id"] = slug
            unique.append(c)

    if len(unique) <= 1:
        return unique

    # Pass 2: embedding cosine similarity dedup
    names = [c["name"] for c in unique]
    embeddings = model.encode(names, normalize_embeddings=True)  # shape (N, D)
    keep = [True] * len(unique)

    for i in range(len(unique)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(unique)):
            if not keep[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= _DEDUP_THRESHOLD:
                keep[j] = False  # drop the later duplicate

    return [c for c, k in zip(unique, keep, strict=True) if k]


# ---------------------------------------------------------------------------
# Write to DuckDB
# ---------------------------------------------------------------------------


def _write_concepts(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    concepts: list[dict[str, Any]],
) -> None:
    """Upsert concepts into the concepts table."""
    for c in concepts:
        source_refs = c.get("source_refs") or []
        conn.execute(
            """
            INSERT INTO concepts (id, topic_id, name, description, difficulty, source_refs)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                description = excluded.description,
                difficulty  = excluded.difficulty,
                source_refs = excluded.source_refs
            """,
            [
                c["id"],
                topic_id,
                c["name"],
                c.get("description", ""),
                max(1, min(5, int(c.get("difficulty", 3)))),
                source_refs,
            ],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_concepts(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    *,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract, deduplicate, and persist concepts for *topic_id*.

    Returns the final list of concept dicts written to DuckDB.
    """
    topic_query = topic_id.replace("-", " ")
    max_concepts_per_chunk = settings.max_concepts_per_chunk

    # Load ingested materials from the DLT destination (kenquest_raw schema)
    try:
        materials = conn.execute(
            """
            SELECT id, title, content
            FROM kenquest_raw.raw_materials
            WHERE topic_id = ?
              AND content IS NOT NULL
              AND LENGTH(content) > 100
            ORDER BY LENGTH(content) DESC
            """,
            [topic_query],  # DLT sources store topic_id as the space-separated query string
        ).fetchall()
    except Exception as e:
        logger.error("Failed to load raw_materials for topic '%s': %s", topic_id, e)
        return []

    if not materials:
        logger.warning("No raw materials found for topic '%s'", topic_id)
        return []

    logger.info("Extracting concepts from %d materials for topic '%s'", len(materials), topic_id)

    model = SentenceTransformer(settings.embedding_model)
    all_candidates: list[dict[str, Any]] = []

    for mat_idx, (mat_id, mat_title, content) in enumerate(materials, 1):
        chunks = _chunk_text(content)
        label = mat_title or mat_id
        logger.info(
            "[%d/%d] Processing '%s' → %d chunks", mat_idx, len(materials), label, len(chunks)
        )
        if progress_callback:
            progress_callback(f"[{mat_idx}/{len(materials)}] {label} ({len(chunks)} chunks)")

        mat_candidates: list[dict[str, Any]] = []
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.debug("  chunk %d/%d (%d chars)", chunk_idx, len(chunks), len(chunk))
            raw = _llm_extract(conn, chunk, mat_id, topic_query, max_concepts_per_chunk)
            valid = [
                c
                for c in raw
                if isinstance(c.get("name"), str)
                and _MIN_WORDS <= _word_count(c["name"]) <= _MAX_WORDS
            ]
            logger.info(
                "  chunk %d/%d → %d raw, %d valid concepts",
                chunk_idx,
                len(chunks),
                len(raw),
                len(valid),
            )
            mat_candidates.extend(valid)

        all_candidates.extend(mat_candidates)
        logger.info(
            "  material total: %d candidates (running total: %d)",
            len(mat_candidates),
            len(all_candidates),
        )

        # Write incrementally after each material so partial results survive a kill
        if mat_candidates:
            per_mat_deduped = _dedup(mat_candidates, model)
            _write_concepts(conn, topic_id, per_mat_deduped)
            logger.info("  wrote %d deduped concepts for '%s'", len(per_mat_deduped), label)

    # Final global dedup across everything written
    all_rows = conn.execute(
        "SELECT id, name, description, difficulty, source_refs FROM concepts WHERE topic_id = ?",
        [topic_id],
    ).fetchall()
    all_so_far = [
        {
            "id": r[0],
            "name": r[1],
            "description": r[2],
            "difficulty": r[3],
            "source_refs": r[4] or [],
        }
        for r in all_rows
    ]
    deduplicated = _dedup(all_so_far, model)

    # Delete embedding-duplicates that were merged out (no hard cap — quality is the only gate)
    kept_ids = {c["id"] for c in deduplicated}
    removed_ids = list({r[0] for r in all_rows} - kept_ids)
    if removed_ids:
        placeholders = ", ".join("?" * len(removed_ids))
        conn.execute(f"DELETE FROM concepts WHERE id IN ({placeholders})", removed_ids)  # noqa: S608

    # Update topic concept_count
    conn.execute(
        "UPDATE topics SET concept_count = ?, updated_at = NOW() WHERE id = ?",
        [len(deduplicated), topic_id],
    )

    logger.info("Extracted %d unique concepts for topic '%s'", len(deduplicated), topic_id)
    return deduplicated
