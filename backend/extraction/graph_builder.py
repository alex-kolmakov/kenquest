"""Prerequisite graph construction from extracted concepts.

Pipeline:
  1. Load concepts for a topic from DuckDB
  2. LiteLLM call → list of prerequisite edges [{source, target, strength, rationale}]
  3. Match LLM edge names → concept IDs (slug-based fuzzy match)
  4. Cycle detection via DFS on DuckDB edge list
  5. Cycle resolution: LLM-guided (2 attempts) → hard-break weakest edge (3rd attempt)
  6. Persist edges to prerequisite_edges table
  7. Set root concepts (no prerequisites) to UNLOCKED in concept_progress
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import duckdb
import litellm

from backend.config import settings
from backend.extraction.prompts import (
    CYCLE_RESOLUTION_SYSTEM,
    CYCLE_RESOLUTION_USER,
    GRAPH_BUILDER_SYSTEM,
    GRAPH_BUILDER_USER,
)
from backend.graph.queries import (
    detect_cycles,
    fetch_cycle_edges,
    root_concepts,
    weakest_edge_in_cycle,
)
from backend.graph.store import delete_edge, save_edges_bulk
from backend.models.concept import MasteryStatus
from backend.utils import parse_llm_json_list, safe_temperature, slugify

logger = logging.getLogger(__name__)

_MAX_CYCLE_RESOLUTION_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Name → concept ID matching
# ---------------------------------------------------------------------------


def _build_name_index(concepts: list[dict[str, Any]]) -> dict[str, str]:
    """Map slug variants of a concept name → concept id."""
    index: dict[str, str] = {}
    for c in concepts:
        index[c["id"]] = c["id"]
        index[slugify(c["name"])] = c["id"]
        # Also index individual significant words for fallback
        for word in c["name"].lower().split():
            if len(word) > 4 and word not in index:
                index[word] = c["id"]
    return index


def _resolve_concept_id(name: str, index: dict[str, str]) -> str | None:
    slug = slugify(name)
    if slug in index:
        return index[slug]
    # Partial match: slug is a prefix or suffix of an indexed key
    for key, cid in index.items():
        if slug in key or key in slug:
            return cid
    return None


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def _llm_infer_edges(
    topic: str,
    concepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ask LLM to infer prerequisite edges between concepts."""
    concepts_list = "\n".join(
        f"- {c['name']} (difficulty {c.get('difficulty', 3)}): {c.get('description', '')[:100]}"
        for c in concepts
    )
    user_msg = GRAPH_BUILDER_USER.format(topic=topic, concepts_list=concepts_list)

    try:
        response = litellm.completion(  # type: ignore[operator]
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            messages=[
                {"role": "system", "content": GRAPH_BUILDER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=safe_temperature(0.1),
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
    except Exception as e:
        logger.warning("LLM graph building failed: %s", e)
        return []

    return parse_llm_json_list(raw, "edges")


def _llm_resolve_cycle(
    conn: duckdb.DuckDBPyConnection,
    topic: str,
    topic_id: str,
    cycle: list[str],
) -> tuple[str, str] | None:
    """Ask LLM which edge to remove to break a cycle. Returns (source_id, target_id) or None."""
    # Fetch node names and edge attributes from DuckDB
    placeholders = ", ".join(["?"] * len(cycle))
    node_rows = conn.execute(
        f"SELECT id, name FROM concepts WHERE id IN ({placeholders})", cycle
    ).fetchall()
    node_names = dict(node_rows)

    cycle_pairs = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    edge_attrs = fetch_cycle_edges(conn, cycle)

    cycle_edges = []
    for src, tgt in cycle_pairs:
        if (src, tgt) in edge_attrs:
            attrs = edge_attrs[(src, tgt)]
            cycle_edges.append(
                f"  {node_names.get(src, src)} → {node_names.get(tgt, tgt)}"
                f" (strength={attrs['strength']:.2f},"
                f" rationale='{attrs['rationale']}')"
            )

    cycle_node_names = " → ".join(node_names.get(n, n) for n in cycle)

    user_msg = CYCLE_RESOLUTION_USER.format(
        topic=topic,
        cycle_nodes=cycle_node_names,
        cycle_edges="\n".join(cycle_edges),
    )

    try:
        response = litellm.completion(  # type: ignore[operator]
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            messages=[
                {"role": "system", "content": CYCLE_RESOLUTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=safe_temperature(0),
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        src_name = parsed.get("remove_source", "")
        tgt_name = parsed.get("remove_target", "")

        # Map names back to IDs
        name_to_id = {v: k for k, v in node_names.items()}
        src_id = name_to_id.get(src_name)
        tgt_id = name_to_id.get(tgt_name)

        if src_id and tgt_id and (src_id, tgt_id) in edge_attrs:
            return src_id, tgt_id
    except Exception as e:
        logger.warning("LLM cycle resolution failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Cycle resolution
# ---------------------------------------------------------------------------


def _resolve_cycles(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    topic: str,
) -> int:
    """Detect and resolve all cycles. Returns number of edges removed."""
    removed = 0
    for attempt in range(_MAX_CYCLE_RESOLUTION_ATTEMPTS * 10):  # outer safety limit
        cycles = detect_cycles(conn, topic_id)
        if not cycles:
            break

        cycle = cycles[0]
        logger.info("Cycle detected (attempt %d): %s", attempt + 1, " → ".join(cycle))

        if attempt < _MAX_CYCLE_RESOLUTION_ATTEMPTS - 1:
            edge = _llm_resolve_cycle(conn, topic, topic_id, cycle)
        else:
            edge = None  # force hard-break

        if edge is None:
            edge = weakest_edge_in_cycle(conn, cycle)
            logger.info("Hard-breaking weakest edge: %s → %s", *edge)

        src, tgt = edge
        delete_edge(conn, src, tgt)
        logger.info("Removed edge %s → %s to break cycle", src, tgt)
        removed += 1

    return removed


# ---------------------------------------------------------------------------
# Progress initialisation (root concepts → UNLOCKED)
# ---------------------------------------------------------------------------


def _init_progress(conn: duckdb.DuckDBPyConnection, topic_id: str) -> None:
    """Set root concepts (no prerequisites) to UNLOCKED in concept_progress."""
    roots = set(root_concepts(conn, topic_id))
    all_ids = [
        r[0]
        for r in conn.execute("SELECT id FROM concepts WHERE topic_id = ?", [topic_id]).fetchall()
    ]

    # Batch-insert roots (overwrite status if re-running) and non-roots (skip if exists)
    root_rows = [(topic_id, nid, MasteryStatus.UNLOCKED) for nid in all_ids if nid in roots]
    locked_rows = [(topic_id, nid, MasteryStatus.LOCKED) for nid in all_ids if nid not in roots]

    if root_rows:
        conn.executemany(
            """
            INSERT INTO concept_progress (topic_id, concept_id, status)
            VALUES (?, ?, ?)
            ON CONFLICT (topic_id, concept_id) DO UPDATE SET status = excluded.status
            """,
            root_rows,
        )
    if locked_rows:
        conn.executemany(
            """
            INSERT INTO concept_progress (topic_id, concept_id, status)
            VALUES (?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            locked_rows,
        )

    logger.info(
        "Initialised progress for topic '%s': %d roots unlocked, %d locked",
        topic_id,
        len(roots),
        len(all_ids) - len(roots),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_BATCH_SIZE = 40  # max concepts per LLM graph-building call
_MAX_LOWER_CONTEXT = 40  # max lower-tier concepts sent as prerequisite context per batch


def _resolve_edges(
    raw_edges: list[dict[str, Any]],
    name_index: dict[str, str],
) -> list[dict[str, Any]]:
    """Map LLM concept name strings → stable IDs, drop unresolvable pairs."""
    resolved = []
    for edge in raw_edges:
        src_id = _resolve_concept_id(edge.get("source", ""), name_index)
        tgt_id = _resolve_concept_id(edge.get("target", ""), name_index)
        if src_id and tgt_id and src_id != tgt_id:
            resolved.append(
                {
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "strength": float(edge.get("strength", 1.0)),
                    "rationale": str(edge.get("rationale", "")),
                }
            )
    return resolved


def build_graph(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Infer prerequisite edges in difficulty-tier batches, resolve cycles, init progress.

    Strategy:
      1. Group concepts by difficulty tier (1-2, 3, 4-5).
      2. For each batch of ≤_BATCH_SIZE concepts, call the LLM with:
           - The batch as the "focus" concepts to find edges for.
           - All lower-tier concept names as "available prerequisites" context.
      3. Collect and save all resolved edges.
      4. Detect/resolve cycles.
      5. Init progress (roots → UNLOCKED).

    Returns summary: {"edges_added": int, "edges_removed_cycles": int}
    """
    topic = topic_id.replace("-", " ")

    rows = conn.execute(
        "SELECT id, name, description, difficulty FROM concepts WHERE topic_id = ? ORDER BY difficulty, name",
        [topic_id],
    ).fetchall()

    if not rows:
        logger.warning("No concepts found for topic '%s'; skipping graph build", topic_id)
        return {"edges_added": 0, "edges_removed_cycles": 0}

    all_concepts = [
        {"id": r[0], "name": r[1], "description": r[2] or "", "difficulty": r[3] or 3} for r in rows
    ]
    name_index = _build_name_index(all_concepts)

    # Group into difficulty tiers for batching
    tier_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in all_concepts:
        diff = c["difficulty"]
        tier = 1 if diff <= 2 else (2 if diff == 3 else 3)
        tier_map[tier].append(c)

    total_concepts = len(all_concepts)
    if progress_callback:
        progress_callback(
            f"Building graph for {total_concepts} concepts across {len(tier_map)} difficulty tiers…"
        )

    all_resolved: list[dict[str, Any]] = []
    workers = settings.graph_builder_workers

    # Precompute lower-tier context per tier (sorted by difficulty DESC so the concepts
    # closest to the current tier appear first — most likely to be direct prerequisites).
    # Capped at _MAX_LOWER_CONTEXT to keep LLM prompts tractable.
    sorted_tiers = sorted(tier_map.keys())
    tier_lower_context: dict[int, list[dict[str, Any]]] = {}
    for tier in sorted_tiers:
        candidates = sorted(
            [c for t in sorted_tiers if t < tier for c in tier_map[t]],
            key=lambda c: c["difficulty"],
            reverse=True,
        )
        tier_lower_context[tier] = candidates[:_MAX_LOWER_CONTEXT]

    # Build the full list of (batch_index, tier, batch, lower_context) jobs per tier.
    # Batches within a tier are independent — they only read from `concepts` (already
    # in memory) and return resolved edges. DB writes happen after each tier completes
    # so the next tier's lower_context reflects saved edges. Workers never touch DuckDB.
    def _process_batch(
        batch_idx: int,
        tier: int,
        batch: list[dict[str, Any]],
        lower_ctx: list[dict[str, Any]],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        context = batch + lower_ctx  # disjoint by tier construction
        logger.info(
            "Batch %d: tier %d, %d focus concepts + %d lower-tier context",
            batch_idx,
            tier,
            len(batch),
            len(lower_ctx),
        )
        raw_edges = _llm_infer_edges(topic, context)
        batch_ids = {c["id"] for c in batch}
        resolved = [e for e in _resolve_edges(raw_edges, name_index) if e["target_id"] in batch_ids]
        logger.info("  Batch %d → %d edges", batch_idx, len(resolved))
        return batch_idx, tier, resolved

    for tier in sorted_tiers:
        tier_concepts = tier_map[tier]
        lower_context = tier_lower_context[tier]

        batches = [
            tier_concepts[s : s + _BATCH_SIZE] for s in range(0, len(tier_concepts), _BATCH_SIZE)
        ]
        global_batch_start = sum(
            len(range(0, len(tier_map[t]), _BATCH_SIZE)) for t in sorted_tiers if t < tier
        )

        if progress_callback:
            progress_callback(
                f"Tier {tier}: {len(batches)} batches × {len(batches[0])} concepts"
                f" ({workers} parallel workers)…"
            )

        tier_resolved: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_batch,
                    global_batch_start + i + 1,
                    tier,
                    batch,
                    lower_context,
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                _, _, resolved = future.result()
                tier_resolved.extend(resolved)

        # Save tier edges before moving to the next tier (so lower_context for
        # subsequent tiers reflects persisted edges, though currently we use concept
        # difficulty not edge data for context selection — kept for correctness).
        save_edges_bulk(conn, tier_resolved)
        all_resolved.extend(tier_resolved)
        logger.info(
            "Tier %d complete: %d edges saved (running total: %d)",
            tier,
            len(tier_resolved),
            len(all_resolved),
        )
    logger.info("Saved %d total prerequisite edges for '%s'", len(all_resolved), topic_id)

    if progress_callback:
        progress_callback(f"Checking {len(all_resolved)} edges for cycles…")

    removed = _resolve_cycles(conn, topic_id, topic)
    _init_progress(conn, topic_id)

    if progress_callback:
        progress_callback("Graph build complete.")

    return {"edges_added": len(all_resolved), "edges_removed_cycles": removed}
