"""Knowledge graph store.

DuckDB (prerequisite_edges table) is the authoritative persistence layer —
all writes go there first and it survives restarts without FalkorDB.

FalkorDB is synced from DuckDB and used for:
  - Cypher multi-hop traversal (ancestors, descendants, unlock candidates)
  - Visual graph exploration via FalkorDB browser

Node label:         :Concept  {id, topic_id, name, description, difficulty}
Relationship type:  -[:REQUIRES {strength, rationale}]->
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb
from falkordb import FalkorDB  # type: ignore[import]

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FalkorDB connection
# ---------------------------------------------------------------------------


def get_falkordb_graph(topic_id: str) -> Any:
    """Return a FalkorDB graph handle for *topic_id*.

    Each topic gets its own named graph inside the FalkorDB instance.
    Raises on connection failure — callers should handle gracefully when
    FalkorDB is optional (e.g. during extraction without Docker).
    """
    db = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
    return db.select_graph(topic_id)


def sync_to_falkordb(conn: duckdb.DuckDBPyConnection, topic_id: str) -> None:
    """Full sync: rebuild the FalkorDB graph for *topic_id* from DuckDB.

    Idempotent — clears the graph first, then recreates nodes and edges.
    Call after a complete graph build or to repair drift.
    """
    try:
        graph = get_falkordb_graph(topic_id)
    except Exception as e:
        logger.warning("FalkorDB unavailable, skipping sync for '%s': %s", topic_id, e)
        return

    # Wipe existing graph data for this topic
    graph.query("MATCH (n) DETACH DELETE n")

    # Sync concept nodes — single UNWIND instead of N individual CREATEs
    nodes = conn.execute(
        "SELECT id, name, description, difficulty FROM concepts WHERE topic_id = ?",
        [topic_id],
    ).fetchall()
    if nodes:
        graph.query(
            "UNWIND $nodes AS n CREATE (:Concept {id: n.id, topic_id: n.tid, name: n.name, description: n.desc, difficulty: n.diff})",
            {
                "nodes": [
                    {
                        "id": nid,
                        "tid": topic_id,
                        "name": name or "",
                        "desc": desc or "",
                        "diff": diff or 3,
                    }
                    for nid, name, desc, diff in nodes
                ]
            },
        )

    # Sync prerequisite edges — single UNWIND instead of M individual MATCHes
    edges = conn.execute(
        """
        SELECT e.source_id, e.target_id, COALESCE(e.strength, 1.0), COALESCE(e.rationale, '')
        FROM prerequisite_edges e
        JOIN concepts s ON s.id = e.source_id AND s.topic_id = ?
        JOIN concepts t ON t.id = e.target_id AND t.topic_id = ?
        """,
        [topic_id, topic_id],
    ).fetchall()
    if edges:
        graph.query(
            """
            UNWIND $edges AS e
            MATCH (a:Concept {id: e.src}), (b:Concept {id: e.tgt})
            CREATE (a)-[:REQUIRES {strength: e.w, rationale: e.rat}]->(b)
            """,
            {
                "edges": [
                    {"src": src, "tgt": tgt, "w": strength, "rat": rationale}
                    for src, tgt, strength, rationale in edges
                ]
            },
        )

    logger.info("Synced '%s' to FalkorDB: %d nodes, %d edges", topic_id, len(nodes), len(edges))


# ---------------------------------------------------------------------------
# DuckDB edge CRUD (source of truth)
# ---------------------------------------------------------------------------


def save_edge(
    conn: duckdb.DuckDBPyConnection,
    source_id: str,
    target_id: str,
    strength: float = 1.0,
    rationale: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO prerequisite_edges (source_id, target_id, strength, rationale)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (source_id, target_id) DO UPDATE SET
            strength  = excluded.strength,
            rationale = excluded.rationale
        """,
        [source_id, target_id, strength, rationale],
    )


def save_edges_bulk(
    conn: duckdb.DuckDBPyConnection,
    edges: list[dict[str, Any]],
) -> None:
    """Upsert edges to DuckDB. Call sync_to_falkordb() afterward to propagate."""
    if not edges:
        return
    conn.executemany(
        """
        INSERT INTO prerequisite_edges (source_id, target_id, strength, rationale)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (source_id, target_id) DO UPDATE SET
            strength  = excluded.strength,
            rationale = excluded.rationale
        """,
        [
            (e["source_id"], e["target_id"], e.get("strength", 1.0), e.get("rationale", ""))
            for e in edges
        ],
    )


def delete_edge(conn: duckdb.DuckDBPyConnection, source_id: str, target_id: str) -> None:
    conn.execute(
        "DELETE FROM prerequisite_edges WHERE source_id = ? AND target_id = ?",
        [source_id, target_id],
    )
