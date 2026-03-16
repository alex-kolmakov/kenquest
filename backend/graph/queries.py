"""Graph query functions — split into two layers.

Layer 1 — DuckDB-backed pure Python (testable without FalkorDB):
  root_concepts, detect_cycles, weakest_edge_in_cycle,
  topological_order, unlock_candidates, graph_to_serializable

Layer 2 — FalkorDB Cypher (optional, for multi-hop traversal):
  ancestors, descendants

Edge direction convention:
  source → target  means  source IS A PREREQUISITE OF target
  i.e. you must learn `source` before `target`.

Unlock logic:
  A concept is UNLOCKED when ALL its direct prerequisites are MASTERED.
  Root nodes (no incoming edges) start as UNLOCKED.
"""

from __future__ import annotations

import logging
from collections import deque

import duckdb

from backend.graph.store import get_falkordb_graph
from backend.models.concept import MasteryStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_edges(conn: duckdb.DuckDBPyConnection, topic_id: str) -> list[tuple[str, str, float]]:
    """Return all prerequisite edges for *topic_id* as (src, tgt, strength)."""
    return conn.execute(
        """
        SELECT e.source_id, e.target_id, COALESCE(e.strength, 1.0)
        FROM prerequisite_edges e
        JOIN concepts s ON s.id = e.source_id AND s.topic_id = ?
        JOIN concepts t ON t.id = e.target_id AND t.topic_id = ?
        """,
        [topic_id, topic_id],
    ).fetchall()


def _fetch_concept_ids(conn: duckdb.DuckDBPyConnection, topic_id: str) -> list[str]:
    """Return all concept IDs for *topic_id* (includes isolated nodes with no edges)."""
    return [
        r[0]
        for r in conn.execute("SELECT id FROM concepts WHERE topic_id = ?", [topic_id]).fetchall()
    ]


def fetch_cycle_edges(
    conn: duckdb.DuckDBPyConnection, cycle: list[str]
) -> dict[tuple[str, str], dict]:
    """Fetch edge attributes for all edges in *cycle* from DuckDB.

    Returns {(src, tgt): {"strength": float, "rationale": str}} for each
    edge that exists in prerequisite_edges. Used by both weakest_edge_in_cycle
    and _llm_resolve_cycle to avoid duplicating the OR-clause query pattern.
    """
    cycle_pairs = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    conditions = " OR ".join(["(source_id = ? AND target_id = ?)"] * len(cycle_pairs))
    flat: list[str] = [x for pair in cycle_pairs for x in pair]
    rows = conn.execute(
        f"SELECT source_id, target_id, COALESCE(strength, 1.0), COALESCE(rationale, '') "
        f"FROM prerequisite_edges WHERE {conditions}",
        flat,
    ).fetchall()
    return {(src, tgt): {"strength": s, "rationale": r} for src, tgt, s, r in rows}


def _find_one_cycle(adj: dict[str, set[str]]) -> list[str] | None:
    """Iterative DFS-based cycle detection. Returns one cycle as a list of node IDs, or None.

    Uses an explicit stack instead of recursion to avoid Python's default
    recursion limit (1000 frames), which could be hit on deep prerequisite chains.

    Returned list represents nodes where each consecutive pair (and the closing
    edge from last → first) forms a directed cycle.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(adj, WHITE)
    parent: dict[str, str] = {}

    for start in list(adj):
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        stack = [(start, iter(adj.get(start, set())))]

        while stack:
            node, neighbors = stack[-1]
            try:
                nbr = next(neighbors)
                nbr_color = color.get(nbr, WHITE)
                if nbr_color == GRAY:
                    # Back edge found: reconstruct cycle [nbr, ..., node]
                    cycle = [node]
                    cur = node
                    while cur != nbr:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.reverse()
                    return cycle
                if nbr_color == WHITE:
                    color[nbr] = GRAY
                    parent[nbr] = node
                    stack.append((nbr, iter(adj.get(nbr, set()))))
            except StopIteration:
                color[node] = BLACK
                stack.pop()

    return None


# ---------------------------------------------------------------------------
# DuckDB-backed graph functions
# ---------------------------------------------------------------------------


def root_concepts(conn: duckdb.DuckDBPyConnection, topic_id: str) -> list[str]:
    """Concepts with no prerequisites — the starting points of the curriculum."""
    edges = _fetch_edges(conn, topic_id)
    has_incoming = {tgt for _, tgt, *_ in edges}
    return [n for n in _fetch_concept_ids(conn, topic_id) if n not in has_incoming]


def detect_cycles(conn: duckdb.DuckDBPyConnection, topic_id: str) -> list[list[str]]:
    """Return a list containing one cycle (empty list = valid DAG).

    Callers should loop until the result is empty, resolving one cycle per iteration.
    """
    edges = _fetch_edges(conn, topic_id)
    adj: dict[str, set[str]] = {}
    for src, tgt, *_ in edges:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set())
    cycle = _find_one_cycle(adj)
    return [cycle] if cycle else []


def weakest_edge_in_cycle(conn: duckdb.DuckDBPyConnection, cycle: list[str]) -> tuple[str, str]:
    """Find the edge with the lowest strength score within *cycle* (from DuckDB).

    *cycle* is a list of node IDs; the closing edge is cycle[-1] → cycle[0].
    Always returns a valid (source_id, target_id) tuple.
    """
    edge_attrs = fetch_cycle_edges(conn, cycle)
    if not edge_attrs:
        return cycle[-1], cycle[0]
    return min(edge_attrs, key=lambda k: edge_attrs[k]["strength"])


def topological_order(conn: duckdb.DuckDBPyConnection, topic_id: str) -> list[str]:
    """Kahn's algorithm topological sort. Raises ValueError if cycles exist."""
    edges = _fetch_edges(conn, topic_id)
    all_ids = _fetch_concept_ids(conn, topic_id)

    in_degree: dict[str, int] = dict.fromkeys(all_ids, 0)
    adj: dict[str, list[str]] = {n: [] for n in all_ids}  # mutable values, can't use fromkeys

    for src, tgt, *_ in edges:
        adj.setdefault(src, []).append(tgt)
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue: deque[str] = deque(n for n in all_ids if in_degree[n] == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()  # O(1) with deque, vs O(N) for list.pop(0)
        order.append(node)
        for nbr in adj.get(node, []):
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    if len(order) != len(all_ids):
        raise ValueError("Graph contains cycles — topological sort not possible")
    return order


def unlock_candidates(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    mastered_ids: set[str],
) -> list[str]:
    """Return concept IDs that become UNLOCKED given the currently mastered set.

    A concept is unlocked when ALL its direct prerequisites are in *mastered_ids*.
    Already-mastered concepts are excluded.
    """
    edges = _fetch_edges(conn, topic_id)
    all_ids = _fetch_concept_ids(conn, topic_id)

    prereqs: dict[str, set[str]] = {n: set() for n in all_ids}
    for src, tgt, *_ in edges:
        prereqs[tgt].add(src)

    return [n for n in all_ids if n not in mastered_ids and prereqs[n].issubset(mastered_ids)]


def graph_to_serializable(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    mastered_ids: set[str] | None = None,
    unlocked_ids: set[str] | None = None,
) -> dict:
    """Convert graph to a JSON-serializable dict for the React Flow frontend.

    Returns {"nodes": [...], "edges": [...]} where each node has a `status` field.
    """
    mastered_ids = mastered_ids or set()
    unlocked_ids = unlocked_ids or set()

    nodes_rows = conn.execute(
        "SELECT id, name, description, difficulty FROM concepts WHERE topic_id = ?",
        [topic_id],
    ).fetchall()

    edges_rows = conn.execute(
        """
        SELECT e.source_id, e.target_id, COALESCE(e.strength, 1.0), COALESCE(e.rationale, '')
        FROM prerequisite_edges e
        JOIN concepts s ON s.id = e.source_id AND s.topic_id = ?
        JOIN concepts t ON t.id = e.target_id AND t.topic_id = ?
        """,
        [topic_id, topic_id],
    ).fetchall()

    nodes = []
    for nid, name, desc, diff in nodes_rows:
        if nid in mastered_ids:
            status = MasteryStatus.MASTERED
        elif nid in unlocked_ids:
            status = MasteryStatus.UNLOCKED
        else:
            status = MasteryStatus.LOCKED
        nodes.append(
            {
                "id": nid,
                "data": {
                    "name": name or nid,
                    "description": desc or "",
                    "difficulty": diff or 3,
                    "status": status,
                },
            }
        )

    edges = [
        {
            "id": f"{src}->{tgt}",
            "source": src,
            "target": tgt,
            "data": {"strength": strength, "rationale": rationale},
        }
        for src, tgt, strength, rationale in edges_rows
    ]

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# FalkorDB Cypher — multi-hop traversal (optional, degrades gracefully)
# ---------------------------------------------------------------------------


def ancestors(topic_id: str, concept_id: str) -> set[str]:
    """All concepts that are transitive prerequisites of *concept_id* (via FalkorDB)."""
    try:
        graph = get_falkordb_graph(topic_id)
        result = graph.query(
            "MATCH (a:Concept)-[:REQUIRES*]->(b:Concept {id: $id}) RETURN a.id",
            {"id": concept_id},
        )
        return {row[0] for row in result.result_set}
    except Exception as e:
        logger.warning("FalkorDB ancestors query failed for '%s': %s", concept_id, e)
        return set()


def descendants(topic_id: str, concept_id: str) -> set[str]:
    """All concepts that transitively depend on *concept_id* (via FalkorDB)."""
    try:
        graph = get_falkordb_graph(topic_id)
        result = graph.query(
            "MATCH (a:Concept {id: $id})-[:REQUIRES*]->(b:Concept) RETURN b.id",
            {"id": concept_id},
        )
        return {row[0] for row in result.result_set}
    except Exception as e:
        logger.warning("FalkorDB descendants query failed for '%s': %s", concept_id, e)
        return set()
