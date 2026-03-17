"""Knowledge graph API endpoints."""

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException

from backend.db.duckdb_client import db_dependency

router = APIRouter(tags=["graph"])

_DEFAULT_STATUS = "locked"

# Umbrella-concept words used to apply criticality penalty
_UMBRELLA = {
    "systems",
    "processes",
    "activities",
    "principles",
    "organisms",
    "functions",
    "mechanisms",
    "environments",
    "conditions",
    "factors",
    "interactions",
    "resources",
    "habitats",
    "populations",
    "communities",
    "dynamics",
    "variations",
    "changes",
    "effects",
    "impacts",
}


def _is_umbrella(name: str, fanout: int) -> bool:
    if fanout < 5:
        return False
    return bool(set(name.lower().split()) & _UMBRELLA)


def _compute_graph_data(conn, topic_id: str) -> dict:
    """Return nodes + edges with criticality scores for the given topic."""
    rows = conn.execute(
        "SELECT id, name, description, difficulty FROM concepts WHERE topic_id = ? ORDER BY name",
        [topic_id],
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No concepts found for topic '{topic_id}'")

    edge_rows = conn.execute(
        """
        SELECT pe.source_id, pe.target_id, pe.strength, pe.rationale
        FROM prerequisite_edges pe
        JOIN concepts s ON s.id = pe.source_id AND s.topic_id = ?
        JOIN concepts t ON t.id = pe.target_id AND t.topic_id = ?
        """,
        [topic_id, topic_id],
    ).fetchall()

    progress_rows = conn.execute(
        "SELECT concept_id, status FROM concept_progress WHERE topic_id = ?",
        [topic_id],
    ).fetchall()
    progress = {r[0]: r[1] for r in progress_rows}

    # Build networkx graph for transitive fanout computation
    G = nx.DiGraph()
    for r in rows:
        G.add_node(r[0])
    for src, tgt, *_ in edge_rows:
        G.add_edge(src, tgt)

    transitive_fanout = {n: len(nx.descendants(G, n)) for n in G.nodes()}  # type: ignore[attr-defined]

    # Criticality scoring
    id_to_name = {r[0]: r[1] for r in rows}
    scores = {
        nid: transitive_fanout[nid]
        * (0.3 if _is_umbrella(id_to_name[nid], transitive_fanout[nid]) else 1.0)
        for nid in G.nodes()
    }
    sorted_scores = sorted(scores.values(), reverse=True)
    n_total = len(sorted_scores)
    thresholds = {
        "CORE": sorted_scores[max(0, int(n_total * 0.08))],
        "IMPORTANT": sorted_scores[max(0, int(n_total * 0.25))],
        "STANDARD": sorted_scores[max(0, int(n_total * 0.60))],
    }

    def tier(score: float) -> str:
        if score >= thresholds["CORE"] and score > 0:
            return "CORE"
        if score >= thresholds["IMPORTANT"] and score > 0:
            return "IMPORTANT"
        if score >= thresholds["STANDARD"] and score > 0:
            return "STANDARD"
        return "SUPPLEMENTARY"

    # Topological layer via longest-path BFS from roots
    layers: dict[str, int] = {}
    roots = [n for n in G if G.in_degree(n) == 0]
    for r in roots:
        layers[r] = 0
    queue = list(roots)
    visited = set(roots)
    while queue:
        nxt = []
        for node in queue:
            for succ in G.successors(node):
                nl = layers[node] + 1
                if succ not in layers or layers[succ] < nl:
                    layers[succ] = nl
                if succ not in visited:
                    visited.add(succ)
                    nxt.append(succ)
        queue = nxt
    for n in G:
        layers.setdefault(n, 0)

    nodes = [
        {
            "id": r[0],
            "name": r[1],
            "description": r[2] or "",
            "difficulty": r[3],
            "status": progress.get(r[0], _DEFAULT_STATUS),
            "transitive_fanout": transitive_fanout[r[0]],
            "criticality_score": round(scores[r[0]], 2),
            "tier": tier(scores[r[0]]),
            "layer": layers.get(r[0], 0),
        }
        for r in rows
    ]
    edges = [
        {
            "source": src,
            "target": tgt,
            "strength": round(strength, 3),
            "rationale": rationale or "",
        }
        for src, tgt, strength, rationale in edge_rows
    ]

    return {"topic_id": topic_id, "nodes": nodes, "edges": edges}


@router.get("/topics/{topic_id}/graph")
async def get_graph(topic_id: str, conn=Depends(db_dependency)) -> dict:
    return _compute_graph_data(conn, topic_id)


@router.get("/topics/{topic_id}/graph/concept/{concept_id}")
async def get_concept(topic_id: str, concept_id: str, conn=Depends(db_dependency)) -> dict:
    row = conn.execute(
        "SELECT id, name, description, difficulty FROM concepts WHERE id = ? AND topic_id = ?",
        [concept_id, topic_id],
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Concept not found")
    prereqs = conn.execute(
        """SELECT c.id, c.name FROM prerequisite_edges pe
           JOIN concepts c ON c.id = pe.source_id
           WHERE pe.target_id = ? AND c.topic_id = ?""",
        [concept_id, topic_id],
    ).fetchall()
    dependents = conn.execute(
        """SELECT c.id, c.name FROM prerequisite_edges pe
           JOIN concepts c ON c.id = pe.target_id
           WHERE pe.source_id = ? AND c.topic_id = ?""",
        [concept_id, topic_id],
    ).fetchall()
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "difficulty": row[3],
        "prerequisites": [{"id": r[0], "name": r[1]} for r in prereqs],
        "dependents": [{"id": r[0], "name": r[1]} for r in dependents],
    }


@router.post("/topics/{topic_id}/graph/validate")
async def validate_graph(topic_id: str, conn=Depends(db_dependency)) -> dict:
    from backend.graph.queries import detect_cycles

    cycles = detect_cycles(conn, topic_id)
    return {"topic_id": topic_id, "cycles": cycles, "valid": len(cycles) == 0}
