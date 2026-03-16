"""Unit tests for graph/queries.py — pure DuckDB logic, no FalkorDB required."""

from __future__ import annotations

import duckdb
import pytest

from backend.graph.queries import (
    detect_cycles,
    graph_to_serializable,
    root_concepts,
    topological_order,
    unlock_candidates,
    weakest_edge_in_cycle,
)

# ---------------------------------------------------------------------------
# Fixture: a 4-node DAG in DuckDB
#   algebra → calculus → differential-equations
#             calculus → linear-algebra
# ---------------------------------------------------------------------------


@pytest.fixture
def math_graph(db_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Insert a simple math prerequisite graph into the test DuckDB connection."""
    db_conn.execute("INSERT INTO topics (id, name) VALUES (?, ?)", ["math", "Mathematics"])
    concepts = [
        ("algebra", "math", "Algebra", "", 1),
        ("calculus", "math", "Calculus", "", 2),
        ("diff-eq", "math", "Differential Equations", "", 3),
        ("linear-alg", "math", "Linear Algebra", "", 3),
    ]
    for cid, tid, name, desc, diff in concepts:
        db_conn.execute(
            "INSERT INTO concepts (id, topic_id, name, description, difficulty) VALUES (?, ?, ?, ?, ?)",
            [cid, tid, name, desc, diff],
        )
    edges = [
        ("algebra", "calculus", 1.0, "need algebra for calculus"),
        ("calculus", "diff-eq", 0.9, "calculus needed for diff eq"),
        ("calculus", "linear-alg", 0.7, "calculus helps linear algebra"),
    ]
    for src, tgt, strength, rationale in edges:
        db_conn.execute(
            "INSERT INTO prerequisite_edges (source_id, target_id, strength, rationale) VALUES (?, ?, ?, ?)",
            [src, tgt, strength, rationale],
        )
    return db_conn


@pytest.fixture
def cyclic_graph(db_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Insert a 3-node graph with a cycle: a → b → c → a."""
    db_conn.execute("INSERT INTO topics (id, name) VALUES (?, ?)", ["cycle-topic", "Cycle Test"])
    for cid in ["a", "b", "c"]:
        db_conn.execute(
            "INSERT INTO concepts (id, topic_id, name, difficulty) VALUES (?, ?, ?, ?)",
            [cid, "cycle-topic", cid.upper(), 1],
        )
    edges = [("a", "b", 0.9), ("b", "c", 0.5), ("c", "a", 0.2)]
    for src, tgt, strength in edges:
        db_conn.execute(
            "INSERT INTO prerequisite_edges (source_id, target_id, strength) VALUES (?, ?, ?)",
            [src, tgt, strength],
        )
    return db_conn


# ---------------------------------------------------------------------------
# root_concepts
# ---------------------------------------------------------------------------


def test_root_concepts(math_graph):
    roots = root_concepts(math_graph, "math")
    assert roots == ["algebra"]


def test_root_concepts_cyclic(cyclic_graph):
    # All nodes have incoming edges in a pure cycle → no roots
    roots = root_concepts(cyclic_graph, "cycle-topic")
    assert roots == []


# ---------------------------------------------------------------------------
# detect_cycles
# ---------------------------------------------------------------------------


def test_no_cycles_in_dag(math_graph):
    assert detect_cycles(math_graph, "math") == []


def test_detect_cycle(cyclic_graph):
    cycles = detect_cycles(cyclic_graph, "cycle-topic")
    assert len(cycles) == 1
    cycle = cycles[0]
    # All cycle nodes should be from {a, b, c}
    assert set(cycle).issubset({"a", "b", "c"})
    assert len(cycle) == 3


# ---------------------------------------------------------------------------
# weakest_edge_in_cycle
# ---------------------------------------------------------------------------


def test_weakest_edge_in_cycle(cyclic_graph):
    # Cycle: a(0.9) → b(0.5) → c(0.2) → a  — weakest is c → a (0.2)
    src, tgt = weakest_edge_in_cycle(cyclic_graph, ["a", "b", "c"])
    assert (src, tgt) == ("c", "a")


def test_weakest_edge_fallback(math_graph):
    # Non-existent edges → fallback to last → first
    src, tgt = weakest_edge_in_cycle(math_graph, ["x", "y", "z"])
    assert (src, tgt) == ("z", "x")


# ---------------------------------------------------------------------------
# topological_order
# ---------------------------------------------------------------------------


def test_topological_order_dag(math_graph):
    order = topological_order(math_graph, "math")
    assert order.index("algebra") < order.index("calculus")
    assert order.index("calculus") < order.index("diff-eq")
    assert order.index("calculus") < order.index("linear-alg")


def test_topological_order_raises_on_cycle(cyclic_graph):
    with pytest.raises(ValueError, match="cycles"):
        topological_order(cyclic_graph, "cycle-topic")


# ---------------------------------------------------------------------------
# unlock_candidates
# ---------------------------------------------------------------------------


def test_unlock_candidates_no_mastered(math_graph):
    # With nothing mastered, only root nodes (algebra) should be returned
    candidates = unlock_candidates(math_graph, "math", mastered_ids=set())
    assert "algebra" in candidates
    assert "calculus" not in candidates


def test_unlock_candidates_after_algebra(math_graph):
    candidates = unlock_candidates(math_graph, "math", mastered_ids={"algebra"})
    assert "calculus" in candidates
    assert "diff-eq" not in candidates


def test_unlock_candidates_excludes_mastered(math_graph):
    candidates = unlock_candidates(math_graph, "math", mastered_ids={"algebra", "calculus"})
    assert "algebra" not in candidates
    assert "calculus" not in candidates
    assert "diff-eq" in candidates
    assert "linear-alg" in candidates


# ---------------------------------------------------------------------------
# graph_to_serializable
# ---------------------------------------------------------------------------


def test_graph_to_serializable_structure(math_graph):
    result = graph_to_serializable(math_graph, "math")
    assert "nodes" in result
    assert "edges" in result
    node_ids = {n["id"] for n in result["nodes"]}
    assert node_ids == {"algebra", "calculus", "diff-eq", "linear-alg"}
    assert len(result["edges"]) == 3


def test_graph_to_serializable_status(math_graph):
    result = graph_to_serializable(
        math_graph,
        "math",
        mastered_ids={"algebra"},
        unlocked_ids={"calculus"},
    )
    status_map = {n["id"]: n["data"]["status"] for n in result["nodes"]}
    assert status_map["algebra"] == "mastered"
    assert status_map["calculus"] == "unlocked"
    assert status_map["diff-eq"] == "locked"
