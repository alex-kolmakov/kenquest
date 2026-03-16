"""
Live integration tests — hit real public APIs.

Run with:  KENQUEST_RUN_LIVE=1 uv run pytest tests/test_ingestion/test_pipeline_live.py -v -s

Primary test topic: marine conservation (domain-science, non-tech)
Secondary topic:    machine learning (validates broader topic support)
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("KENQUEST_RUN_LIVE", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="Set KENQUEST_RUN_LIVE=1 to run live tests")


@pytest.fixture
def live_db(tmp_path):  # type: ignore[no-untyped-def]
    import backend.config as cfg

    original = cfg.settings.database_path
    cfg.settings.database_path = tmp_path / "test_live.duckdb"
    yield cfg.settings.database_path
    cfg.settings.database_path = original


# ── Phase 1: Syllabus Discovery ───────────────────────────────────────────────


@pytest.mark.parametrize("topic", ["marine conservation", "machine learning"])
def test_wikipedia_live(live_db, topic: str) -> None:  # type: ignore[no-untyped-def]
    from backend.ingestion.sources.wikipedia import _wikipedia_resource

    results = list(_wikipedia_resource(topic, max_linked=0))
    assert len(results) >= 1, f"Wikipedia: expected root article for '{topic}'"
    assert len(results[0]["content"]) > 500

    print(f"\nWikipedia root [{topic}]: {results[0]['title']} — {len(results[0]['content'])} chars")


@pytest.mark.parametrize("topic", ["marine conservation", "machine learning"])
def test_arxiv_survey_live(live_db, topic: str) -> None:  # type: ignore[no-untyped-def]
    """arXiv targeted at survey/review papers for structured field overview."""
    from backend.ingestion.sources.arxiv import _arxiv_resource

    results = list(_arxiv_resource(f"{topic} survey review", max_results=3))
    print(f"\narXiv [{topic}]: {len(results)} papers")
    for r in results:
        print(f"  {r['title'][:70]}")


# ── Phase 2: Open Book Sources ────────────────────────────────────────────────


@pytest.mark.parametrize("topic", ["marine biology", "marine conservation"])
def test_openstax_live(live_db, topic: str) -> None:  # type: ignore[no-untyped-def]
    from backend.ingestion.sources.openstax import _openstax_resource

    results = list(_openstax_resource(topic, max_chapters=4))
    assert len(results) >= 1, f"OpenStax: expected chapters for '{topic}'"

    print(f"\nOpenStax [{topic}]: {len(results)} chapters")
    for r in results:
        print(f"  [{r['id']}] {r['title']} — license: {r['license']}")
        assert r["license"] == "CC BY 4.0"
        assert r["publisher"] == "OpenStax, Rice University"
        assert len(r["content"]) > 100


@pytest.mark.parametrize("topic", ["marine conservation", "machine learning"])
def test_doab_live(live_db, topic: str) -> None:  # type: ignore[no-untyped-def]
    from backend.ingestion.sources.doab import _doab_resource

    results = list(_doab_resource(topic, max_books=3))
    assert len(results) >= 1, f"DOAB: expected books for '{topic}'"

    print(f"\nDOAB [{topic}]: {len(results)} books")
    for r in results:
        print(f"  {r['title'][:60]} | {r['license']} | {r['authors'][:40]}")
        assert r["source"] == "doab"
        # License metadata is best-effort — DOAB records vary in completeness
        if not r["license"]:
            print(f"    ⚠ no license metadata for: {r['title'][:50]}")
        assert len(r["content"]) > 200


# ── Full Pipeline ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("topic_id", ["marine-conservation", "machine-learning"])
def test_full_pipeline_live(live_db, topic_id: str) -> None:  # type: ignore[no-untyped-def]
    from backend.ingestion.pipeline import count_materials, run_ingestion

    results = run_ingestion(topic_id)
    total = count_materials(topic_id)

    print(f"\nPipeline [{topic_id}]: {results.get('total', 0)} rows loaded")
    print(f"  Total in DB: {total}")

    assert total >= 3, f"Expected ≥3 materials for '{topic_id}', got {total}. results={results}"
