"""Unit tests for concept_extractor — all LLM calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import duckdb
import pytest

from backend.extraction.concept_extractor import _chunk_text, _dedup, extract_concepts
from backend.utils import slugify as _slugify

# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


def test_slugify_basic():
    assert _slugify("Cell Division") == "cell-division"


def test_slugify_special_chars():
    assert _slugify("DNA replication (advanced)") == "dna-replication-advanced"


def test_slugify_extra_spaces():
    assert _slugify("  plasma  membrane  ") == "plasma-membrane"


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_short_returns_single():
    # Each paragraph must be >= _MIN_CHUNK_CHARS (200) to survive the filter
    para = "The cell membrane controls what enters and leaves the cell. " * 4
    text = f"{para}\n\n{para}"
    chunks = _chunk_text(text)
    assert len(chunks) == 1


def test_chunk_text_long_splits():
    # Generate 10 paragraphs of 400 chars each → should be split
    para = "x" * 400
    text = "\n\n".join([para] * 10)
    chunks = _chunk_text(text)
    assert len(chunks) > 1


def test_chunk_text_filters_tiny():
    text = "ok\n\n" + "x" * 300
    chunks = _chunk_text(text)
    # "ok" (2 chars) should be filtered, big paragraph kept
    assert all(len(c) >= 200 for c in chunks)


# ---------------------------------------------------------------------------
# _dedup
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """A mock SentenceTransformer that returns orthogonal unit embeddings."""
    import numpy as np

    model = MagicMock()

    def encode(texts, **kwargs):
        # Each text gets a unique unit vector so no two are similar
        n = len(texts)
        embeddings = np.zeros((n, n))
        for i in range(n):
            embeddings[i, i] = 1.0
        return embeddings

    model.encode.side_effect = encode
    return model


def test_dedup_removes_exact_slug_duplicates(mock_model):
    candidates = [
        {"name": "Cell Division", "description": "a", "difficulty": 2, "source_refs": []},
        {"name": "cell division", "description": "b", "difficulty": 2, "source_refs": []},
        {"name": "Mitosis", "description": "c", "difficulty": 3, "source_refs": []},
    ]
    result = _dedup(candidates, mock_model)
    names = [c["name"] for c in result]
    assert "Cell Division" in names
    assert "cell division" not in names
    assert "Mitosis" in names
    assert len(result) == 2


def test_dedup_assigns_id(mock_model):
    candidates = [
        {"name": "DNA Replication", "description": "d", "difficulty": 3, "source_refs": []},
    ]
    result = _dedup(candidates, mock_model)
    assert result[0]["id"] == "dna-replication"


def test_dedup_removes_high_similarity():
    """Use real numpy: make two embeddings almost identical so cosine sim > 0.92."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    candidates = [
        {"name": "Cell Membrane", "description": "a", "difficulty": 2, "source_refs": []},
        {"name": "Plasma Membrane", "description": "b", "difficulty": 2, "source_refs": []},
        {"name": "Mitosis", "description": "c", "difficulty": 3, "source_refs": []},
    ]

    # Embeddings: Cell Membrane and Plasma Membrane are nearly identical (sim ≈ 0.999),
    # Mitosis is orthogonal
    emb = np.array(
        [
            [1.0, 0.001, 0.0],
            [1.0, 0.001, 0.0],  # identical → sim = 1.0 > 0.92 → deduped
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Normalise rows
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms

    model = MagicMock(spec=SentenceTransformer)
    model.encode.return_value = emb

    result = _dedup(candidates, model)
    names = [c["name"] for c in result]
    assert "Cell Membrane" in names
    assert "Plasma Membrane" not in names
    assert "Mitosis" in names
    assert len(result) == 2


# ---------------------------------------------------------------------------
# extract_concepts (integration with mocked LLM and in-memory DuckDB)
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_db():
    """In-memory DuckDB with required schema including kenquest_raw schema."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS kenquest_raw")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kenquest_raw.raw_materials (
            id VARCHAR PRIMARY KEY,
            topic_id VARCHAR,
            source VARCHAR,
            url VARCHAR,
            title VARCHAR,
            content TEXT,
            authors VARCHAR,
            publisher VARCHAR,
            license VARCHAR,
            license_url VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id VARCHAR PRIMARY KEY,
            topic_id VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            description TEXT,
            difficulty INTEGER,
            source_refs VARCHAR[]
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description VARCHAR DEFAULT '',
            status VARCHAR DEFAULT 'pending',
            concept_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            content_hash VARCHAR PRIMARY KEY,
            prompt_hash VARCHAR NOT NULL,
            response_json JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    yield conn
    conn.close()


def test_extract_concepts_empty_materials(mem_db):
    """Returns empty list when no raw materials exist."""
    result = extract_concepts(mem_db, "marine-biology")
    assert result == []


def test_extract_concepts_writes_to_db(mem_db):
    """LLM-extracted concepts are written to the concepts table."""
    # Insert a material (build content string in Python, not SQL)
    long_content = "Marine biology is the study of ocean life. " * 50
    mem_db.execute(
        "INSERT INTO kenquest_raw.raw_materials (id, topic_id, source, title, content) VALUES (?, ?, ?, ?, ?)",
        ["mat1", "marine biology", "wikipedia", "Marine Bio", long_content],
    )
    mem_db.execute("""
        INSERT INTO topics (id, name) VALUES ('marine-biology', 'Marine Biology')
    """)

    llm_response = MagicMock()
    llm_response.choices[
        0
    ].message.content = '[{"name": "Ocean Ecology", "description": "Study of marine ecosystems", "difficulty": 2, "source_refs": ["mat1"]}]'

    with patch(
        "backend.extraction.concept_extractor.litellm.completion", return_value=llm_response
    ):
        with patch("backend.extraction.concept_extractor.SentenceTransformer") as mock_st:
            import numpy as np

            mock_st.return_value.encode.return_value = np.array([[1.0, 0.0]])
            result = extract_concepts(mem_db, "marine-biology")

    assert len(result) == 1
    assert result[0]["name"] == "Ocean Ecology"
    row = mem_db.execute(
        "SELECT id, name FROM concepts WHERE topic_id = 'marine-biology'"
    ).fetchone()
    assert row is not None
    assert row[1] == "Ocean Ecology"
