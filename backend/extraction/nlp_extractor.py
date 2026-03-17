"""NLP-based concept extraction — replaces LLM-per-chunk extraction.

Pipeline:
  1. Load raw_materials for the topic from kenquest_raw schema.
  2. Extract noun-phrase candidates from all documents using spaCy.
  3. Cross-document frequency filter — keep candidates that appear in ≥ N docs
     (N scales with corpus size so small topics are not starved).
  4. Wikipedia API lookup for each surviving candidate:
       - Found → Wikipedia intro paragraph = description; article length → difficulty.
       - Not found → marked for LLM fallback batch.
  5. LLM batch call (CONCEPT_VALIDATION prompt) for Wikipedia-unknown candidates,
     grouped in batches of 30. The LLM validates relevance + writes descriptions.
  6. Two-pass embedding deduplication (slug-exact → cosine similarity).
  7. Write to concepts table.

LLM is only called for ~20-30% of candidates (those absent from Wikipedia),
reducing cost ~75% compared to LLM-from-raw-text extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import duckdb
import httpx
import litellm
import spacy
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.extraction.prompts import CONCEPT_VALIDATION_SYSTEM, CONCEPT_VALIDATION_USER
from backend.utils import parse_llm_json_list, safe_temperature, slugify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_WIKI_API = "https://en.wikipedia.org/w/api.php"

_MIN_WORDS = 2
_MAX_WORDS = 6
_MIN_DESCRIPTION_CHARS = 80  # discard Wikipedia intros that are stubs
_DEDUP_THRESHOLD = 0.88  # cosine similarity above this → duplicate
_LLM_BATCH_SIZE = 30  # candidates per LLM validation call
_WIKI_WORKERS = 10  # parallel Wikipedia lookup threads

# Tokens whose entity types signal a proper noun / non-concept
_SKIP_ENT_TYPES = frozenset(
    {
        "PERSON",
        "GPE",
        "LOC",
        "FAC",
        "ORG",
        "DATE",
        "TIME",
        "CARDINAL",
        "ORDINAL",
        "MONEY",
        "PERCENT",
        "QUANTITY",
    }
)

# Determiners to strip from the start of a noun phrase
_LEADING_DET = re.compile(
    r"^(the|a|an|this|these|those|that|some|any|each|every|both|all|its|their|our|my|your|his|her)\s+",
    re.IGNORECASE,
)

# Generic adjectives/quantifiers that commonly lead non-concept NPs.
# Phrases starting with these followed by a generic noun are filtered out.
_GENERIC_LEADING = frozenset(
    {
        "important",
        "significant",
        "major",
        "minor",
        "key",
        "main",
        "primary",
        "secondary",
        "general",
        "specific",
        "various",
        "different",
        "certain",
        "large",
        "small",
        "high",
        "low",
        "great",
        "little",
        "much",
        "many",
        "new",
        "old",
        "recent",
        "current",
        "modern",
        "traditional",
        "common",
        "similar",
        "same",
        "other",
        "additional",
        "further",
        "potential",
        "overall",
        "total",
        "local",
        "global",
        "natural",
        "human",
        "direct",
        "indirect",
        "positive",
        "negative",
        "physical",
        "chemical",
        "biological",
        "such",
        "single",
        "multiple",
        "several",
        "few",
        "number",
    }
)

# Generic nouns that, when paired with a generic adjective head, form non-concept phrases
_GENERIC_TAIL_NOUNS = frozenset(
    {
        "role",
        "factor",
        "impact",
        "effect",
        "level",
        "amount",
        "rate",
        "way",
        "number",
        "type",
        "form",
        "part",
        "use",
        "result",
        "change",
        "example",
        "case",
        "point",
        "aspect",
        "feature",
        "process",
        "problem",
        "issue",
        "area",
        "region",
        "site",
        "location",
        "period",
        "time",
        "year",
        "study",
        "research",
        "work",
        "approach",
        "method",
        "analysis",
    }
)

_NLP: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        logger.info("Loading spaCy model en_core_web_sm …")
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        _NLP.max_length = 2_000_000
    return _NLP


# ---------------------------------------------------------------------------
# Step 1 — NP candidate extraction
# ---------------------------------------------------------------------------


def _extract_candidates_from_text(text: str, nlp: spacy.language.Language) -> set[str]:
    """Return cleaned noun-phrase candidates from a single document."""
    # Process in slices to handle very large documents
    slice_size = 100_000
    candidates: set[str] = set()

    for start in range(0, len(text), slice_size):
        chunk = text[start : start + slice_size]
        doc = nlp(chunk)
        for np in doc.noun_chunks:
            # Strip leading determiners
            raw = _LEADING_DET.sub("", np.text).strip()
            # Drop phrases that start with punctuation artifacts like "("
            if raw and not raw[0].isalpha():
                continue
            words = raw.split()
            if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
                continue
            # Skip phrases dominated by stop words
            non_stop = [t for t in np if not t.is_stop and t.is_alpha]
            if not non_stop:
                continue
            # Skip if any token is an unwanted entity type
            if any(t.ent_type_ in _SKIP_ENT_TYPES for t in np):
                continue
            # Skip phrases where a generic adjective leads into a generic noun
            # e.g. "important role", "high level", "major factor"
            if (
                len(words) == 2
                and words[0].lower() in _GENERIC_LEADING
                and words[-1].lower() in _GENERIC_TAIL_NOUNS
            ):
                continue
            candidates.add(raw.lower())

    return candidates


# ---------------------------------------------------------------------------
# Step 2 — Cross-document frequency filter
# ---------------------------------------------------------------------------


def _cross_doc_candidates(
    materials: list[tuple[str, str]],  # [(mat_id, content), ...]
    nlp: spacy.language.Language,
    min_docs: int,
) -> dict[str, dict[str, Any]]:
    """
    Return {slug: {name, source_ids}} for candidates appearing in ≥ min_docs documents.
    Slug normalisation is the same as the main dedup pass.
    """
    slug_to_sources: dict[str, set[str]] = defaultdict(set)
    slug_to_name: dict[str, str] = {}

    for mat_id, content in materials:
        candidates = _extract_candidates_from_text(content, nlp)
        for name in candidates:
            slug = slugify(name)
            if not slug:
                continue
            slug_to_sources[slug].add(mat_id)
            # Keep the most common capitalisation (first seen wins here)
            if slug not in slug_to_name:
                slug_to_name[slug] = name

    return {
        slug: {"name": slug_to_name[slug], "source_ids": list(sources)}
        for slug, sources in slug_to_sources.items()
        if len(sources) >= min_docs
    }


# ---------------------------------------------------------------------------
# Step 3 — Wikipedia lookup (cached in llm_cache)
# ---------------------------------------------------------------------------


def _wiki_cache_key(slug: str) -> str:
    return hashlib.sha256(f"wiki_summary||{slug}".encode()).hexdigest()


def _load_wiki_cache(conn: duckdb.DuckDBPyConnection, slugs: list[str]) -> dict[str, Any]:
    """Bulk-load cached Wikipedia results. Returns {slug: result_or_None}."""
    keys = {s: _wiki_cache_key(s) for s in slugs}
    rev = {v: k for k, v in keys.items()}
    placeholders = ", ".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT content_hash, response_json FROM llm_cache WHERE content_hash IN ({placeholders})",
        list(keys.values()),
    ).fetchall()
    cached: dict[str, Any] = {}
    for content_hash, response_json in rows:
        slug = rev.get(content_hash)
        if slug:
            try:
                cached[slug] = json.loads(response_json)
            except json.JSONDecodeError:
                pass
    return cached


def _save_wiki_cache(conn: duckdb.DuckDBPyConnection, slug: str, result: Any) -> None:
    key = _wiki_cache_key(slug)
    conn.execute(
        "INSERT OR IGNORE INTO llm_cache (content_hash, prompt_hash, response_json) VALUES (?, ?, ?)",
        [key, key, json.dumps(result)],
    )


def _difficulty_from_length(length: int) -> int:
    """Map Wikipedia article byte-length to a 1-5 difficulty score."""
    if length < 5_000:
        return 1
    if length < 15_000:
        return 2
    if length < 40_000:
        return 3
    if length < 90_000:
        return 4
    return 5


def _wikipedia_lookup_one(client: httpx.Client, name: str) -> dict[str, Any] | None:
    """
    Fetch Wikipedia intro + article length for a concept name.
    Returns {description, difficulty} or None if not found / intro too short.
    """
    params = {
        "action": "query",
        "titles": name,
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "redirects": True,
        "format": "json",
    }
    try:
        resp = client.get(_WIKI_API, params=params, timeout=12)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
    except Exception as e:
        logger.debug("Wikipedia lookup failed for '%s': %s", name, e)
        return None

    if "missing" in page:
        return None

    extract = (page.get("extract") or "").replace("\n", " ").strip()
    length = page.get("length", 0)

    # Build a 2-sentence description from the intro
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", extract) if len(s.strip()) > 20]
    description = " ".join(sentences[:2])

    if len(description) < _MIN_DESCRIPTION_CHARS:
        return None  # stub article — not useful as a definition

    return {
        "description": description[:500],
        "difficulty": _difficulty_from_length(length),
    }


def _wikipedia_lookup_batch(
    conn: duckdb.DuckDBPyConnection,
    candidates: dict[str, dict[str, Any]],  # {slug: {name, source_ids}}
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Look up all candidates in Wikipedia (with caching).

    Returns:
      found: {slug: {name, description, difficulty, source_ids}}
      unknown_slugs: slugs for which Wikipedia had no usable result
    """
    slugs = list(candidates.keys())
    cached = _load_wiki_cache(conn, slugs)

    # Split into already-cached and needs-fetch
    to_fetch = [s for s in slugs if s not in cached]

    if to_fetch:
        logger.info("Wikipedia: looking up %d candidates (%d cached) …", len(to_fetch), len(cached))

        def _fetch(slug: str) -> tuple[str, Any]:
            name = candidates[slug]["name"]
            with httpx.Client(headers=_HEADERS, timeout=12) as client:
                result = _wikipedia_lookup_one(client, name)
            _save_wiki_cache(conn, slug, result)
            return slug, result

        with ThreadPoolExecutor(max_workers=_WIKI_WORKERS) as pool:
            futures = {pool.submit(_fetch, s): s for s in to_fetch}
            for future in as_completed(futures):
                slug, result = future.result()
                cached[slug] = result

    found: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []

    for slug in slugs:
        result = cached.get(slug)
        if result:
            found[slug] = {
                "name": candidates[slug]["name"],
                "description": result["description"],
                "difficulty": result["difficulty"],
                "source_ids": candidates[slug]["source_ids"],
            }
        else:
            unknown.append(slug)

    logger.info("Wikipedia: %d found, %d unknown → LLM batch", len(found), len(unknown))
    return found, unknown


# ---------------------------------------------------------------------------
# Step 4 — LLM batch validation for Wikipedia-unknown candidates
# ---------------------------------------------------------------------------

_LLM_PARALLEL_WORKERS = 8  # parallel LLM validation workers


def _llm_validate_batch_raw(names: list[str], topic: str) -> tuple[str, list[dict[str, Any]]]:
    """Call LLM for one validation batch. Returns (cache_key, parsed_results).
    No DB access — safe to call from threads.
    """
    terms_list = "\n".join(f"- {n}" for n in names)
    user_msg = CONCEPT_VALIDATION_USER.format(topic=topic, terms_list=terms_list)
    cache_key = hashlib.sha256(f"validate||{topic}||{terms_list}".encode()).hexdigest()

    try:
        response = litellm.completion(  # type: ignore[operator]
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            messages=[
                {"role": "system", "content": CONCEPT_VALIDATION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=safe_temperature(0.1),
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
    except Exception as e:
        logger.warning("LLM validation batch failed: %s", e)
        return cache_key, []

    return cache_key, parse_llm_json_list(raw, "concepts")


def _llm_validate_all(
    conn: duckdb.DuckDBPyConnection,
    unknown_slugs: list[str],
    candidates: dict[str, dict[str, Any]],
    topic: str,
) -> dict[str, dict[str, Any]]:
    """Run LLM validation in parallel batches, return {slug: {name, description, difficulty}}.

    Pattern mirrors graph_builder: LLM calls run in parallel threads (no DB access),
    then cache writes happen sequentially on the main thread.
    """
    if not unknown_slugs:
        return {}

    names_per_slug = {s: candidates[s]["name"] for s in unknown_slugs}
    slugs_ordered = list(names_per_slug.keys())
    batches = [
        slugs_ordered[i : i + _LLM_BATCH_SIZE]
        for i in range(0, len(slugs_ordered), _LLM_BATCH_SIZE)
    ]

    logger.info(
        "LLM validation: %d unknowns in %d batches (%d parallel workers)",
        len(unknown_slugs),
        len(batches),
        _LLM_PARALLEL_WORKERS,
    )

    # Check cache first — skip batches already cached
    all_batch_names = [[names_per_slug[s] for s in batch_slugs] for batch_slugs in batches]
    batch_cache_keys = [
        hashlib.sha256(
            f"validate||{topic}||{chr(10).join('- ' + n for n in batch_names)}".encode()
        ).hexdigest()
        for batch_names in all_batch_names
    ]
    cached_keys = {
        row[0]
        for row in conn.execute(
            f"SELECT content_hash FROM llm_cache WHERE content_hash IN ({','.join('?' * len(batch_cache_keys))})",
            batch_cache_keys,
        ).fetchall()
    }

    # Gather batches that need LLM calls
    to_call = [
        (i, batch_slugs, all_batch_names[i])
        for i, batch_slugs in enumerate(batches)
        if batch_cache_keys[i] not in cached_keys
    ]
    logger.info(
        "LLM validation: %d cached, %d need LLM calls", len(batches) - len(to_call), len(to_call)
    )

    # Parallel LLM calls (no DB access in threads)
    batch_results: dict[int, tuple[str, list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=_LLM_PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(_llm_validate_batch_raw, batch_names, topic): idx
            for idx, _batch_slugs, batch_names in to_call
        }
        for future in as_completed(futures):
            idx = futures[future]
            cache_key, parsed = future.result()
            batch_results[idx] = (cache_key, parsed)

    # Sequential: write new cache entries + load cached ones
    result: dict[str, dict[str, Any]] = {}

    def _process_batch(idx: int, batch_slugs: list[str], parsed: list[dict[str, Any]]) -> None:
        name_to_slug = {names_per_slug[s]: s for s in batch_slugs}
        for item in parsed:
            name = item.get("name", "")
            if not item.get("valid", False):
                continue
            description = (item.get("description") or "").strip()
            difficulty = max(1, min(5, int(item.get("difficulty") or 2)))
            if len(description) < 30:
                continue
            slug = name_to_slug.get(name) or slugify(name)
            if slug:
                result[slug] = {
                    "name": candidates.get(slug, {}).get("name", name),
                    "description": description,
                    "difficulty": difficulty,
                    "source_ids": candidates.get(slug, {}).get("source_ids", []),
                }

    for idx, batch_slugs, _batch_names in to_call:
        cache_key, parsed = batch_results.get(idx, (batch_cache_keys[idx], []))
        conn.execute(
            "INSERT OR IGNORE INTO llm_cache (content_hash, prompt_hash, response_json) VALUES (?, ?, ?)",
            [cache_key, cache_key, json.dumps(parsed)],
        )
        _process_batch(idx, batch_slugs, parsed)

    # Load and process already-cached batches
    for i, (batch_slugs, _batch_names) in enumerate(zip(batches, all_batch_names, strict=False)):
        cache_key = batch_cache_keys[i]
        if cache_key not in cached_keys:
            continue
        row = conn.execute(
            "SELECT response_json FROM llm_cache WHERE content_hash = ?", [cache_key]
        ).fetchone()
        if row:
            try:
                _process_batch(i, batch_slugs, json.loads(row[0]))
            except json.JSONDecodeError:
                pass

    logger.info(
        "LLM validation: %d concepts validated from %d unknowns", len(result), len(unknown_slugs)
    )
    return result


# ---------------------------------------------------------------------------
# Step 5 — Deduplication (reused from original extractor)
# ---------------------------------------------------------------------------


def _dedup(
    concepts: list[dict[str, Any]],
    model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """Remove duplicates: slug-exact first, then cosine similarity."""
    import numpy as np

    seen_slugs: set[str] = set()
    unique: list[dict[str, Any]] = []

    for c in concepts:
        slug = c["id"]
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            unique.append(c)

    if len(unique) <= 1:
        return unique

    names = [c["name"] for c in unique]
    embeddings = model.encode(names, normalize_embeddings=True)
    keep = [True] * len(unique)

    for i in range(len(unique)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(unique)):
            if not keep[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= _DEDUP_THRESHOLD:
                keep[j] = False

    return [c for c, k in zip(unique, keep, strict=True) if k]


# ---------------------------------------------------------------------------
# Step 6 — Write to DuckDB
# ---------------------------------------------------------------------------


def _write_concepts(
    conn: duckdb.DuckDBPyConnection,
    topic_id: str,
    concepts: list[dict[str, Any]],
) -> None:
    for c in concepts:
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
                max(1, min(5, int(c.get("difficulty", 2)))),
                c.get("source_ids", []),
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
    """
    Extract, validate, deduplicate, and persist concepts for *topic_id*.

    Uses NLP candidate extraction + Wikipedia lookup + LLM batch fallback.
    Returns the final list of concept dicts written to DuckDB.
    """
    topic = topic_id.replace("-", " ")

    # -- Load raw materials --------------------------------------------------
    try:
        materials_raw = conn.execute(
            """
            SELECT id, content
            FROM kenquest_raw.raw_materials
            WHERE topic_id = ?
              AND content IS NOT NULL
              AND LENGTH(content) > 200
            ORDER BY LENGTH(content) DESC
            """,
            [topic],
        ).fetchall()
    except Exception as e:
        logger.error("Failed to load raw_materials for '%s': %s", topic_id, e)
        return []

    if not materials_raw:
        logger.warning("No raw materials found for topic '%s'", topic_id)
        return []

    materials: list[tuple[str, str]] = [(r[0], r[1]) for r in materials_raw]
    logger.info("NLP extraction: %d materials for topic '%s'", len(materials), topic_id)

    if progress_callback:
        progress_callback(f"Extracting noun-phrase candidates from {len(materials)} materials …")

    # -- NP candidate extraction + cross-doc frequency filter ---------------
    nlp = _get_nlp()

    # Adaptive minimum: require ≥ 2 docs for large corpora, 1 for small
    min_docs = 2 if len(materials) >= 5 else 1
    candidates = _cross_doc_candidates(materials, nlp, min_docs=min_docs)
    logger.info(
        "NP extraction: %d candidates after cross-doc filter (min_docs=%d)",
        len(candidates),
        min_docs,
    )

    if progress_callback:
        progress_callback(f"{len(candidates)} candidates → Wikipedia lookup …")

    # -- Wikipedia lookup ----------------------------------------------------
    wiki_found, unknown_slugs = _wikipedia_lookup_batch(conn, candidates)

    if progress_callback:
        progress_callback(f"Wikipedia: {len(wiki_found)} found, {len(unknown_slugs)} → LLM batch …")

    # -- LLM batch for unknowns ----------------------------------------------
    llm_found = _llm_validate_all(conn, unknown_slugs, candidates, topic)

    # -- Merge all results ---------------------------------------------------
    all_concepts: list[dict[str, Any]] = []
    for slug, data in {**wiki_found, **llm_found}.items():
        all_concepts.append(
            {
                "id": slug,
                "name": _capitalise(data["name"]),
                "description": data["description"],
                "difficulty": data["difficulty"],
                "source_ids": data.get("source_ids", []),
            }
        )

    logger.info("Pre-dedup total: %d concepts", len(all_concepts))

    if progress_callback:
        progress_callback(f"{len(all_concepts)} concepts → relevance filter + deduplication …")

    # -- Embedding model (shared for relevance filter + dedup) ---------------
    model = SentenceTransformer(settings.embedding_model)

    # -- Topic relevance filter ----------------------------------------------
    # Score each concept against the topic using cosine similarity.
    # Removes off-topic concepts that leaked in via Wikipedia (e.g. "Mesenchymal Cells"
    # when topic is "ocean conservation") because the Wikipedia path bypasses LLM relevance checks.
    all_concepts = _filter_by_topic_relevance(all_concepts, topic, model)
    logger.info("Post-relevance-filter: %d concepts", len(all_concepts))

    # -- Embedding dedup -----------------------------------------------------
    deduplicated = _dedup(all_concepts, model)
    logger.info("Post-dedup: %d unique concepts", len(deduplicated))

    # -- Persist -------------------------------------------------------------
    _write_concepts(conn, topic_id, deduplicated)
    conn.execute(
        "UPDATE topics SET concept_count = ?, updated_at = NOW() WHERE id = ?",
        [len(deduplicated), topic_id],
    )

    if progress_callback:
        progress_callback(f"Extracted {len(deduplicated)} unique concepts.")

    logger.info("Extracted %d concepts for topic '%s'", len(deduplicated), topic_id)
    return deduplicated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RELEVANCE_THRESHOLD = 0.20  # cosine similarity vs topic string; below this → off-topic


def _filter_by_topic_relevance(
    concepts: list[dict[str, Any]],
    topic: str,
    model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """Remove concepts whose description+name is cosine-dissimilar to the topic.

    This catches off-topic concepts that entered via the Wikipedia path, which
    bypasses the LLM relevance check (e.g. 'Mesenchymal Cells' for 'ocean conservation').
    Uses the description when available, falls back to concept name only.
    """
    import numpy as np

    if not concepts:
        return concepts

    topic_vec = model.encode(topic, normalize_embeddings=True)

    texts = [
        f"{c['name']}. {c.get('description', '')}"[:300] if c.get("description") else c["name"]
        for c in concepts
    ]
    concept_vecs = model.encode(texts, normalize_embeddings=True, batch_size=64)

    kept = []
    removed = []
    for c, vec in zip(concepts, concept_vecs, strict=False):
        sim = float(np.dot(topic_vec, vec))
        if sim >= _RELEVANCE_THRESHOLD:
            kept.append(c)
        else:
            removed.append((c["name"], round(sim, 3)))

    if removed:
        logger.info(
            "Relevance filter removed %d off-topic concepts (threshold=%.2f). Sample: %s",
            len(removed),
            _RELEVANCE_THRESHOLD,
            [n for n, _ in removed[:10]],
        )
    return kept


def _capitalise(name: str) -> str:
    """Title-case a concept name, preserving known acronyms."""
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split())
