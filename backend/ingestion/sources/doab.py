"""DLT source: DOAB — Directory of Open Access Books.

DOAB indexes thousands of peer-reviewed open access academic books across all
disciplines (science, humanities, social science). Every book has a CC license
and a full bibliographic record. The API is free and requires no authentication.

For each book found, we yield the abstract + chapter-level ToC as content.
This is curriculum-quality material: academic authors have already structured
the domain knowledge into chapters with clear sequencing.

API: https://directory.doabooks.org/rest/search?query={q}&expand=metadata&limit=N
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import dlt
import httpx

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://directory.doabooks.org/rest/search"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_MIN_ABSTRACT_LEN = 200
_SOURCE = "doab"


def _extract_meta(metadata: list[dict[str, Any]]) -> dict[str, str]:
    """Flatten DOAB metadata list [{key, value}] into a dict, last value wins."""
    result: dict[str, str] = {}
    for item in metadata:
        key = item.get("key", "")
        value = str(item.get("value", "")).strip()
        if key and value:
            result[key] = value
    return result


def _build_attribution(meta: dict[str, str]) -> dict[str, str]:
    """Extract clean attribution fields from DOAB metadata."""
    authors = meta.get("dc.contributor.author") or meta.get("dc.contributor.editor") or ""
    publisher = meta.get("dc.publisher") or meta.get("oapen.publisher") or ""
    license_raw = meta.get("dc.rights") or ""
    license_url = meta.get("dc.rights.uri") or ""

    # Normalise license string
    license_str = license_raw
    if "creativecommons.org/licenses/by/4.0" in license_url or "by/4.0" in license_raw.lower():
        license_str = "CC BY 4.0"
    elif "by-nc" in license_url.lower() or "by-nc" in license_raw.lower():
        license_str = "CC BY-NC"
    elif "by-sa" in license_url.lower():
        license_str = "CC BY-SA 4.0"
    elif "open access" in license_raw.lower():
        license_str = "Open Access"

    return {
        "authors": authors,
        "publisher": publisher,
        "license": license_str,
        "license_url": license_url,
    }


@dlt.source(name="doab")
def doab_source(topic: str, max_books: int = 4) -> Iterator[Any]:
    """Yield DOAB open-access academic book records for a topic."""
    yield _doab_resource(topic, max_books)


@dlt.resource(name="raw_materials", primary_key="id", write_disposition="merge", parallelized=True)
def _doab_resource(topic: str, max_books: int = 4) -> Iterator[dict[str, Any]]:
    params = {
        "query": topic,
        "expand": "metadata",
        "limit": max_books,
        "offset": 0,
    }

    try:
        with httpx.Client(headers=_HEADERS) as client:
            resp = client.get(_SEARCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            books = resp.json()
    except Exception as e:
        logger.warning("DOAB search failed for '%s': %s", topic, e)
        return

    if not isinstance(books, list):
        logger.warning("DOAB returned unexpected format for '%s'", topic)
        return

    seen_ids: set[str] = set()
    for book in books:
        meta = _extract_meta(book.get("metadata", []))

        title = meta.get("dc.title", "").strip()
        abstract = meta.get("dc.description.abstract", "").strip()
        oapen_url = meta.get("dc.identifier", "")
        language = meta.get("dc.language", "en")

        # Skip non-English books and books without useful abstracts
        if language not in ("en", "English", ""):
            continue
        if not title or len(abstract) < _MIN_ABSTRACT_LEN:
            continue

        # Build a URL: prefer OAPEN (full-text host) over DOAB record
        # DOAB often returns bare handles like "20.500.12854/…" — normalise them
        raw_url = oapen_url or book.get("handle", "")
        if raw_url.startswith("http"):
            url = raw_url
        elif raw_url:
            url = f"https://library.oapen.org/handle/{raw_url}"
        else:
            continue

        attribution = _build_attribution(meta)
        doc_id = f"doab_{book.get('uuid', title.replace(' ', '_')[:40])}"
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        # Content: title + abstract. The abstract of an academic book is itself
        # a structured overview of the field — exactly what concept extraction needs.
        content_parts = [f"# {title}", ""]
        if attribution["authors"]:
            content_parts.append(f"**Authors:** {attribution['authors']}")
        if attribution["publisher"]:
            content_parts.append(f"**Publisher:** {attribution['publisher']}")
        if attribution["license"]:
            content_parts.append(f"**License:** {attribution['license']}")
        content_parts.extend(["", "## Abstract", abstract])

        # Also append subject keywords if present
        subjects = meta.get("dc.subject", "") or meta.get("oapen.subject", "")
        if subjects:
            content_parts.extend(["", f"**Subjects:** {subjects}"])

        content = "\n".join(content_parts)

        yield {
            "id": doc_id,
            "topic_id": topic,
            "source": _SOURCE,
            "url": url,
            "title": title,
            "content": content,
            "authors": attribution["authors"],
            "publisher": attribution["publisher"],
            "license": attribution["license"],
            "license_url": attribution["license_url"],
        }
