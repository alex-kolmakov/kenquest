"""DLT source: Open Textbook Library (open.umn.edu/opentextbooks).

The Open Textbook Library is a catalog of 500+ peer-reviewed open textbooks
maintained by the University of Minnesota. Every book is CC-licensed, covers
undergraduate-level subjects (STEM, humanities, social sciences, workforce),
and has been reviewed by instructors on comprehensiveness, accuracy, and clarity.

Unlike DOAB (international research monographs), OTL focuses on *teaching*
textbooks — structured for learning, ordered by curriculum, with clear chapter
sequences. This makes them ideal as syllabus scaffolding material.

API: https://open.umn.edu/opentextbooks/textbooks.json?term={query}&limit=N
No authentication required. Returns JSON.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import dlt
import httpx

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://open.umn.edu/opentextbooks/textbooks.json"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_MIN_DESCRIPTION_LEN = 150
_SOURCE = "opentextbook"

# OTL uses full Creative Commons license names — map to short form
_LICENSE_NAME_MAP = {
    "attribution": "CC BY",
    "attribution-sharealike": "CC BY-SA",
    "attribution-noderivatives": "CC BY-ND",
    "attribution-noncommercial": "CC BY-NC",
    "attribution-noncommercial-sharealike": "CC BY-NC-SA",
    "attribution-noncommercial-noderivatives": "CC BY-NC-ND",
    "public domain": "Public Domain",
    "cc0": "CC0",
}


def _normalise_license(raw: str) -> str:
    """Convert OTL license name to short CC form."""
    key = raw.strip().lower()
    # Check longest (most specific) fragments first so "attribution-noncommercial-sharealike"
    # is matched before the shorter "attribution" prefix catches it.
    for fragment, short in sorted(_LICENSE_NAME_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if fragment in key:
            return short
    return raw.strip() or ""


def _pick_url(formats: list[dict[str, Any]]) -> str:
    """Return the best available URL from OTL format list.

    Preference: Online (readable) > PDF > any.
    """
    online = next((f["url"] for f in formats if f.get("type") == "Online" and f.get("url")), None)
    if online:
        return online
    pdf = next((f["url"] for f in formats if f.get("type") == "PDF" and f.get("url")), None)
    if pdf:
        return pdf
    for f in formats:
        if f.get("url"):
            return f["url"]
    return ""


@dlt.source(name="opentextbook")
def opentextbook_source(topic: str, max_books: int = 4) -> Iterator[Any]:
    """Yield Open Textbook Library records as curriculum materials."""
    yield _opentextbook_resource(topic, max_books)


@dlt.resource(name="raw_materials", primary_key="id", write_disposition="merge", parallelized=True)
def _opentextbook_resource(topic: str, max_books: int = 4) -> Iterator[dict[str, Any]]:
    params = {"q": topic, "limit": max_books + 3}  # small buffer for description-length filtering

    try:
        with httpx.Client(headers=_HEADERS) as client:
            resp = client.get(_SEARCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            books = resp.json()
    except Exception as e:
        logger.warning("Open Textbook Library search failed for '%s': %s", topic, e)
        return

    # API wraps results: {"data": [...], "links": {...}}
    if isinstance(books, dict):
        books = books.get("data", [])
    if not isinstance(books, list):
        logger.warning("Open Textbook Library returned unexpected format for '%s'", topic)
        return

    yielded = 0
    for book in books:
        if yielded >= max_books:
            break

        title = (book.get("title") or "").strip()
        description = (book.get("description") or "").strip()
        license_raw = book.get("license") or ""
        book_id = book.get("id")

        if not title or not book_id:
            continue
        if len(description) < _MIN_DESCRIPTION_LEN:
            continue

        license_str = _normalise_license(license_raw)
        # Prefer the top-level url field (canonical OTL page), then format URLs
        url = book.get("url") or _pick_url(book.get("formats") or [])
        if not url:
            url = f"https://open.umn.edu/opentextbooks/textbooks/{book_id}"

        # Attribution
        publishers = book.get("publishers") or []
        publisher = publishers[0].get("name", "") if publishers else ""
        copyright_year = book.get("copyright_year") or ""

        subjects = book.get("subjects") or []
        subject_names = ", ".join(s["name"] for s in subjects if s.get("name"))

        content_parts = [f"# {title}", ""]
        if publisher:
            content_parts.append(f"**Publisher:** {publisher}")
        if copyright_year:
            content_parts.append(f"**Year:** {copyright_year}")
        if license_str:
            content_parts.append(f"**License:** {license_str}")
        if subject_names:
            content_parts.append(f"**Subjects:** {subject_names}")
        content_parts.extend(["", "## Overview", description])

        content = "\n".join(content_parts)

        yield {
            "id": f"otl_{book_id}",
            "topic_id": topic,
            "source": _SOURCE,
            "url": url,
            "title": title,
            "content": content,
            "authors": "",  # OTL attributes to publishers, not individual authors
            "publisher": publisher,
            "license": license_str,
            "license_url": "",
        }
        yielded += 1
