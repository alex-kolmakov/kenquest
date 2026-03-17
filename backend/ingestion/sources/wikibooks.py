"""DLT source: Wikibooks — free structured textbooks (CC BY-SA).

Wikibooks hosts open-content textbooks organised as chapter subpages,
e.g. "Applied Ecology/Conservation Management".

Strategy:
  1. Search Wikibooks for the topic — results are typically chapter pages.
  2. Extract unique book roots from the results (part before the first "/").
  3. For each root, discover all chapter subpages via the allpages prefix query.
  4. Fetch each chapter's plain-text extract (one raw_materials row per chapter).
  5. If a book has no subpages, fall back to fetching its root page directly.

Skip pages that are navigation artefacts: "Printable version", "Cover", "Index",
and deeply-nested subpages (more than one "/" level deep).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

import dlt
import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://en.wikibooks.org/w/api.php"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_SOURCE = "wikibooks"
_LICENSE = "CC BY-SA 4.0"
_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

_MIN_CHAPTER_CHARS = 400

# Page title suffixes that are navigation artefacts, not learning content
_SKIP_SUFFIXES = frozenset(
    {
        "Printable version",
        "Print version",
        "Cover",
        "Index",
        "Glossary",
        "Bibliography",
        "Appendix",
        "About",
        "Authors",
        "Contributors",
    }
)


def _is_nav_page(chapter_title: str) -> bool:
    """Return True if the page is a navigation/utility artefact."""
    leaf = chapter_title.rsplit("/", 1)[-1]
    return leaf in _SKIP_SUFFIXES


_STOP_WORDS = frozenset({"the", "a", "an", "of", "in", "and", "or", "for", "to", "with", "by"})


def _topic_keywords(topic: str) -> set[str]:
    """Extract content words from the topic string for relevance scoring."""
    return {w.lower() for w in topic.split() if w.lower() not in _STOP_WORDS and len(w) > 2}


def _relevance_score(root: str, snippet: str, topic_kws: set[str]) -> int:
    """Return number of topic keywords found in the book root title + search snippet."""
    text = (root + " " + snippet).lower()
    return sum(1 for kw in topic_kws if kw in text)


def _book_roots_from_search(client: httpx.Client, topic: str, max_books: int) -> list[str]:
    """Search Wikibooks for the topic; extract unique book root titles.

    Only returns roots where at least one topic keyword appears in either the
    book title or the search snippet — prevents off-topic books (e.g. "Scouting")
    from being returned for a query like "ocean conservation".
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srnamespace": "0",
        "srlimit": max_books * 6,  # buffer — many results are chapters of the same book
        "srprop": "snippet",  # include snippet for relevance scoring
        "format": "json",
    }
    try:
        resp = client.get(_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
    except Exception as e:
        logger.warning("Wikibooks search failed for '%s': %s", topic, e)
        return []

    topic_kws = _topic_keywords(topic)
    seen: set[str] = set()
    roots: list[str] = []
    skipped: list[str] = []

    for r in results:
        title: str = r.get("title", "")
        snippet: str = re.sub(r"<[^>]+>", "", r.get("snippet", ""))  # strip HTML tags
        if not title:
            continue
        root = title.split("/")[0].strip()
        if not root or root in seen:
            continue
        seen.add(root)
        score = _relevance_score(root, snippet, topic_kws)
        if score == 0:
            skipped.append(root)
            continue
        roots.append(root)
        if len(roots) >= max_books:
            break

    if skipped:
        logger.info("Wikibooks: skipped %d off-topic roots: %s", len(skipped), skipped[:5])
    logger.info("Wikibooks: found %d relevant book roots for '%s': %s", len(roots), topic, roots)
    return roots


def _chapter_titles(client: httpx.Client, book_root: str, max_chapters: int) -> list[str]:
    """Return subpage titles (chapters) for a Wikibooks book."""
    params = {
        "action": "query",
        "list": "allpages",
        "apprefix": f"{book_root}/",
        "apnamespace": "0",
        "aplimit": min(max_chapters * 2, 50),  # buffer for filtering
        "format": "json",
    }
    try:
        resp = client.get(_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("allpages", [])
    except Exception as e:
        logger.warning("Wikibooks chapter list failed for '%s': %s", book_root, e)
        return []

    chapters = []
    for p in pages:
        title: str = p.get("title", "")
        if not title or _is_nav_page(title):
            continue
        # Skip deeply nested subpages (e.g. "Book/Chapter/Sub/Sub") — usually case studies
        # Allow one level of nesting (e.g. "Book/Chapter") but skip two+
        if title.count("/") > 1:
            continue
        chapters.append(title)
        if len(chapters) >= max_chapters:
            break

    return chapters


def _fetch_text(client: httpx.Client, title: str) -> str | None:
    """Fetch the plain-text extract for a Wikibooks page. Returns None if missing or too short."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    try:
        resp = client.get(_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "missing" in page:
            return None
        extract = (page.get("extract") or "").strip()
        return extract if len(extract) >= _MIN_CHAPTER_CHARS else None
    except Exception as e:
        logger.warning("Wikibooks content fetch failed for '%s': %s", title, e)
        return None


def _make_row(title: str, book_root: str, text: str, topic: str) -> dict[str, Any]:
    """Build a raw_materials row from a Wikibooks chapter."""
    chapter_name = title.split("/", 1)[-1] if "/" in title else title
    slug = title.lower().replace(" ", "_").replace("/", "__")
    return {
        "id": f"wikibooks_{slug}",
        "topic_id": topic,
        "source": _SOURCE,
        "url": f"https://en.wikibooks.org/wiki/{title.replace(' ', '_').replace('/', '%2F')}",
        "title": f"{book_root} — {chapter_name}" if chapter_name != book_root else book_root,
        "content": f"# {book_root}: {chapter_name}\n\n{text}",
        "authors": "Wikibooks contributors",
        "publisher": "Wikibooks",
        "license": _LICENSE,
        "license_url": _LICENSE_URL,
    }


@dlt.source(name="wikibooks")
def wikibooks_source(
    topic: str,
    max_books: int = 3,
    max_chapters_per_book: int = 20,
) -> Iterator[Any]:
    """Yield Wikibooks chapter text as curriculum materials."""
    yield _wikibooks_resource(topic, max_books, max_chapters_per_book)


@dlt.resource(
    name="raw_materials",
    primary_key="id",
    write_disposition="merge",
    parallelized=True,
)
def _wikibooks_resource(
    topic: str,
    max_books: int = 3,
    max_chapters_per_book: int = 20,
) -> Iterator[dict[str, Any]]:
    with httpx.Client(headers=_HEADERS) as client:
        roots = _book_roots_from_search(client, topic, max_books)
        if not roots:
            return

        for book_root in roots:
            chapters = _chapter_titles(client, book_root, max_chapters_per_book)

            if not chapters:
                # Book has no subpages — try the root page itself
                text = _fetch_text(client, book_root)
                if text:
                    yield _make_row(book_root, book_root, text, topic)
                continue

            logger.info("Wikibooks '%s': %d chapters to fetch", book_root, len(chapters))
            yielded = 0
            for chapter_title in chapters:
                text = _fetch_text(client, chapter_title)
                if not text:
                    continue
                yield _make_row(chapter_title, book_root, text, topic)
                yielded += 1

            logger.info("Wikibooks '%s': yielded %d chapters", book_root, yielded)
