"""DLT source: Wikipedia articles — primary curriculum skeleton source.

Strategy:
  1. Fetch the root article for the topic (sections → concept units).
  2. Derive linked article titles from root section headings (via search).
  3. Fetch each linked article sequentially within this resource's thread.

Parallelism:
  This resource runs in its own thread (parallelized=True), concurrently with
  arXiv, OpenStax, Open Textbook Library, and DOAB resources. Within-resource
  linked article fetching remains sequential — DLT does not support multiple
  resources with the same destination name in a single source.

This gives a structured, reliable curriculum for any topic, not just CS.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

import dlt
import httpx

logger = logging.getLogger(__name__)

_ACTION_BASE = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_SOURCE = "wikipedia"

# Sections that rarely contain educational content
_SKIP_SECTION_TITLES = frozenset(
    {
        "References",
        "Further reading",
        "External links",
        "See also",
        "Notes",
        "Citations",
        "Bibliography",
        "Footnotes",
        "Gallery",
    }
)

# Section headings to skip when deriving related article searches
_SKIP_SECTION_WORDS = frozenset({"history", "references", "further", "see", "notes"})


def _fetch_article(client: httpx.Client, title: str) -> dict[str, Any] | None:
    """Return {title, url, sections: [{title, content}]} or None on failure."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "exsectionformat": "wiki",
        "explaintext": True,
        "inprop": "url",
        "format": "json",
        "redirects": True,
    }
    try:
        resp = client.get(_ACTION_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        logger.warning("Wikipedia fetch failed for '%s': %s", title, e)
        return None

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if "missing" in page:
        logger.debug("Wikipedia page not found: '%s'", title)
        return None

    raw_extract = page.get("extract", "") or ""
    canonical_url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
    sections = _split_into_sections(raw_extract)
    return {"title": page.get("title", title), "url": canonical_url, "sections": sections}


def _split_into_sections(text: str) -> list[dict[str, str]]:
    """Parse Wikipedia plaintext extract into named sections."""
    pattern = re.compile(r"\n==+\s*(.+?)\s*==+\n", re.MULTILINE)
    parts = pattern.split(text)

    sections: list[dict[str, str]] = []

    lead = parts[0].strip()
    if len(lead) > 50:
        sections.append({"title": "Introduction", "content": lead})

    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if heading in _SKIP_SECTION_TITLES or len(content) < 60:
            continue
        sections.append({"title": heading, "content": content})

    return sections


def _search_wikipedia(client: httpx.Client, query: str, limit: int = 3) -> list[str]:
    """Search Wikipedia and return matching article titles."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srnamespace": 0,
        "format": "json",
    }
    try:
        resp = client.get(_ACTION_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]
    except Exception:
        return []


def _related_titles_from_sections(
    client: httpx.Client, sections: list[dict[str, str]], topic: str, limit: int
) -> list[str]:
    """
    Derive related article titles from root article section headings.

    Each section title is used as a Wikipedia search query. This gives
    conceptually relevant sub-articles tied to the topic's own structure.
    """
    seen: set[str] = set()
    seen.add(topic.lower())
    titles: list[str] = []

    for section in sections:
        if len(titles) >= limit:
            break
        heading = section["title"]
        if any(w in heading.lower() for w in _SKIP_SECTION_WORDS):
            continue
        query = f"{heading} {topic}"
        for found_title in _search_wikipedia(client, query, limit=2):
            if found_title.startswith(("List of", "Outline of", "Index of")):
                continue
            if found_title.lower() not in seen:
                seen.add(found_title.lower())
                titles.append(found_title)
                if len(titles) >= limit:
                    break

    return titles


def _resolve_root(client: httpx.Client, topic: str) -> dict[str, Any] | None:
    """Fetch root article with progressive fallback."""
    root = _fetch_article(client, topic)
    if root is not None:
        return root

    logger.info("Wikipedia: no direct article for '%s', trying fallback", topic)
    words = [w for w in topic.split() if len(w) >= 3][:5]

    # Step 1: try individual words — first word is usually the primary subject noun,
    # then try longer words first (more specific) before shorter ones
    ordered = words[:1] + sorted(words[1:], key=len, reverse=True)
    for word in ordered:
        candidate = word.capitalize()
        root = _fetch_article(client, candidate)
        if root is not None:
            logger.info("Wikipedia: using '%s' as root for '%s'", candidate, topic)
            return root

    # Step 2: full-phrase search fallback
    for candidate in _search_wikipedia(client, topic, limit=5):
        if candidate.startswith(("List of", "Outline of", "Index of")):
            continue
        root = _fetch_article(client, candidate)
        if root is not None:
            logger.info("Wikipedia: using '%s' (search) as root for '%s'", candidate, topic)
            return root

    return None


def _article_to_document(article: dict[str, Any], topic_id: str) -> dict[str, Any]:
    """Convert a fetched article dict to a raw_materials row."""
    title = article["title"]
    sections = article["sections"]

    content_parts = [f"# {title}", ""]
    for sec in sections:
        content_parts.append(f"## {sec['title']}")
        content_parts.append(sec["content"])
        content_parts.append("")

    content = "\n".join(content_parts).strip()
    slug = title.lower().replace(" ", "_").replace("/", "_")

    return {
        "id": f"wiki_{slug}",
        "topic_id": topic_id,
        "source": _SOURCE,
        "url": article["url"],
        "title": title,
        "content": content,
    }


def _fetch_linked_article(item: dict[str, str]) -> Iterator[dict[str, Any]]:
    """Fetch one linked article and yield as a raw_materials row. Extracted for testability."""
    with httpx.Client(headers=_HEADERS) as client:
        article = _fetch_article(client, item["title"])
    if article is None:
        return
    if sum(len(s["content"]) for s in article["sections"]) < 300:
        return
    yield _article_to_document(article, item["topic_id"])


@dlt.source(name="wikipedia")
def wikipedia_source(topic: str, max_linked: int = 8) -> Iterator[Any]:
    """Yield Wikipedia articles (root + linked pages) as curriculum materials."""
    yield _wikipedia_resource(topic, max_linked)


@dlt.resource(name="raw_materials", primary_key="id", write_disposition="merge", parallelized=True)
def _wikipedia_resource(topic: str, max_linked: int = 8) -> Iterator[dict[str, Any]]:
    """Fetch root article then linked articles sequentially within this thread.

    Runs concurrently with other sources (arXiv, OpenStax, OTL, DOAB) via
    parallelized=True. Within-resource linked fetching is sequential — DLT
    does not support multiple same-named resources in one source.
    """
    with httpx.Client(headers=_HEADERS) as client:
        root = _resolve_root(client, topic)
        if root is None:
            logger.warning("Wikipedia: no article found for '%s'", topic)
            return

        yield _article_to_document(root, topic)

        linked_titles = _related_titles_from_sections(
            client, root["sections"], root["title"], limit=max_linked
        )
        logger.info(
            "Wikipedia: fetching %d section-derived articles for '%s'", len(linked_titles), topic
        )

        for linked_title in linked_titles:
            article = _fetch_article(client, linked_title)
            if article is None:
                continue
            if sum(len(s["content"]) for s in article["sections"]) < 300:
                continue
            yield _article_to_document(article, topic)
