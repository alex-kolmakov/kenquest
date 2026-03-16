"""DLT source: arXiv papers (abstracts via the Atom API).

Query strategy:
  - Search title AND abstract for the exact topic phrase (quoted).
  - Prefer survey/review papers which give structured field overviews.
  - Fall back to general relevance search if survey query yields nothing.

arXiv Atom API field syntax:
  ti:"phrase"   — title contains phrase (exact)
  abs:"phrase"  — abstract contains phrase (exact)
  OR / AND      — boolean operators (must be uppercase)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from xml.etree import ElementTree as ET

import dlt
import httpx

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://export.arxiv.org/api/query"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_MIN_ABSTRACT_LEN = 150
_SOURCE = "arxiv"


def _build_query(topic: str, prefer_surveys: bool) -> str:
    """Build a properly quoted arXiv Atom API query string."""
    # Quote the topic phrase so multi-word topics are treated as a phrase
    topic_clause = f'ti:"{topic}" OR abs:"{topic}"'
    if prefer_surveys:
        # AND with survey/review to prefer structured overview papers
        return f"({topic_clause}) AND (ti:survey OR ti:review OR ti:introduction OR ti:tutorial)"
    return topic_clause


def _parse_entries(xml_text: str, topic: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("atom:entry", _NS):
        arxiv_id_el = entry.find("atom:id", _NS)
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)

        if arxiv_id_el is None or title_el is None or summary_el is None:
            continue

        arxiv_url = arxiv_id_el.text or ""
        arxiv_id = arxiv_url.rstrip("/").split("/")[-1]
        title = (title_el.text or "").strip().replace("\n", " ")
        abstract = (summary_el.text or "").strip().replace("\n", " ")

        if len(abstract) < _MIN_ABSTRACT_LEN:
            continue

        results.append(
            {
                "id": f"arxiv_{arxiv_id}",
                "topic_id": topic,
                "source": _SOURCE,
                "url": arxiv_url,
                "title": title,
                "content": f"# {title}\n\n{abstract}",
            }
        )
    return results


@dlt.source(name="arxiv")
def arxiv_source(topic: str, max_results: int = 5) -> Iterator[Any]:
    """Yield arXiv paper abstracts (survey/review papers preferred)."""
    yield _arxiv_resource(topic, max_results)


@dlt.resource(name="raw_materials", primary_key="id", write_disposition="merge", parallelized=True)
def _arxiv_resource(topic: str, max_results: int = 5) -> Iterator[dict[str, Any]]:
    with httpx.Client(headers=_HEADERS) as client:
        # Pass 1: prefer survey/review papers
        query = _build_query(topic, prefer_surveys=True)
        try:
            resp = client.get(
                _SEARCH_URL,
                params={"search_query": query, "max_results": max_results, "sortBy": "relevance"},
                timeout=20,
            )
            resp.raise_for_status()
            results = _parse_entries(resp.text, topic)
        except Exception as e:
            logger.warning("arXiv survey search failed for '%s': %s", topic, e)
            results = []

        # Pass 2: if surveys yielded nothing, fall back to general relevance
        if not results:
            query = _build_query(topic, prefer_surveys=False)
            try:
                resp = client.get(
                    _SEARCH_URL,
                    params={
                        "search_query": query,
                        "max_results": max_results,
                        "sortBy": "relevance",
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                results = _parse_entries(resp.text, topic)
            except Exception as e:
                logger.warning("arXiv general search failed for '%s': %s", topic, e)

    yield from results
