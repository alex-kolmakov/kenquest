"""DLT source: OpenStax open textbooks via GitHub.

OpenStax publishes all their books as CC BY 4.0 on GitHub (github.com/openstax).
Each book has:
  - collections/{slug}.collection.xml  — full chapter/section ToC tree
  - modules/{id}/index.cnxml           — CNXML content per section

Strategy:
  1. Maintain a curated map of topic keywords → OpenStax book repos.
  2. For a matching book, parse the collection XML to extract chapter titles
     (this becomes the curriculum skeleton).
  3. Fetch module CNXML for each chapter intro module and extract plain text.

Attribution: CC BY 4.0, OpenStax, Rice University.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any
from xml.etree.ElementTree import Element, ParseError

import dlt
import httpx
from defusedxml.ElementTree import fromstring as _xml_fromstring

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/openstax"
_API_BASE = "https://api.github.com/repos/openstax"
_HEADERS = {"User-Agent": "KenQuest/0.1 (educational platform; contact@kenquest.app)"}
_LICENSE = "CC BY 4.0"
_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_PUBLISHER = "OpenStax, Rice University"
_SOURCE = "openstax"

# Curated map: topic keyword → (repo_name, collection_slug, book_title)
# Only books with clear ecology/life-science relevance included.
_BOOK_MAP: list[tuple[str, str, str, str]] = [
    # (match_keyword, repo, collection_slug, human_title)
    ("biology", "osbooks-biology-bundle", "biology-2e", "Biology 2e"),
    ("marine", "osbooks-biology-bundle", "biology-2e", "Biology 2e"),
    ("ecology", "osbooks-biology-bundle", "biology-2e", "Biology 2e"),
    ("conservation", "osbooks-biology-bundle", "biology-2e", "Biology 2e"),
    ("microbiology", "osbooks-microbiology", "microbiology", "Microbiology"),
    ("chemistry", "osbooks-chemistry-bundle", "chemistry-2e", "Chemistry 2e"),
    (
        "physics",
        "osbooks-university-physics-bundle",
        "university-physics-volume-1",
        "University Physics Vol 1",
    ),
    (
        "anatomy",
        "osbooks-anatomy-and-physiology",
        "anatomy-and-physiology-2e",
        "Anatomy and Physiology 2e",
    ),
]

# Namespace constants for CNXML/collection XML
_COL_NS = "http://cnx.rice.edu/collxml"
_MD_NS = "http://cnx.rice.edu/mdml"
_CNX_NS = "http://cnx.rice.edu/cnxml"

# CNXML element tags to skip when extracting plain text (non-educational content)
_SKIP_TAGS = frozenset({"image", "media", "figure", "subfigure", "caption", "note", "glossary"})


def _find_matching_books(topic: str) -> list[tuple[str, str, str]]:
    """Return (repo, collection_slug, book_title) for books matching the topic."""
    topic_lower = topic.lower()
    seen: set[str] = set()
    matches = []
    for keyword, repo, slug, title in _BOOK_MAP:
        if keyword in topic_lower and slug not in seen:
            seen.add(slug)
            matches.append((repo, slug, title))
    return matches


def _fetch_collection_xml(client: httpx.Client, repo: str, slug: str) -> str | None:
    url = f"{_RAW_BASE}/{repo}/main/collections/{slug}.collection.xml"
    try:
        resp = client.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except httpx.RequestError as e:
        logger.warning("Failed to fetch collection XML %s: %s", url, e)
    return None


def _parse_chapters(xml_text: str) -> list[dict[str, Any]]:
    """
    Parse collection XML → list of {unit, chapter, module_ids}.

    Collection XML structure:
      <collection>
        <content>
          <subcollection>         ← unit
            <title>Unit 1</title>
            <content>
              <subcollection>     ← chapter
                <title>Ch 1</title>
                <content>
                  <module document="m12345"/>
                  ...
    """
    root = _xml_fromstring(xml_text)
    chapters = []

    for unit_el in root.findall(f".//{{{_COL_NS}}}subcollection"):
        unit_title_el = unit_el.find(f"{{{_MD_NS}}}title")
        unit_title = (
            unit_title_el.text.strip() if unit_title_el is not None and unit_title_el.text else ""
        )

        # Chapters are nested subcollections
        for chap_el in unit_el.findall(f"{{{_COL_NS}}}content/{{{_COL_NS}}}subcollection"):
            chap_title_el = chap_el.find(f"{{{_MD_NS}}}title")
            chap_title = (
                chap_title_el.text.strip()
                if chap_title_el is not None and chap_title_el.text
                else ""
            )

            module_ids = [
                m.get("document", "")
                for m in chap_el.findall(f".//{{{_COL_NS}}}module")
                if m.get("document")
            ]

            if chap_title and module_ids:
                chapters.append(
                    {
                        "unit": unit_title,
                        "chapter": chap_title,
                        "module_ids": module_ids,
                    }
                )

    return chapters


def _fetch_module_text(client: httpx.Client, repo: str, module_id: str) -> str:
    """Fetch a CNXML module and return clean plain text."""
    url = f"{_RAW_BASE}/{repo}/main/modules/{module_id}/index.cnxml"
    try:
        resp = client.get(url, timeout=15)
        if resp.status_code != 200:
            return ""
    except httpx.RequestError:
        return ""

    try:
        root = _xml_fromstring(resp.text)
    except ParseError:
        return ""

    # Extract text, skipping media/image/figure nodes
    def extract(el: Element) -> list[str]:
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in _SKIP_TAGS:
            return []
        parts = []
        if el.text:
            parts.append(el.text.strip())
        for child in el:
            parts.extend(extract(child))
        if el.tail:
            parts.append(el.tail.strip())
        return parts

    text = " ".join(p for p in extract(root) if p)
    return re.sub(r"\s+", " ", text).strip()


@dlt.source(name="openstax")
def openstax_source(topic: str) -> Iterator[Any]:
    """Yield all OpenStax textbook chapters matching the topic."""
    yield _openstax_resource(topic)


@dlt.resource(name="raw_materials", primary_key="id", write_disposition="merge", parallelized=True)
def _openstax_resource(topic: str, max_chapters: int = 0) -> Iterator[dict[str, Any]]:
    """Yield raw material dicts for OpenStax chapters matching *topic*.

    Args:
        topic: Topic keyword string for book matching.
        max_chapters: If >0, cap chapters fetched per book (useful in tests).
    """
    matching_books = _find_matching_books(topic)
    if not matching_books:
        logger.info("No OpenStax books matched topic '%s'", topic)
        return

    with httpx.Client(headers=_HEADERS) as client:
        for repo, slug, book_title in matching_books:
            xml_text = _fetch_collection_xml(client, repo, slug)
            if not xml_text:
                continue

            chapters = _parse_chapters(xml_text)
            if max_chapters > 0:
                chapters = chapters[:max_chapters]
            logger.info("OpenStax %s: %d chapters to fetch", book_title, len(chapters))

            total_chars = 0
            for chap in chapters:
                if not chap["module_ids"]:
                    continue

                # Fetch ALL modules in the chapter and concatenate into one document.
                # Previously only the first (intro) module was fetched — this captures
                # the full chapter text across all sections.
                module_texts = [_fetch_module_text(client, repo, mid) for mid in chap["module_ids"]]
                text = "\n\n".join(t for t in module_texts if t)

                if len(text) < 100:
                    continue

                chapter_title = chap["chapter"]
                unit_title = chap["unit"]
                # Stable ID derived from chapter title (not module id) so re-runs merge cleanly
                slug_title = re.sub(r"[^a-z0-9]+", "-", chapter_title.lower()).strip("-")
                doc_id = f"openstax_{slug}_{slug_title}"

                content = f"# {book_title} — {unit_title}: {chapter_title}\n\n{text}"
                total_chars += len(content)

                yield {
                    "id": doc_id,
                    "topic_id": topic,
                    "source": _SOURCE,
                    "url": f"https://openstax.org/books/{slug}/pages/{slug_title}",
                    "title": f"{book_title} — {chapter_title}",
                    "content": content,
                    "authors": "OpenStax Contributors",
                    "publisher": _PUBLISHER,
                    "license": _LICENSE,
                    "license_url": _LICENSE_URL,
                }

            logger.info(
                "OpenStax %s: complete — %d chapters, %d total chars",
                book_title,
                len(chapters),
                total_chars,
            )
