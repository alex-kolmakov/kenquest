"""Unit tests for ingestion sources (network mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from backend.ingestion.sources.arxiv import _arxiv_resource
from backend.ingestion.sources.doab import _build_attribution, _doab_resource, _extract_meta
from backend.ingestion.sources.openstax import _openstax_resource, _parse_chapters
from backend.ingestion.sources.opentextbook import (
    _normalise_license,
    _opentextbook_resource,
    _pick_url,
)
from backend.ingestion.sources.wikipedia import (
    _fetch_linked_article,
    _related_titles_from_sections,
    _split_into_sections,
    _wikipedia_resource,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def mock_resp(json_data: Any = None, status: int = 200, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    r.text = text
    r.raise_for_status = MagicMock()
    return r


# ── Wikipedia ─────────────────────────────────────────────────────────────────

WIKI_ARTICLE = {
    "query": {
        "pages": {
            "1": {
                "title": "Marine conservation",
                "fullurl": "https://en.wikipedia.org/wiki/Marine_conservation",
                "extract": (
                    "Marine conservation is the protection and preservation of ocean ecosystems "
                    "through deliberate action by governments, NGOs, and individuals.\n\n"
                    "== Threats ==\n"
                    "Major threats include overfishing, climate change, ocean acidification, "
                    "and plastic pollution affecting over 700 species of marine life globally.\n\n"
                    "== Marine Protected Areas ==\n"
                    "Marine protected areas (MPAs) restrict human activity in designated regions. "
                    "Roughly 8% of the world ocean has some form of protected status as of 2023.\n\n"
                    "== References ==\nSome refs.\n"
                ),
            }
        }
    }
}

WIKI_SEARCH = {"query": {"search": [{"title": "Coral reef"}, {"title": "Overfishing"}]}}

WIKI_LINKED = {
    "query": {
        "pages": {
            "2": {
                "title": "Coral reef",
                "fullurl": "https://en.wikipedia.org/wiki/Coral_reef",
                "extract": (
                    "A coral reef is an underwater ecosystem built by colonies of coral polyps "
                    "held together by calcium carbonate. Reefs are among the most biodiverse "
                    "ecosystems on Earth, supporting roughly 25% of all marine species.\n\n"
                    "== Threats ==\n"
                    "Rising ocean temperatures cause mass bleaching events, destroying reef "
                    "structure and triggering ecosystem collapse across large reef systems.\n"
                ),
            }
        }
    }
}


def test_split_sections_filters_references() -> None:
    text = (
        "Lead intro paragraph with enough content to qualify as introduction section.\n\n"
        "== Threats ==\n"
        "Overfishing and plastic pollution are major threats to marine ecosystems globally.\n\n"
        "== References ==\nRef 1.\n"
    )
    sections = _split_into_sections(text)
    titles = [s["title"] for s in sections]
    assert "Introduction" in titles
    assert "Threats" in titles
    assert "References" not in titles


@patch("backend.ingestion.sources.wikipedia.httpx.Client")
def test_wikipedia_resource_yields_root_doc(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.return_value = mock_resp(WIKI_ARTICLE)

    results = list(_wikipedia_resource("Marine conservation", max_linked=0))
    assert len(results) == 1
    root = results[0]
    assert root["id"] == "wiki_marine_conservation"
    assert "Threats" in root["content"]
    assert "References" not in root["content"]
    assert root["source"] == "wikipedia"


def test_related_titles_from_sections_returns_titles() -> None:
    client = MagicMock()
    client.get.side_effect = [
        mock_resp(WIKI_SEARCH),  # search for "Threats Marine conservation"
        mock_resp(WIKI_SEARCH),  # search for "Marine Protected Areas Marine conservation"
    ]

    sections = [
        {"title": "Threats", "content": "Some content"},
        {"title": "Marine Protected Areas", "content": "Some content"},
    ]
    titles = _related_titles_from_sections(client, sections, "Marine conservation", limit=2)
    assert len(titles) >= 1
    assert all(isinstance(t, str) for t in titles)


@patch("backend.ingestion.sources.wikipedia.httpx.Client")
def test_wikipedia_linked_fetcher_yields_doc(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.return_value = mock_resp(WIKI_LINKED)

    # Test the implementation function directly (_fetch_linked_article),
    # not the DLT-decorated transformer which cannot be called standalone.
    results = list(
        _fetch_linked_article({"title": "Coral reef", "topic_id": "marine conservation"})
    )
    assert len(results) == 1
    assert results[0]["source"] == "wikipedia"
    assert "Coral reef" in results[0]["title"]


@patch("backend.ingestion.sources.wikipedia.httpx.Client")
def test_wikipedia_missing_article_yields_nothing(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    missing = {"query": {"pages": {"-1": {"missing": ""}}}}
    # _resolve_root tries direct fetch + word fallbacks + search fallback — all return missing
    client.get.return_value = mock_resp(missing)
    assert list(_wikipedia_resource("NonExistentXYZ")) == []


# ── arXiv ─────────────────────────────────────────────────────────────────────

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Marine Conservation Biology: A Comprehensive Survey</title>
    <summary>
      This survey reviews the field of marine conservation biology covering threatened species,
      habitat degradation, marine protected areas, and evidence-based management strategies
      for ocean ecosystems. We discuss key challenges and emerging approaches including
      blue carbon accounting and international policy frameworks.
    </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Short Paper</title>
    <summary>Too short.</summary>
  </entry>
</feed>"""


@patch("backend.ingestion.sources.arxiv.httpx.Client")
def test_arxiv_filters_short_abstracts(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    resp = MagicMock()
    resp.text = ARXIV_XML
    resp.raise_for_status = MagicMock()
    client.get.return_value = resp

    results = list(_arxiv_resource("marine conservation", max_results=5))
    assert len(results) == 1
    assert results[0]["id"] == "arxiv_2301.00001v1"
    assert "Survey" in results[0]["content"]
    assert results[0]["source"] == "arxiv"


# ── OpenStax ──────────────────────────────────────────────────────────────────

COLLECTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://cnx.rice.edu/collxml"
            xmlns:md="http://cnx.rice.edu/mdml">
  <metadata mdml-version="0.5">
    <md:title>Biology 2e</md:title>
    <md:license url="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</md:license>
  </metadata>
  <content>
    <subcollection>
      <md:title>The Chemistry of Life</md:title>
      <content>
        <subcollection>
          <md:title>The Study of Life</md:title>
          <content>
            <module document="m66426"/>
            <module document="m66427"/>
          </content>
        </subcollection>
        <subcollection>
          <md:title>Biological Macromolecules</md:title>
          <content>
            <module document="m66437"/>
          </content>
        </subcollection>
      </content>
    </subcollection>
  </content>
</collection>"""

MODULE_CNXML = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="http://cnx.rice.edu/cnxml"
          xmlns:md="http://cnx.rice.edu/mdml">
  <metadata mdml-version="0.5">
    <md:title>The Study of Life</md:title>
  </metadata>
  <content>
    <para id="p1">Biology is the science that studies life. What exactly is life?
    This may sound like a silly question, but it is not always easy to define life.
    For example, a branch of biology called virology studies viruses which exhibit
    some characteristics of living entities but lack others.</para>
    <para id="p2">All living organisms share several key characteristics or functions:
    order, sensitivity or response to the environment, reproduction, growth and
    development, regulation, homeostasis, and energy processing.</para>
  </content>
</document>"""


def test_parse_chapters_extracts_structure() -> None:
    chapters = _parse_chapters(COLLECTION_XML)
    assert len(chapters) == 2
    titles = [c["chapter"] for c in chapters]
    assert "The Study of Life" in titles
    assert "Biological Macromolecules" in titles
    assert chapters[0]["unit"] == "The Chemistry of Life"
    assert "m66426" in chapters[0]["module_ids"]


@patch("backend.ingestion.sources.openstax.httpx.Client")
def test_openstax_yields_chapters(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client

    col_resp = mock_resp(text=COLLECTION_XML, status=200)
    mod_resp = mock_resp(text=MODULE_CNXML, status=200)
    client.get.side_effect = [
        col_resp,
        mod_resp,
        mod_resp,
        mod_resp,
    ]  # collection + 3 modules (2 in ch1, 1 in ch2)

    results = list(_openstax_resource("marine biology", max_chapters=2))
    assert len(results) >= 1
    row = results[0]
    assert row["source"] == "openstax"
    assert row["license"] == "CC BY 4.0"
    assert row["authors"] == "OpenStax Contributors"
    assert row["publisher"] == "OpenStax, Rice University"
    assert "Biology 2e" in row["title"]
    assert len(row["content"]) > 100


@patch("backend.ingestion.sources.openstax.httpx.Client")
def test_openstax_no_match_yields_nothing(mock_cls: MagicMock) -> None:
    # Topic with no keyword match
    results = list(_openstax_resource("quantum field theory", max_chapters=3))
    assert results == []
    mock_cls.assert_not_called()


# ── DOAB ──────────────────────────────────────────────────────────────────────

DOAB_RESPONSE = [
    {
        "uuid": "abc-123",
        "handle": "https://library.oapen.org/handle/abc",
        "metadata": [
            {"key": "dc.title", "value": "Marine Conservation Handbook"},
            {"key": "dc.contributor.author", "value": "Smith, Jane"},
            {"key": "dc.publisher", "value": "Ocean Press"},
            {"key": "dc.rights", "value": "open access"},
            {"key": "dc.rights.uri", "value": "https://creativecommons.org/licenses/by/4.0/"},
            {"key": "dc.language", "value": "en"},
            {"key": "dc.identifier", "value": "https://library.oapen.org/handle/abc"},
            {
                "key": "dc.description.abstract",
                "value": (
                    "This handbook provides a comprehensive overview of marine conservation "
                    "strategies, covering ecosystem-based management, marine protected areas, "
                    "fisheries regulation, and climate adaptation. Each chapter addresses "
                    "a specific conservation challenge with case studies from the Atlantic, "
                    "Pacific, and Indian Oceans. Aimed at practitioners and researchers alike."
                ),
            },
            {"key": "dc.subject", "value": "Marine biology; Conservation; Ocean policy"},
        ],
    },
    {
        "uuid": "short-1",
        "handle": "https://library.oapen.org/handle/short",
        "metadata": [
            {"key": "dc.title", "value": "Short Abstract Book"},
            {"key": "dc.language", "value": "en"},
            {"key": "dc.identifier", "value": "https://library.oapen.org/handle/short"},
            {"key": "dc.description.abstract", "value": "Too short."},
        ],
    },
]


def test_extract_meta_flattens_list() -> None:
    meta = _extract_meta(
        [
            {"key": "dc.title", "value": "Test Book"},
            {"key": "dc.language", "value": "en"},
        ]
    )
    assert meta["dc.title"] == "Test Book"


def test_build_attribution_normalises_cc_license() -> None:
    meta = {
        "dc.contributor.author": "Doe, John",
        "dc.publisher": "Test Press",
        "dc.rights.uri": "https://creativecommons.org/licenses/by/4.0/",
        "dc.rights": "open access",
    }
    attr = _build_attribution(meta)
    assert attr["license"] == "CC BY 4.0"
    assert attr["authors"] == "Doe, John"


@patch("backend.ingestion.sources.doab.httpx.Client")
def test_doab_yields_books(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.return_value = mock_resp(json_data=DOAB_RESPONSE)

    results = list(_doab_resource("marine conservation", max_books=4))
    assert len(results) == 1  # short abstract filtered
    row = results[0]
    assert row["source"] == "doab"
    assert row["id"] == "doab_abc-123"
    assert row["license"] == "CC BY 4.0"
    assert "Smith, Jane" in row["authors"]
    assert "Abstract" in row["content"]
    assert "Marine Conservation Handbook" in row["title"]


@patch("backend.ingestion.sources.doab.httpx.Client")
def test_doab_handles_api_error(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.side_effect = Exception("connection error")
    assert list(_doab_resource("marine conservation", max_books=4)) == []


DOAB_BARE_HANDLE_RESPONSE = [
    {
        "uuid": "bare-handle-1",
        "handle": "",
        "metadata": [
            {"key": "dc.title", "value": "Bare Handle Book"},
            {"key": "dc.contributor.author", "value": "Author, A"},
            {"key": "dc.language", "value": "en"},
            # dc.identifier is a bare handle, not a full URL
            {"key": "dc.identifier", "value": "20.500.12854/99999"},
            {
                "key": "dc.description.abstract",
                "value": (
                    "This book covers marine conservation in depth, examining ecosystem-based "
                    "management strategies and the role of international policy frameworks "
                    "in protecting ocean biodiversity across Atlantic, Pacific, and Indian Oceans."
                ),
            },
        ],
    }
]


@patch("backend.ingestion.sources.doab.httpx.Client")
def test_doab_normalises_bare_handle_url(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.return_value = mock_resp(json_data=DOAB_BARE_HANDLE_RESPONSE)

    results = list(_doab_resource("marine conservation", max_books=4))
    assert len(results) == 1
    assert results[0]["url"] == "https://library.oapen.org/handle/20.500.12854/99999"


# ── Open Textbook Library ─────────────────────────────────────────────────────

OTL_RESPONSE = [
    {
        "id": 101,
        "title": "Introduction to Geology",
        "description": (
            "This open textbook covers the fundamentals of physical geology including minerals, "
            "rocks, plate tectonics, geologic time, earthquakes, volcanoes, and surface processes. "
            "Designed for introductory undergraduate courses. Peer-reviewed by instructors at "
            "multiple institutions for comprehensiveness and accuracy."
        ),
        "license": "Attribution",
        "copyright_year": 2021,
        "formats": [
            {"type": "Online", "url": "https://geo.libretexts.org/intro-geology"},
            {"type": "PDF", "url": "https://geo.libretexts.org/intro-geology.pdf"},
        ],
        "publishers": [{"name": "LibreTexts"}],
        "subjects": [{"name": "Geology"}, {"name": "Earth Sciences"}],
    },
    {
        "id": 102,
        "title": "Short Book",
        "description": "Too short.",
        "license": "Attribution",
        "copyright_year": 2020,
        "formats": [],
        "publishers": [],
        "subjects": [],
    },
]


def test_normalise_license_cc_variants() -> None:
    assert _normalise_license("Attribution") == "CC BY"
    assert _normalise_license("Attribution-NonCommercial-ShareAlike") == "CC BY-NC-SA"
    assert _normalise_license("Attribution-ShareAlike") == "CC BY-SA"
    assert _normalise_license("CC0") == "CC0"
    assert _normalise_license("") == ""


def test_pick_url_prefers_online() -> None:
    formats = [
        {"type": "PDF", "url": "https://example.com/book.pdf"},
        {"type": "Online", "url": "https://example.com/book"},
    ]
    assert _pick_url(formats) == "https://example.com/book"


def test_pick_url_falls_back_to_pdf() -> None:
    formats = [{"type": "PDF", "url": "https://example.com/book.pdf"}]
    assert _pick_url(formats) == "https://example.com/book.pdf"


@patch("backend.ingestion.sources.opentextbook.httpx.Client")
def test_opentextbook_yields_books(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.return_value = mock_resp(json_data=OTL_RESPONSE)

    results = list(_opentextbook_resource("geology", max_books=4))
    assert len(results) == 1  # short description filtered
    row = results[0]
    assert row["id"] == "otl_101"
    assert row["source"] == "opentextbook"
    assert row["license"] == "CC BY"
    assert row["url"] == "https://geo.libretexts.org/intro-geology"
    assert "Geology" in row["content"]
    assert "Overview" in row["content"]
    assert row["publisher"] == "LibreTexts"


@patch("backend.ingestion.sources.opentextbook.httpx.Client")
def test_opentextbook_handles_api_error(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    client.get.side_effect = Exception("timeout")
    assert list(_opentextbook_resource("geology", max_books=4)) == []


@patch("backend.ingestion.sources.opentextbook.httpx.Client")
def test_opentextbook_constructs_url_when_formats_empty(mock_cls: MagicMock) -> None:
    """When no format URLs are provided, fall back to canonical OTL URL."""
    client = MagicMock()
    mock_cls.return_value.__enter__.return_value = client
    no_format_response = [
        {
            "id": 999,
            "title": "Geology Without Formats",
            "description": (
                "Covers all major topics in physical geology including plate tectonics, "
                "rock cycles, geologic time, surface processes, and mineralogy for "
                "introductory undergraduate geology courses."
            ),
            "license": "Attribution",
            "copyright_year": 2022,
            "formats": [],
            "publishers": [{"name": "Test Press"}],
            "subjects": [{"name": "Geology"}],
        }
    ]
    client.get.return_value = mock_resp(json_data=no_format_response)

    results = list(_opentextbook_resource("geology", max_books=4))
    assert len(results) == 1
    assert results[0]["url"] == "https://open.umn.edu/opentextbooks/textbooks/999"
