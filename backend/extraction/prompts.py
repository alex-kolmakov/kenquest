"""LLM prompt templates for concept extraction and graph building.

All prompts are versioned string constants — no dynamic prompt construction
outside of the explicit {placeholder} substitution in each function.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------

CONCEPT_EXTRACTION_SYSTEM = """\
You are a curriculum designer building a prerequisite knowledge graph.

Your task: extract DEFINITIONAL CONCEPTS — terms that have precise scientific or domain meanings
and that a student must understand before they can learn higher-level ideas.

A valid concept:
  - Has a clear, precise definition (not just a description of something that exists)
  - Is a general term, phenomenon, mechanism, process, or biological/chemical entity
  - Can appear as a prerequisite for OTHER concepts ("you need X to understand Y")
  - Is reusable across contexts — not tied to a single specific event, location, or individual
  - Is a building block: foundational concepts explain advanced ones

A valid concept IS things like:
  photosynthesis, symbiosis, coral bleaching, trophic cascade, calcification,
  zooxanthellae, ocean acidification, carrying capacity, keystone species,
  nutrient cycling, primary productivity, thermal stratification, upwelling,
  coral polyp, reef ecosystem, species diversity, population dynamics

A valid concept is NOT:
  - Proper nouns: specific places ("Great Barrier Reef"), people ("James Cook"), expeditions
  - Species counts or diversity statistics ("cetacean biodiversity", "mollusc species richness")
  - Historical events or dates ("1770 expedition", "Holocene reef formation")
  - Geographic or morphological labels ("deltaic reefs", "ribbon reefs", "wonky holes")
  - Administrative or legal entities ("Marine Park zones", "Reef Outlook Reports")
  - Chapter titles or summaries ("island plant propagation", "sea snake bathymetric range")

Respond ONLY with a JSON array. No commentary, no markdown fences.
Schema: [{{"name": str, "description": str (1-2 sentence precise definition), "difficulty": int (1-5), "source_refs": [str]}}]

difficulty scale:
  1 = foundational — high school level, no prior domain knowledge needed
  2 = early undergraduate — builds on a few foundational concepts
  3 = mid undergraduate — requires multiple prerequisite concepts
  4 = advanced — requires solid domain foundation
  5 = specialist / research-level
"""

CONCEPT_EXTRACTION_USER = """\
Source document id: {source_id}
Topic: {topic}

Text:
{text}

Extract up to {max_concepts} definitional concepts that are building blocks for understanding {topic}.
Focus on terms a student must be able to DEFINE and USE to understand the topic.
Skip historical facts, place names, species lists, and chapter-level summaries.
Return JSON array only.
"""

# ---------------------------------------------------------------------------
# Prerequisite edge inference
# ---------------------------------------------------------------------------

GRAPH_BUILDER_SYSTEM = """\
You are an expert educator building a prerequisite knowledge graph for a curriculum.

Your task: for EACH concept in the list, identify ALL other concepts in the list that
are its direct prerequisites — things a student must already understand before they can
learn this concept.

Think like a teacher writing a syllabus from scratch:
  - Every concept builds on prior knowledge. Find that prior knowledge.
  - If a student would struggle or be confused without knowing X first, X is a prerequisite.
  - Be thorough — a single concept may have 2-5 prerequisites from the list.
  - Do NOT be conservative. Missing a real dependency leaves gaps in the learning path.

Only omit an edge if the two concepts are genuinely independent (parallel topics).

Respond ONLY with a JSON array of edges. No commentary, no markdown fences.
Schema: [{{"source": str (prerequisite — must be learned FIRST),
          "target": str (dependent — requires the prerequisite),
          "strength": float (0.0-1.0, how essential: 1.0 = cannot understand without it),
          "rationale": str (one sentence: why source must come before target)}}]
"""

GRAPH_BUILDER_USER = """\
Topic: {topic}

Concepts (go through EACH one and ask: what from this list must be known first?):
{concepts_list}

Return all prerequisite edges as a JSON array.
Aim for density: most concepts should have at least one prerequisite from this list.
"""

# ---------------------------------------------------------------------------
# Cycle resolution
# ---------------------------------------------------------------------------

CYCLE_RESOLUTION_SYSTEM = """\
You are resolving a circular prerequisite dependency in a learning curriculum.
A cycle means the current graph has no valid learning order.

Given the cycle and all edges involved, identify the WEAKEST or LEAST ESSENTIAL
prerequisite edge to remove. Prefer removing the edge with the lowest strength score,
or the one whose rationale is most questionable.

Respond ONLY with a JSON object:
{{"remove_source": str, "remove_target": str, "reason": str}}
"""

CYCLE_RESOLUTION_USER = """\
Topic: {topic}

Cycle detected (sequence of concept names forming the loop):
{cycle_nodes}

All edges in or near the cycle:
{cycle_edges}

Which single edge should be removed to break this cycle?
"""
