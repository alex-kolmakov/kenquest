"""Knowledge graph API endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["graph"])


@router.get("/topics/{topic_id}/graph")
async def get_graph(topic_id: str) -> dict:
    return {"nodes": [], "edges": [], "topic_id": topic_id}


@router.get("/topics/{topic_id}/graph/concept/{concept_id}")
async def get_concept(topic_id: str, concept_id: str) -> dict:
    return {"topic_id": topic_id, "concept_id": concept_id}


@router.post("/topics/{topic_id}/graph/validate")
async def validate_graph(topic_id: str) -> dict:
    return {"topic_id": topic_id, "cycles": []}
