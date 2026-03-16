"""Progress tracking API endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["progress"])


@router.get("/topics/{topic_id}/progress")
async def get_progress(topic_id: str) -> dict:
    return {"topic_id": topic_id, "concepts": {}}


@router.get("/topics/{topic_id}/progress/{concept_id}")
async def get_concept_progress(topic_id: str, concept_id: str) -> dict:
    return {"concept_id": concept_id, "status": "locked"}
