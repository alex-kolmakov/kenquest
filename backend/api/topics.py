"""Topics API — CRUD + pipeline trigger."""

from fastapi import APIRouter

router = APIRouter(tags=["topics"])


@router.get("/topics")
async def list_topics() -> list[dict]:
    return []


@router.post("/topics", status_code=202)
async def create_topic(body: dict) -> dict:
    return {"message": "Topic creation not yet implemented"}


@router.get("/topics/{topic_id}")
async def get_topic(topic_id: str) -> dict:
    return {"topic_id": topic_id, "status": "not_implemented"}


@router.get("/topics/{topic_id}/status")
async def topic_status(topic_id: str) -> dict:
    return {"topic_id": topic_id, "status": "not_implemented"}
