"""Topics API — CRUD + pipeline trigger."""

from fastapi import APIRouter, Depends

from backend.db.duckdb_client import db_dependency

router = APIRouter(tags=["topics"])


@router.get("/topics")
async def list_topics(conn=Depends(db_dependency)) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, status, concept_count FROM topics ORDER BY name"
    ).fetchall()
    return [{"id": r[0], "name": r[1], "status": r[2], "concept_count": r[3] or 0} for r in rows]


@router.post("/topics", status_code=202)
async def create_topic(body: dict) -> dict:
    return {"message": "Topic creation not yet implemented"}


@router.get("/topics/{topic_id}")
async def get_topic(topic_id: str) -> dict:
    return {"topic_id": topic_id, "status": "not_implemented"}


@router.get("/topics/{topic_id}/status")
async def topic_status(topic_id: str) -> dict:
    return {"topic_id": topic_id, "status": "not_implemented"}
