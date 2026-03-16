"""Tutor API — helper Q&A and quiz endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["tutor"])


@router.post("/topics/{topic_id}/tutor/ask")
async def ask_helper(topic_id: str, body: dict) -> dict:
    return {"message": "Helper not yet implemented"}


@router.post("/topics/{topic_id}/tutor/quiz/start")
async def start_quiz(topic_id: str, body: dict) -> dict:
    return {"message": "Quiz not yet implemented"}


@router.post("/topics/{topic_id}/tutor/quiz/{session_id}/answer")
async def submit_answer(topic_id: str, session_id: str, body: dict) -> dict:
    return {"message": "Answer submission not yet implemented"}


@router.get("/topics/{topic_id}/tutor/quiz/{session_id}")
async def get_quiz_session(topic_id: str, session_id: str) -> dict:
    return {"session_id": session_id}
