"""KenQuest FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import graph, progress, topics, tutor
from backend.config import settings

app = FastAPI(
    title="KenQuest",
    description="Self-hosted AI learning platform with prerequisite-aware knowledge graphs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nginx reverse-proxy is the public boundary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(topics.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(tutor.router, prefix="/api")
app.include_router(progress.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}
