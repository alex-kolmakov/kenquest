"""Concept domain models."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MasteryStatus(StrEnum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"


class Concept(BaseModel):
    id: str  # stable slug e.g. "cell-division"
    topic_id: str
    name: str
    description: str
    difficulty: int = Field(ge=1, le=5)
    source_refs: list[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PrerequisiteEdge(BaseModel):
    source_id: str  # prerequisite concept
    target_id: str  # dependent concept
    strength: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str = ""  # LLM-generated reason
