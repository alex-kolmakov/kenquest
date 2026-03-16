"""Topic models."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TopicCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    description: str = ""


class Topic(BaseModel):
    id: str
    name: str
    description: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    concept_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
