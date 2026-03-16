"""Quiz domain models."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    id: str
    concept_id: str
    question: str
    expected_answer_summary: str  # used by scorer, not shown to user


class QuizAttempt(BaseModel):
    question: QuizQuestion
    user_answer: str
    score: float = Field(ge=0.0, le=1.0)
    feedback: str


class QuizSession(BaseModel):
    id: str
    concept_id: str
    questions: list[QuizQuestion]
    attempts: list[QuizAttempt] = []
    passed: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @property
    def avg_score(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(a.score for a in self.attempts) / len(self.attempts)
