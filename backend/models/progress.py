"""User progress models."""

from pydantic import BaseModel

from backend.models.concept import MasteryStatus


class ConceptProgress(BaseModel):
    concept_id: str
    status: MasteryStatus = MasteryStatus.LOCKED
    best_score: float | None = None
    attempts: int = 0


class UserProgress(BaseModel):
    topic_id: str
    concepts: dict[str, ConceptProgress]

    def unlocked_concept_ids(self) -> list[str]:
        """Concept IDs the user may access (not locked)."""
        return [
            cid
            for cid, cp in self.concepts.items()
            if cp.status
            in (MasteryStatus.UNLOCKED, MasteryStatus.IN_PROGRESS, MasteryStatus.MASTERED)
        ]
