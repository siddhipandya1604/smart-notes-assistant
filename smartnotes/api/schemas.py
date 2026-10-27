"""Request and response models. Validation here is what turns bad input into a 4xx."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from smartnotes.config import LABELS, MAX_TEXT_CHARS


class NoteText(BaseModel):
    """A note body: a real JSON string with some non-whitespace content, at most 1,000 characters."""

    model_config = ConfigDict(str_max_length=MAX_TEXT_CHARS)

    text: StrictStr  # strict: 42, null and lists are rejected instead of being coerced

    @field_validator("text")
    @classmethod
    def must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Note text cannot be empty.")
        return value.strip()


class ClassifyRequest(NoteText):
    pass


class ClassifyResponse(BaseModel):
    category: str
    uncertain: bool
    confidence: float
    probabilities: dict[str, float]


Category = Literal[*LABELS]  # the four allowed labels, exact spelling


class NoteOut(BaseModel):
    id: str
    text: str
    category: Category
    uncertain: bool
    confidence: float
    source: str
    created_at: str


class LoadStarterResponse(BaseModel):
    added: int
    skipped: int
    total: int
