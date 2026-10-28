"""Shared types and helpers for the search approaches."""
import re
from dataclasses import dataclass
from typing import Protocol

from smartnotes.text import STOPWORDS

_TOKEN = re.compile(r"[a-z0-9₹]+")


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    match: str  # "keyword": shares words with the query, "topic": same subject, "both"


class Retriever(Protocol):
    name: str

    def fit(self, docs: list[Doc]) -> "Retriever": ...

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        """Relevant documents, best first. Fewer than `limit` (even none) when nothing matches well."""
        ...


def content_words(text: str) -> set[str]:
    """Lowercase words minus common ones like "my", "the", "is", "a", "when", "what"."""
    return {t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS}
