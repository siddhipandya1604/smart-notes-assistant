"""Approach 1, the baseline: count the content words a note shares with the query."""
from smartnotes.search.base import Doc, Hit, content_words


class KeywordRetriever:
    name = "keyword overlap"

    def fit(self, docs: list[Doc]) -> "KeywordRetriever":
        self._docs = [(d.id, content_words(d.text)) for d in docs]
        return self

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        words = content_words(query)
        if not words:
            return []
        scored = [(len(words & doc_words) / len(words), doc_id) for doc_id, doc_words in self._docs]
        hits = [Hit(doc_id, score, "keyword") for score, doc_id in scored if score > 0]
        return sorted(hits, key=lambda h: -h.score)[:limit]  # stable sort: ties keep note order
