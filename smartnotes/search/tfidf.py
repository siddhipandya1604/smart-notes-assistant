"""Approach 2: TF-IDF cosine similarity over words and character n-grams.

Character n-grams give some tolerance to inflections and typos ("bills" ~ "bill"), but the query
still has to look like the note: it cannot bridge "money transfer" -> "Move ₹2,000 into savings".
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from smartnotes.search.base import Doc, Hit
from smartnotes.text import STOPWORDS

MIN_SCORE = 0.2  # cosine below this is treated as "not a match"


def _fit(vectorizer: TfidfVectorizer, texts: list[str]):
    """Fit one view. A corpus with nothing to index (no notes, or only "???" / "the") has an empty
    vocabulary, which scikit-learn reports as an error; that view then simply scores 0."""
    try:
        return vectorizer, vectorizer.fit_transform(texts)
    except ValueError:
        return None, None


def _cosine(vectorizer, matrix, query: str, size: int) -> np.ndarray:
    if vectorizer is None:
        return np.zeros(size)
    return (matrix @ vectorizer.transform([query]).T).toarray().ravel()


class TfidfRetriever:
    name = "tf-idf cosine"

    def __init__(self, min_score: float = MIN_SCORE):
        self.min_score = min_score

    def fit(self, docs: list[Doc]) -> "TfidfRetriever":
        self._ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        self._word = _fit(TfidfVectorizer(stop_words=sorted(STOPWORDS), sublinear_tf=True), texts)
        self._char = _fit(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), texts)
        return self

    def lexical_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query with every note: the mean of the word and char views."""
        size = len(self._ids)
        return 0.5 * _cosine(*self._word, query, size) + 0.5 * _cosine(*self._char, query, size)

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        scores = self.lexical_scores(query)
        order = np.argsort(-scores, kind="stable")
        return [Hit(self._ids[i], float(scores[i]), "keyword") for i in order[:limit] if scores[i] >= self.min_score]
