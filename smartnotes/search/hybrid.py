"""Approach 3: TF-IDF plus the trained classifier's sense of *topic*.

The classifier that labels notes also labels the query. A query like "money transfer" is
recognised as Money, so Money notes ("Move ₹2,000 into the savings account") get a boost even
though they share no words with it.

    score = tf-idf cosine  +  TOPIC_WEIGHT * (how much query and note agree on category)

Guard rail: the topic boost is applied only when the classifier is *confident* about the query
(the same threshold that decides `uncertain` for notes). For a query it does not understand
("walk the dog") the boost is off, so only real word overlap can produce a result and we do not
present unrelated notes as matches.
"""
import numpy as np

from smartnotes.search.base import Doc, Hit
from smartnotes.search.tfidf import MIN_SCORE, TfidfRetriever
from smartnotes.services.classifier import NoteClassifier

TOPIC_WEIGHT = 0.5
TOPIC_MATCH = 0.25  # topic agreement contributes at least TOPIC_WEIGHT * this to count as a topic match


class HybridRetriever:
    name = "hybrid (tf-idf + classifier topic)"

    def __init__(self, classifier: NoteClassifier, topic_weight: float = TOPIC_WEIGHT, min_score: float = MIN_SCORE):
        self.classifier = classifier
        self.topic_weight = topic_weight
        self.min_score = min_score
        self._lexical = TfidfRetriever()

    def fit(self, docs: list[Doc]) -> "HybridRetriever":
        self._ids = [d.id for d in docs]
        self._lexical.fit(docs)
        self._note_topics = self.classifier.predict_proba([d.text for d in docs]) if docs else np.zeros((0, 0))
        return self

    def topic_scores(self, query: str) -> np.ndarray:
        """Agreement between the query's and each note's category probabilities, or zeros if the
        classifier is not confident about the query."""
        if not self._ids:
            return np.zeros(0)
        query_topic = self.classifier.predict_proba([query])[0]
        if query_topic.max() < self.classifier.threshold:
            return np.zeros(len(self._ids))
        return self._note_topics @ query_topic

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        lexical = self._lexical.lexical_scores(query)
        topic = self.topic_scores(query)
        scores = lexical + self.topic_weight * topic
        hits = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] < self.min_score:
                break
            has_words = lexical[i] >= self.min_score
            has_topic = self.topic_weight * topic[i] >= self.topic_weight * TOPIC_MATCH
            hits.append(Hit(self._ids[i], float(scores[i]), "both" if has_words and has_topic else "keyword" if has_words else "topic"))
        return hits[:limit]


class CategoryOnlyRetriever(HybridRetriever):
    """Ablation: ignore the words entirely and rank by topic agreement alone. Shows how much of
    the hybrid's hit rate is just 'guess the right category'."""

    name = "category only (ablation)"

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        topic = self.topic_scores(query)
        order = np.argsort(-topic, kind="stable")
        return [Hit(self._ids[i], float(topic[i]), "topic") for i in order[:limit] if topic[i] >= TOPIC_MATCH]
