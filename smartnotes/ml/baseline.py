"""Baselines the trained model has to beat. Neither is used by the app."""
import re
from collections import Counter

from smartnotes.config import LABELS

# Written from the category definitions in CATEGORY_GUIDE.md, not from the validation set.
KEYWORD_RULES = {
    "Study": ["exam", "test", "quiz", "assignment", "homework", "lecture", "course",
              "class", "study", "revise", "chapter", "professor", "tutorial", "syllabus"],
    "Money": ["bill", "pay", "payment", "bank", "savings", "budget", "salary",
              "insurance", "rent", "expenses", "credit", "balance", "transfer"],
    "Travel": ["flight", "train", "bus", "hotel", "trip", "airport", "railway",
               "journey", "itinerary", "boarding", "luggage", "cab", "sightseeing"],
    "Shopping": ["buy", "order", "cart", "sale", "discount", "coupon", "return",
                 "store", "supermarket", "delivery", "groceries", "exchange", "pickup"],
}


class MajorityBaseline:
    """Always predicts the most frequent training label."""

    def fit(self, texts, labels):
        self.label_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.label_] * len(texts)


class KeywordRulesBaseline:
    """Predicts the category with the most keyword hits; ties and no hits go to the first label."""

    def fit(self, texts=None, labels=None):
        return self

    def predict(self, texts):
        return [self._predict_one(t) for t in texts]

    @staticmethod
    def _predict_one(text: str) -> str:
        words = re.findall(r"[a-z]+", text.lower())
        scores = {
            label: sum(any(w.startswith(k) for k in KEYWORD_RULES[label]) for w in words)
            for label in LABELS
        }
        return max(LABELS, key=lambda label: scores[label])
