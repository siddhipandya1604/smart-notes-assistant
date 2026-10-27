"""Baselines the trained model has to beat. Neither is used by the app."""
import re
from collections import Counter

from smartnotes.config import LABELS

# Deliberately naive: only words that appear in the "Includes" column of CATEGORY_GUIDE.md,
# plus their plain inflections. Not tuned on train.csv or validation.csv, so this is a fair
# "what you get from the category definitions alone" reference point.
KEYWORD_RULES = {
    "Study": ["exam", "course", "tutorial", "assignment", "read"],
    "Money": ["bill", "budget", "bank", "payment", "pay", "saving", "invest", "insurance"],
    "Travel": ["trip", "reservation", "train", "bus", "flight", "hotel", "itinerary"],
    "Shopping": ["buy", "order", "compar", "return"],
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
