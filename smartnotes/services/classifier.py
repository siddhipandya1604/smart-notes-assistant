"""Serving wrapper around the trained model bundle. Load once, call many times."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartnotes.config import MODEL_PATH
from smartnotes.ml.model import ModelBundle, load_bundle
from smartnotes.ml.uncertainty import is_uncertain


@dataclass(frozen=True)
class Prediction:
    category: str
    uncertain: bool
    confidence: float                    # highest class probability, rounded for display
    probabilities: dict[str, float]      # every class, rounded for display


class NoteClassifier:
    def __init__(self, bundle: ModelBundle):
        self._bundle = bundle
        self._classes = [str(c) for c in bundle.pipeline.classes_]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "NoteClassifier":
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `python train.py` first.")
        return cls(load_bundle(path))

    @property
    def threshold(self) -> float:
        return self._bundle.threshold

    @property
    def model_name(self) -> str:
        return self._bundle.config.name

    @property
    def labels(self) -> list[str]:
        return list(self._classes)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw class probabilities, columns ordered as `labels`."""
        return self._bundle.pipeline.predict_proba(texts)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        results = []
        for row in self.predict_proba(texts):
            best = int(np.argmax(row))
            results.append(Prediction(
                category=self._classes[best],
                uncertain=is_uncertain(float(row[best]), self.threshold),  # uses the unrounded value
                confidence=round(float(row[best]), 4),
                probabilities={c: round(float(p), 4) for c, p in zip(self._classes, row)},
            ))
        return results

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]
