"""Serving wrapper around the trained model bundles. Load once, call many times.

Two models work together:

* the four-label model (train.py): its best guess is the contract's `category`, unchanged;
* the extended model (smartnotes/ml/extended.py): also knows Health, Family, Home and Work, and
  gives the friendlier `label` shown in the app, plus a second opinion on `uncertain`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartnotes.config import EXTENDED_MODEL_PATH, MODEL_PATH
from smartnotes.ml.extended import OTHER
from smartnotes.ml.model import ModelBundle, load_bundle
from smartnotes.ml.uncertainty import is_uncertain


@dataclass(frozen=True)
class Prediction:
    category: str                        # one of the four contract labels
    uncertain: bool
    confidence: float                    # highest four-label probability, rounded for display
    probabilities: dict[str, float]      # the four-label probabilities, rounded for display
    label: str = ""                      # what the app shows: core label, Health/Family/Home/Work, or Other
    label_confidence: float = 0.0

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", self.category)


class NoteClassifier:
    def __init__(self, bundle: ModelBundle, extended: ModelBundle | None = None):
        self._bundle = bundle
        self._extended = extended
        self._classes = [str(c) for c in bundle.pipeline.classes_]

    @classmethod
    def load(cls, path: Path = MODEL_PATH, extended_path: Path = EXTENDED_MODEL_PATH) -> "NoteClassifier":
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `python train.py` first.")
        extended = load_bundle(extended_path) if extended_path.exists() else None
        return cls(load_bundle(path), extended)

    @property
    def threshold(self) -> float:
        return self._bundle.threshold

    @property
    def model_name(self) -> str:
        return self._bundle.config.name

    @property
    def labels(self) -> list[str]:
        return list(self._classes)

    @property
    def all_labels(self) -> list[str]:
        """Every label the app can show (without Other)."""
        return list(self._extended.labels) if self._extended else self.labels

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw four-label probabilities, columns ordered as `labels`."""
        return self._bundle.pipeline.predict_proba(texts)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        core = self.predict_proba(texts)
        extended = self._extended.pipeline.predict_proba(texts) if self._extended else None
        results = []
        for i, row in enumerate(core):
            best = int(np.argmax(row))
            category, confidence = self._classes[best], float(row[best])
            core_unsure = is_uncertain(confidence, self.threshold)  # uses the unrounded value
            label, label_conf, uncertain = category, confidence, core_unsure
            if extended is not None:
                label, label_conf, uncertain = self._combine(category, confidence, core_unsure, extended[i])
            results.append(Prediction(
                category=category,
                uncertain=uncertain,
                confidence=round(confidence, 4),
                probabilities={c: round(float(p), 4) for c, p in zip(self._classes, row)},
                label=label,
                label_confidence=round(label_conf, 4),
            ))
        return results

    def _combine(self, category: str, confidence: float, core_unsure: bool, ext_row: np.ndarray):
        """Merge the two models' opinions into (label, label confidence, uncertain).

        * The extended model is confident -> its label is shown. If that label is one of the four
          core ones and matches `category`, the note is not uncertain; if it is Health, Home, etc.,
          the note fits none of the four categories, so it is uncertain; a core label that
          disagrees with `category` is also uncertain.
        * Otherwise, if the four-label model is confident, its category is the label.
        * Otherwise the label is Other and the note is uncertain.
        """
        classes = [str(c) for c in self._extended.pipeline.classes_]  # plain str, not numpy.str_
        top = int(np.argmax(ext_row))
        ext_label, ext_conf = classes[top], float(ext_row[top])
        if ext_conf >= self._extended.threshold:
            return ext_label, ext_conf, not (ext_label in self._classes and ext_label == category)
        if not core_unsure:
            return category, confidence, False
        return OTHER, ext_conf, True

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]
