"""Deciding when the model should say "I'm not sure".

A note is *uncertain* when the model's highest class probability is below a threshold.

Evaluation layout
-----------------
* The 12 out-of-scope notes are split once (seeded): 6 to choose the threshold, 6 held out.
* The threshold is chosen from two sets of confidences: validation.csv (the in-scope side,
  which the brief allows for this purpose) and the 6 tuning out-of-scope notes.
* So the validation flagged-rate is tuned-on, not an independent estimate. The independent
  checks are the 6 held-out out-of-scope notes and the grouped out-of-fold confidences on
  train.csv (a pessimistic bound, because those folds hold back whole templates).
"""
import numpy as np

from smartnotes.config import RANDOM_STATE

TUNE_SIZE = 6


def split_out_of_scope(texts: list[str], tune_size: int = TUNE_SIZE) -> tuple[list[str], list[str]]:
    """Deterministic (tune, held_out) split of the out-of-scope notes."""
    order = np.random.RandomState(RANDOM_STATE).permutation(len(texts))
    tune = [texts[i] for i in order[:tune_size]]
    held_out = [texts[i] for i in order[tune_size:]]
    return tune, held_out


def max_confidence(pipeline, texts: list[str]) -> np.ndarray:
    """Highest class probability for each text."""
    return pipeline.predict_proba(texts).max(axis=1)


def flagged_share(confidence: np.ndarray, threshold: float) -> float:
    """Fraction of notes that would be marked uncertain."""
    return float(np.mean(np.asarray(confidence) < threshold)) if len(confidence) else 0.0


def choose_threshold(in_scope_conf: np.ndarray, oos_conf: np.ndarray, max_in_scope_flagged: float) -> float:
    """Flag as many out-of-scope notes as possible while flagging at most
    `max_in_scope_flagged` of the in-scope notes.

    Candidates are the midpoints between neighbouring observed confidences. Ties on
    out-of-scope recall go to the setting that flags fewer in-scope notes, then the lower
    threshold.
    """
    values = np.unique(np.concatenate([in_scope_conf, oos_conf]))
    candidates = np.concatenate([[values[0] / 2], (values[:-1] + values[1:]) / 2])
    best = None
    for t in candidates:
        in_flagged = flagged_share(in_scope_conf, t)
        if in_flagged > max_in_scope_flagged:
            continue
        key = (-flagged_share(oos_conf, t), in_flagged, t)
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1] if best else float(candidates[0])


def is_uncertain(confidence: float, threshold: float) -> bool:
    return bool(confidence < threshold)
