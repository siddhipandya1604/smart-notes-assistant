"""The extended label model: knows more than the four required categories.

The four-label model in train.py is what `/classify` must return, so it stays as it is. This second
model is trained on train.csv plus data/extended_notes.csv and also knows Health, Family, Home and
Work, so a note like "gym at 6 PM" gets a real label instead of a bad Money guess. It is also
trained on more phrasings of the four core categories ("exam of ISRO tomorrow").

Held out and never trained on: data/extended_eval.csv (hand-labelled, includes the 12
out-of-scope notes) and validation.csv.

A prediction is shown with its label only when its probability reaches a threshold; below it the
label is "Other". The threshold is the lowest one at which grouped out-of-fold predictions on the
training data are at least 95% correct (the share of notes that keep a label is reported).
"""
import csv

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

from smartnotes.config import (EXTENDED_EVAL_CSV, EXTENDED_MODEL_PATH, EXTRA_NOTES_CSV, LABELS, TRAIN_CSV,
                               VALIDATION_CSV)
from smartnotes.ml.evaluate import score
from smartnotes.ml.grouping import grouped_folds
from smartnotes.ml.model import ModelBundle, ModelConfig, build_pipeline, save_bundle
from smartnotes.ml.preprocess import load_labeled_csv

OTHER = "Other"
# Grouped OOF predictions are pessimistic (held-out notes rarely have a close twin in training), so
# 90% precision there means the shown labels are right more often than that on ordinary notes.
TARGET_PRECISION = 0.90
CANDIDATES = [
    ModelConfig("naive bayes word+char", "nb", True, normalize_numbers=True, drop_stopwords=True),
    ModelConfig("naive bayes word only", "nb", False, normalize_numbers=True, drop_stopwords=True),
    ModelConfig("logreg C=10 word+char", "logreg", True, 10.0, True, True),
    ModelConfig("logreg C=100 word+char", "logreg", True, 100.0, True, True),
    ModelConfig("logreg C=100 word only", "logreg", False, 100.0, True, True),
]


def load_extended_training() -> tuple[list[str], list[str]]:
    """train.csv plus the extra notes."""
    texts, labels = load_labeled_csv(TRAIN_CSV)
    extra_texts, extra_labels = load_labeled_csv_with_label(EXTRA_NOTES_CSV)
    return texts + extra_texts, labels + extra_labels


def load_labeled_csv_with_label(path) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def load_extended_eval() -> list[dict]:
    with open(EXTENDED_EVAL_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def choose_label_threshold(confidence: np.ndarray, correct: np.ndarray, target: float = TARGET_PRECISION) -> float:
    """Lowest confidence cut-off at which the notes at or above it are at least `target` correct."""
    for t in sorted(set(confidence.round(6))):
        kept = confidence >= t
        if kept.sum() >= 10 and correct[kept].mean() >= target:
            return float(t)
    return 0.5


def apply_threshold(pipeline, texts: list[str], threshold: float) -> list[tuple[str, float]]:
    """(label shown, confidence) per text; the label is OTHER below the threshold."""
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    return [(classes[int(np.argmax(p))] if p.max() >= threshold else OTHER, float(p.max())) for p in proba]


def train_extended(verbose: bool = True) -> ModelBundle:
    texts, labels = load_extended_training()
    classes = sorted(set(labels))
    folds = grouped_folds(texts, labels)
    oof_by_config = {cfg.name: cross_val_predict(build_pipeline(cfg), texts, labels, cv=folds, method="predict_proba")
                     for cfg in CANDIDATES}
    cv = {name: cv_scores(p, labels, classes) for name, p in oof_by_config.items()}
    # Log-loss, not accuracy: the label threshold below is read off these probabilities.
    config = min(CANDIDATES, key=lambda c: cv[c.name]["log_loss"])
    oof = oof_by_config[config.name]
    oof_pred = np.array([classes[i] for i in oof.argmax(axis=1)])
    oof_conf = oof.max(axis=1)
    threshold = choose_label_threshold(oof_conf, oof_pred == np.array(labels))

    pipeline = build_pipeline(config).fit(texts, labels)
    bundle = ModelBundle(pipeline=pipeline, config=config, labels=tuple(classes), threshold=threshold,
                         metadata={"train_rows": len(texts), "classes": classes, "target_precision": TARGET_PRECISION})
    save_bundle(bundle, EXTENDED_MODEL_PATH)

    if verbose:
        print("\n== Extended label model: selection by grouped CV on train.csv + extended_notes.csv ==")
        for name, m in cv.items():
            print(f"{name:<26} accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  log-loss={m['log_loss']:.3f}")
        print(f"selected (lowest CV log-loss): {config.name}")
        print_report(bundle, texts, labels, oof_pred, oof_conf)
    return bundle


def cv_scores(proba: np.ndarray, labels: list[str], classes: list[str]) -> dict:
    predicted = [classes[i] for i in proba.argmax(axis=1)]
    return {**score(labels, predicted, label_set=classes), "log_loss": log_loss(labels, proba, labels=classes)}


def print_report(bundle: ModelBundle, texts, labels, oof_pred, oof_conf) -> None:
    t = bundle.threshold
    kept = oof_conf >= t
    print("\n== Extended label model (8 labels) ==")
    print(f"training rows: {len(texts)} (train.csv + {len(texts) - 156} extra)   labels: {', '.join(bundle.labels)}")
    print(f"grouped-CV on training data: {fmt(score(labels, list(oof_pred), label_set=list(bundle.labels)))}")
    print(f"label threshold {t:.4f}: {kept.mean():.0%} of CV notes keep a label, "
          f"{(oof_pred[kept] == np.array(labels)[kept]).mean():.0%} of those are correct")

    rows = load_extended_eval()
    shown = apply_threshold(bundle.pipeline, [r["text"] for r in rows], t)
    for source, title in (("new", "hand-written new notes"), ("out_of_scope", "the 12 out-of-scope notes")):
        subset = [(r, s) for r, s in zip(rows, shown) if r["source"] == source]
        right = sum(s[0] == r["label"] for r, s in subset)
        other = sum(s[0] == OTHER for _, s in subset)
        print(f"held-out {title} ({len(subset)}): {right}/{len(subset)} get the right label, {other} labelled Other")
        for r, (label, conf) in subset:
            if label != r["label"]:
                print(f"    wrong: {r['text']!r} expected {r['label']}, got {label} ({conf:.2f})")

    val_texts, val_labels = load_labeled_csv(VALIDATION_CSV)
    val = apply_threshold(bundle.pipeline, val_texts, t)
    right = sum(s[0] == label for (s, label) in zip(val, val_labels))
    other = sum(s[0] == OTHER for s in val)
    print(f"validation.csv (50, 4 core labels): {right}/50 get the right label, {other} labelled Other, "
          f"{50 - right - other} wrong")


def fmt(m: dict) -> str:
    return f"accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}"


CORE_LABELS = frozenset(LABELS)

if __name__ == "__main__":
    train_extended()
