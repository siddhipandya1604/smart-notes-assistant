"""How the app's label and uncertain flag behave on held-out notes.  Run with:
    python -m smartnotes.ml.evaluate_labels

Uses the combined classifier (what /classify and the UI use), not a single model.
"""
import csv
import sys

from smartnotes.config import EXTENDED_FINAL_CSV, OUT_OF_SCOPE_CSV, VALIDATION_CSV
from smartnotes.ml.extended import OTHER, load_extended_eval
from smartnotes.ml.preprocess import load_labeled_csv
from smartnotes.services.classifier import NoteClassifier


def report(name: str, notes: list[tuple[str, str]], classifier: NoteClassifier, show_wrong: bool = True) -> dict:
    predictions = classifier.predict_many([t for t, _ in notes])
    right = [p.label == expected for p, (_, expected) in zip(predictions, notes)]
    print(f"{name} ({len(notes)}): label correct {sum(right)}/{len(notes)}   "
          f"labelled Other {sum(p.label == OTHER for p in predictions)}   uncertain {sum(p.uncertain for p in predictions)}")
    if show_wrong:
        for p, (text, expected), ok in zip(predictions, notes, right):
            if not ok:
                print(f"    {text!r}: expected {expected}, got {p.label} ({p.label_confidence:.2f}), "
                      f"category {p.category}, uncertain={p.uncertain}")
    return {"right": sum(right), "n": len(notes)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    classifier = NoteClassifier.load()

    val_texts, val_labels = load_labeled_csv(VALIDATION_CSV)
    predictions = classifier.predict_many(val_texts)
    print("== validation.csv: the four-label contract is unchanged ==")
    print(f"category accuracy {sum(p.category == y for p, y in zip(predictions, val_labels)) / 50:.3f}   "
          f"uncertain {sum(p.uncertain for p in predictions)}/50")
    report("validation, shown label", list(zip(val_texts, val_labels)), classifier, show_wrong=False)

    print("\n== held-out label checks ==")
    rows = load_extended_eval()
    report("development set, new notes", [(r["text"], r["label"]) for r in rows if r["source"] == "new"], classifier)
    report("development set, the 12 out-of-scope notes", [(r["text"], r["label"]) for r in rows if r["source"] == "out_of_scope"], classifier)
    with open(EXTENDED_FINAL_CSV, encoding="utf-8", newline="") as f:
        final = [(r["text"], r["label"]) for r in csv.DictReader(f)]
    report("FINAL set (written after the training data was frozen)", final, classifier)

    with open(OUT_OF_SCOPE_CSV, encoding="utf-8", newline="") as f:
        oos = [r["text"] for r in csv.DictReader(f)]
    flagged = sum(p.uncertain for p in classifier.predict_many(oos))
    print(f"\nout-of-scope notes flagged uncertain: {flagged}/{len(oos)}")


if __name__ == "__main__":
    main()
