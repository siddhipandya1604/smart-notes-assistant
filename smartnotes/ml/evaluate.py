"""Metrics and printing helpers used by training and the docs."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from smartnotes.config import LABELS


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=list(LABELS), zero_division=0),
    }


def per_category_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=list(LABELS), digits=3, zero_division=0)


def confusion_table(y_true, y_pred) -> str:
    """Rows are the true label, columns the predicted label."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    width = max(len(label) for label in LABELS) + 2
    lines = ["true \\ pred".ljust(width + 2) + "".join(label.rjust(width) for label in LABELS)]
    for label, row in zip(LABELS, matrix):
        lines.append(label.ljust(width + 2) + "".join(str(n).rjust(width) for n in row))
    return "\n".join(lines)


def mistakes(texts, y_true, y_pred) -> list[tuple[str, str, str]]:
    return [(t, a, p) for t, a, p in zip(texts, y_true, y_pred) if a != p]
