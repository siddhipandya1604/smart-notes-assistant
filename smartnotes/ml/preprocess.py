"""CSV loading and text normalisation."""
import csv
import re
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d[\d,.]*")


def load_labeled_csv(path: Path) -> tuple[list[str], list[str]]:
    """Read a `text,category` file and return (texts, labels)."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def load_texts_csv(path: Path) -> list[str]:
    """Read a file that only has a `text` column (the out-of-scope notes)."""
    with open(path, encoding="utf-8", newline="") as f:
        return [r["text"] for r in csv.DictReader(f)]


def normalize(text: str, numbers: bool = False) -> str:
    """Lowercase and collapse whitespace.

    With numbers=True every number (7,500 / 25 / 2.5) becomes the token "num", so
    amounts and dates stop acting as one-off features.
    """
    text = _WHITESPACE.sub(" ", text.lower()).strip()
    if numbers:
        text = _NUMBER.sub(" num ", text)
        text = _WHITESPACE.sub(" ", text).strip()
    return text
