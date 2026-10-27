"""Load data/starter_notes.json (N001-N012) into the store.

    python -m smartnotes.store.starter

Safe to run any number of times: notes that already exist are skipped, never duplicated.
"""
import json
from dataclasses import dataclass

from smartnotes.config import STARTER_NOTES_JSON
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.notes import NoteStore


@dataclass(frozen=True)
class LoadResult:
    added: int
    skipped: int

    @property
    def total(self) -> int:
        return self.added + self.skipped


def read_starter_notes() -> list[dict]:
    with open(STARTER_NOTES_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_starter_notes(store: NoteStore, classifier: NoteClassifier) -> LoadResult:
    """Classify each starter note with the trained model and save it under its original ID."""
    notes = read_starter_notes()
    predictions = classifier.predict_many([n["text"] for n in notes])
    added = store.add_with_ids([(n["id"], n["text"], p) for n, p in zip(notes, predictions)], source="starter")
    return LoadResult(added=added, skipped=len(notes) - added)


def main() -> None:
    store = NoteStore()
    result = load_starter_notes(store, NoteClassifier.load())
    print(f"starter notes: {result.added} added, {result.skipped} already present ({store.path})")


if __name__ == "__main__":
    main()
