"""What the app calls: search the current notes with the approach that won the evaluation."""
import threading

from smartnotes.search.base import Doc, Hit
from smartnotes.search.hybrid import HybridRetriever
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.models import Note


class NoteSearch:
    """Hybrid search (see hybrid.py). The index is rebuilt only when the saved notes change;
    at this scale (tens to hundreds of notes) a rebuild takes milliseconds."""

    def __init__(self, classifier: NoteClassifier):
        self._retriever = HybridRetriever(classifier)
        self._indexed: tuple | None = None
        self._lock = threading.Lock()  # requests run in a thread pool; fit + search must not interleave

    def search(self, query: str, notes: list[Note], limit: int = 10) -> list[Hit]:
        snapshot = tuple((n.id, n.text) for n in notes)
        with self._lock:
            if snapshot != self._indexed:
                self._retriever.fit([Doc(n.id, n.text) for n in notes])
                self._indexed = snapshot
            return self._retriever.search(query, limit)
