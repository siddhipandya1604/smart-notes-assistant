"""SQLite-backed note storage. Notes (with their IDs, categories and uncertain markers) survive restarts.

ID policy: user notes get N013, N014, ... from a counter stored in the database. The counter only
moves forward, so an ID is never handed out twice even if the newest note is deleted, and the
starter IDs N001-N012 can never be taken by a user note.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from smartnotes.config import DB_PATH, FIRST_USER_NOTE_NUMBER
from smartnotes.services.classifier import Prediction
from smartnotes.store.models import Note

# One statement per entry: sqlite3's executescript() would commit our open transaction.
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS notes (
        num        INTEGER PRIMARY KEY,
        id         TEXT NOT NULL UNIQUE,
        text       TEXT NOT NULL,
        category   TEXT NOT NULL,
        uncertain  INTEGER NOT NULL,
        confidence REAL NOT NULL,
        source     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        label      TEXT NOT NULL DEFAULT '',
        label_confidence REAL NOT NULL DEFAULT 0
    )""",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NoteStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as conn:
            for statement in SCHEMA:
                conn.execute(statement)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
            for column, definition in (("label", "TEXT NOT NULL DEFAULT ''"),
                                       ("label_confidence", "REAL NOT NULL DEFAULT 0")):
                if column not in columns:  # database created before extended labels existed
                    conn.execute(f"ALTER TABLE notes ADD COLUMN {column} {definition}")
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('next_num', ?)", (FIRST_USER_NOTE_NUMBER,))

    # -- connections -------------------------------------------------------------------------
    @contextmanager
    def _transaction(self):
        """One short-lived connection per call keeps this safe under FastAPI's thread pool."""
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _read(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _to_note(row: sqlite3.Row) -> Note:
        return Note(id=row["id"], text=row["text"], category=row["category"], uncertain=bool(row["uncertain"]),
                    confidence=row["confidence"], source=row["source"], created_at=row["created_at"],
                    label=row["label"], label_confidence=row["label_confidence"])

    # -- writes ------------------------------------------------------------------------------
    def add(self, text: str, prediction: Prediction, source: str = "user") -> Note:
        """Save a user note under the next never-used ID."""
        with self._transaction() as conn:
            num = conn.execute("SELECT value FROM meta WHERE key = 'next_num'").fetchone()["value"]
            conn.execute("UPDATE meta SET value = ? WHERE key = 'next_num'", (num + 1,))
            note = Note(f"N{num:03d}", text, prediction.category, prediction.uncertain,
                        prediction.confidence, source, _now(), prediction.label, prediction.label_confidence)
            self._insert(conn, num, note)
        return note

    def add_with_ids(self, items: list[tuple[str, str, Prediction]], source: str) -> int:
        """Save notes under fixed IDs (the starter notes) in one all-or-nothing transaction.

        Items whose ID already exists are skipped, so repeating the call adds nothing.
        Returns how many notes were actually added."""
        added = 0
        with self._transaction() as conn:
            for note_id, text, prediction in items:
                if conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone():
                    continue
                self._insert(conn, int(note_id[1:]), Note(note_id, text, prediction.category, prediction.uncertain,
                                                          prediction.confidence, source, _now(), prediction.label,
                                                          prediction.label_confidence))
                added += 1
        return added

    @staticmethod
    def _insert(conn: sqlite3.Connection, num: int, note: Note) -> None:
        conn.execute(
            "INSERT INTO notes (num, id, text, category, uncertain, confidence, source, created_at, label, "
            "label_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (num, note.id, note.text, note.category, int(note.uncertain), note.confidence, note.source,
             note.created_at, str(note.label), float(note.label_confidence)),
        )

    def delete(self, note_id: str) -> bool:
        with self._transaction() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0

    # -- reads -------------------------------------------------------------------------------
    def get(self, note_id: str) -> Note | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._to_note(row) if row else None

    def backfill_labels(self, classifier) -> int:
        """Give notes saved before extended labels existed a label, and refresh their uncertain
        marker from the same prediction (it is a model output, and the model has since improved).
        The four-label category is left as it was. Returns how many notes were updated."""
        with self._read() as conn:
            rows = conn.execute("SELECT id, text FROM notes WHERE label = ''").fetchall()
        if not rows:
            return 0
        predictions = classifier.predict_many([r["text"] for r in rows])
        with self._transaction() as conn:
            for row, prediction in zip(rows, predictions):
                conn.execute("UPDATE notes SET label = ?, label_confidence = ?, uncertain = ? WHERE id = ?",
                             (str(prediction.label), float(prediction.label_confidence),
                              int(prediction.uncertain), row["id"]))
        return len(rows)

    def list(self, category: str | None = None, uncertain: bool | None = None,
             label: str | None = None) -> list[Note]:
        """Notes in ID order. `category` matches the four-label model's category (uncertain notes
        show up under their best guess); `label` matches the label the app shows; `uncertain`
        narrows to flagged / unflagged notes."""
        query, args = "SELECT * FROM notes WHERE 1 = 1", []
        if category is not None:
            query += " AND category = ?"
            args.append(category)
        if label is not None:
            query += " AND COALESCE(NULLIF(label, ''), category) = ?"
            args.append(label)
        if uncertain is not None:
            query += " AND uncertain = ?"
            args.append(int(uncertain))
        with self._read() as conn:
            rows = conn.execute(query + " ORDER BY num", args).fetchall()
        return [self._to_note(r) for r in rows]

    def count(self) -> int:
        with self._read() as conn:
            return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
