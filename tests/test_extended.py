import csv
import dataclasses
import sqlite3

from fastapi.testclient import TestClient

from smartnotes.api.app import create_app
from smartnotes.config import EXTENDED_EVAL_CSV, EXTENDED_FINAL_CSV, LABELS, VALIDATION_CSV
from smartnotes.ml.extended import OTHER, load_extended_training
from smartnotes.ml.preprocess import load_labeled_csv
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.notes import NoteStore

ISRO = "I have an exam of ISRO tomorrow"
GYM = "I have to go to gym today at 6:00 p.m."


def read_texts(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [r["text"] for r in csv.DictReader(f)]


def with_extended_threshold(classifier, threshold):
    extended = dataclasses.replace(classifier._extended, threshold=threshold)
    return NoteClassifier(classifier._bundle, extended)


# --- the reported problems --------------------------------------------------------------------
def test_exam_note_is_study_and_not_uncertain(classifier):
    p = classifier.predict(ISRO)
    assert (p.category, p.label, p.uncertain) == ("Study", "Study", False)


def test_gym_note_gets_a_health_label_instead_of_a_bad_core_guess(classifier):
    p = classifier.predict(GYM)
    assert p.label == "Health"
    assert p.category in LABELS            # the contract still returns one of the four
    assert p.uncertain is True             # it fits none of the four core categories


def test_classify_endpoint_returns_label_next_to_the_contract_fields(client):
    body = client.post("/classify", json={"text": GYM}).json()
    assert body["category"] in LABELS and isinstance(body["uncertain"], bool)
    assert body["label"] == "Health" and 0 <= body["label_confidence"] <= 1


def test_the_extended_model_knows_more_than_the_four_core_labels(classifier):
    assert set(classifier.all_labels) >= set(LABELS) | {"Health", "Family", "Home", "Work"}


# --- the four-label contract is untouched ------------------------------------------------------
def test_category_still_comes_from_the_four_label_model_alone(classifier):
    four_only = NoteClassifier(classifier._bundle, None)
    texts, _ = load_labeled_csv(VALIDATION_CSV)
    assert [p.category for p in classifier.predict_many(texts)] == [p.category for p in four_only.predict_many(texts)]


def test_validation_category_accuracy_is_unchanged(classifier):
    texts, labels = load_labeled_csv(VALIDATION_CSV)
    right = sum(p.category == y for p, y in zip(classifier.predict_many(texts), labels))
    assert right / len(labels) == 0.92


def test_without_the_extended_model_the_label_falls_back_to_the_category(classifier):
    p = NoteClassifier(classifier._bundle, None).predict("Confirm the bus seat for Pune.")
    assert p.label == p.category == "Travel"


# --- how the two models combine ----------------------------------------------------------------
def test_unsure_extended_model_falls_back_to_a_confident_core_category(classifier):
    p = with_extended_threshold(classifier, 1.01).predict("Confirm the bus seat for Pune.")
    assert (p.label, p.uncertain) == ("Travel", False)


def test_both_models_unsure_means_other_and_uncertain(classifier):
    p = with_extended_threshold(classifier, 1.01).predict("Call grandma this weekend.")
    assert (p.label, p.uncertain) == (OTHER, True)


# --- no peeking: evaluation notes are never in the training data ------------------------------
def test_evaluation_notes_are_not_training_notes():
    trained = {t.lower() for t in load_extended_training()[0]}
    for path in (EXTENDED_EVAL_CSV, EXTENDED_FINAL_CSV):
        assert not trained & {t.lower() for t in read_texts(path)}, path.name


def test_the_final_set_is_disjoint_from_the_development_set():
    assert not {t.lower() for t in read_texts(EXTENDED_FINAL_CSV)} & {t.lower() for t in read_texts(EXTENDED_EVAL_CSV)}


# --- storage and API ---------------------------------------------------------------------------
def test_label_is_saved_and_filterable(client):
    note = client.post("/api/notes", json={"text": GYM}).json()
    assert note["label"] == "Health"
    assert [n["id"] for n in client.get("/api/notes", params={"label": "Health"}).json()] == [note["id"]]
    assert client.get("/api/notes", params={"label": "Work"}).json() == []


def test_the_confidence_shown_belongs_to_the_label_shown(client):
    note = client.post("/api/notes", json={"text": GYM}).json()
    assert note["label"] == "Health"
    assert note["label_confidence"] > 0.9 and note["confidence"] < 0.6   # four-label model was unsure


def test_label_survives_a_restart(classifier, db_path):
    with TestClient(create_app(classifier, NoteStore(db_path))) as before:
        saved = before.post("/api/notes", json={"text": GYM}).json()
    with TestClient(create_app(classifier, NoteStore(db_path))) as after:
        assert after.get("/api/notes").json() == [saved]


def test_a_database_from_before_labels_existed_is_migrated_and_backfilled(classifier, db_path):
    old = sqlite3.connect(db_path)
    old.executescript("""
        CREATE TABLE notes (num INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, text TEXT NOT NULL,
            category TEXT NOT NULL, uncertain INTEGER NOT NULL, confidence REAL NOT NULL,
            source TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
        INSERT INTO meta VALUES ('next_num', 15);
        INSERT INTO notes VALUES (13, 'N013', 'I have to go to gym today at 6:00 p.m.', 'Money', 1, 0.47,
            'user', '2026-10-27T00:00:00+00:00');
        INSERT INTO notes VALUES (14, 'N014', 'i am having exam of isro tomorrow', 'Study', 1, 0.83,
            'user', '2026-10-27T00:01:00+00:00');
    """)
    old.commit()
    old.close()

    store = NoteStore(db_path)                       # opening migrates the schema
    assert store.get("N013").label == ""             # the old note has no label yet
    assert store.backfill_labels(classifier) == 2
    gym = store.get("N013")
    assert (gym.label, gym.category, gym.uncertain) == ("Health", "Money", True)      # category kept
    exam = store.get("N014")
    assert (exam.label, exam.category, exam.uncertain) == ("Study", "Study", False)   # stale marker refreshed
    assert store.backfill_labels(classifier) == 0    # nothing left to do
    assert store.add("Book a train to Pune.", classifier.predict("Book a train to Pune.")).id == "N015"
