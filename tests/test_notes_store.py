import json

import pytest
from fastapi.testclient import TestClient

from smartnotes.api.app import create_app
from smartnotes.config import FIRST_USER_NOTE_NUMBER, LABELS, STARTER_NOTES_JSON
from smartnotes.store.notes import NoteStore
from smartnotes.store.starter import load_starter_notes

STARTER_IDS = [f"N{i:03d}" for i in range(1, 13)]


def starter_notes():
    return json.loads(STARTER_NOTES_JSON.read_text(encoding="utf-8"))


def test_first_user_id_follows_the_starter_ids():
    """Guards FIRST_USER_NOTE_NUMBER against drifting from the starter file."""
    assert [n["id"] for n in starter_notes()] == STARTER_IDS
    assert FIRST_USER_NOTE_NUMBER == 13


# --- persistence ------------------------------------------------------------------------------
def test_notes_survive_a_restart(classifier, db_path):
    first = NoteStore(db_path)
    note = first.add("Call grandma this weekend.", classifier.predict("Call grandma this weekend."))
    other = first.add("Book a train to Pune.", classifier.predict("Book a train to Pune."))

    reopened = NoteStore(db_path)                      # a fresh object on the same file = a restart
    assert reopened.get(note.id) == note               # id, text, category, uncertain marker, confidence
    assert reopened.get(other.id) == other
    assert [n.id for n in reopened.list()] == [note.id, other.id]


def test_uncertain_note_is_saved_with_its_marker_and_best_guess(classifier, store):
    note = store.add("Call grandma this weekend.", classifier.predict("Call grandma this weekend."))
    assert note.uncertain is True
    assert note.category in LABELS


# --- IDs --------------------------------------------------------------------------------------
def test_user_notes_start_after_the_reserved_starter_ids(classifier, store):
    prediction = classifier.predict("Buy a pair of shoes.")
    ids = [store.add("Buy a pair of shoes.", prediction).id for _ in range(3)]
    assert ids == ["N013", "N014", "N015"]
    assert not set(ids) & set(STARTER_IDS)


def test_ids_are_never_reused_after_a_delete(classifier, store):
    prediction = classifier.predict("Buy a pair of shoes.")
    store.add("a", prediction)
    last = store.add("b", prediction)
    assert store.delete(last.id)
    assert store.add("c", prediction).id != last.id      # N015, not N014 again


def test_id_counter_survives_a_restart(classifier, db_path):
    prediction = classifier.predict("Buy a pair of shoes.")
    NoteStore(db_path).add("a", prediction)
    assert NoteStore(db_path).add("b", prediction).id == "N014"


# --- starter notes ----------------------------------------------------------------------------
def test_starter_notes_load_with_original_ids(classifier, store):
    result = load_starter_notes(store, classifier)
    assert (result.added, result.skipped) == (12, 0)
    saved = {n.id: n.text for n in store.list()}
    assert saved == {n["id"]: n["text"] for n in starter_notes()}


def test_loading_starter_notes_twice_creates_no_duplicates(classifier, store):
    load_starter_notes(store, classifier)
    again = load_starter_notes(store, classifier)
    assert (again.added, again.skipped) == (0, 12)
    assert store.count() == 12


def test_starter_load_does_not_disturb_user_notes(classifier, store):
    user = store.add("Book a train to Pune.", classifier.predict("Book a train to Pune."))
    load_starter_notes(store, classifier)
    assert store.count() == 13
    assert store.get(user.id).text == "Book a train to Pune."
    assert user.id == "N013"


def test_deleted_starter_note_comes_back_on_reload(classifier, store):
    load_starter_notes(store, classifier)
    store.delete("N003")
    assert load_starter_notes(store, classifier).added == 1
    assert store.get("N003") is not None


# --- filtering --------------------------------------------------------------------------------
def test_filter_by_category_includes_uncertain_notes_under_their_best_guess(classifier, store):
    load_starter_notes(store, classifier)
    grandma = store.add("Call grandma this weekend.", classifier.predict("Call grandma this weekend."))
    shown = store.list(category=grandma.category)
    assert grandma.id in {n.id for n in shown}
    assert all(n.category == grandma.category for n in shown)
    assert {n.id for n in store.list(uncertain=True)} >= {grandma.id}


# --- HTTP API ---------------------------------------------------------------------------------
def test_add_and_list_notes_over_http(client):
    created = client.post("/api/notes", json={"text": "Confirm the bus seat for Pune."})
    assert created.status_code == 201
    note = created.json()
    assert note["id"] == "N013" and note["category"] == "Travel" and note["uncertain"] is False

    listed = client.get("/api/notes").json()
    assert [n["id"] for n in listed] == ["N013"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"note": "x"}, {"text": 5}, {"text": None}])
def test_empty_or_invalid_note_is_rejected_with_a_message(client, payload):
    r = client.post("/api/notes", json=payload)
    assert r.status_code == 422 and r.json()["message"]
    assert client.get("/api/notes").json() == []         # nothing was saved


def test_http_filters(client):
    client.post("/api/notes/load-starter")
    study = client.get("/api/notes", params={"category": "Study"}).json()
    assert study and all(n["category"] == "Study" for n in study)
    assert client.get("/api/notes", params={"category": "Gardening"}).status_code == 422


def test_load_starter_over_http_is_idempotent(client):
    first = client.post("/api/notes/load-starter").json()
    second = client.post("/api/notes/load-starter").json()
    assert first == {"added": 12, "skipped": 0, "total": 12}
    assert second == {"added": 0, "skipped": 12, "total": 12}
    assert len(client.get("/api/notes").json()) == 12


def test_delete_and_missing_note(client):
    note = client.post("/api/notes", json={"text": "Book a train to Pune."}).json()
    assert client.delete(f"/api/notes/{note['id']}").status_code == 204
    missing = client.get(f"/api/notes/{note['id']}")
    assert missing.status_code == 404 and missing.json()["message"]
    assert client.delete("/api/notes/N999").status_code == 404


def test_notes_survive_restarting_the_app(classifier, db_path):
    """Two separate app instances on one database file behave like a server restart."""
    with TestClient(create_app(classifier, NoteStore(db_path))) as before:
        saved = before.post("/api/notes", json={"text": "Call grandma this weekend."}).json()
    with TestClient(create_app(classifier, NoteStore(db_path))) as after:
        assert after.get("/api/notes").json() == [saved]
