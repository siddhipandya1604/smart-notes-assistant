import pytest
from fastapi.testclient import TestClient

from smartnotes.api.app import create_app
from smartnotes.config import LABELS, OUT_OF_SCOPE_CSV
from smartnotes.ml.preprocess import load_texts_csv
from tests.conftest import with_threshold


def classify(client, text):
    return client.post("/classify", json={"text": text})


def test_normal_note_returns_valid_category_and_flag(client):
    r = classify(client, "Confirm the bus seat for Pune.")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Travel"
    assert body["category"] in LABELS
    assert isinstance(body["uncertain"], bool)
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(LABELS)


@pytest.mark.parametrize("body", [
    b'{"text": "   "}',          # whitespace only
    b'{"text": ""}',             # empty
    b'{"note": "Pay rent"}',     # missing field
    b'{"text": 42}',             # not a string
    b'{"text": null}',           # null
    b'{"text": ["Pay rent"]}',   # list
    b'{"text": ',                # malformed JSON
    b'["Pay rent"]',             # JSON but not an object
    b'',                         # no body at all
])
def test_invalid_input_is_a_4xx_and_server_keeps_working(client, body):
    r = client.post("/classify", content=body, headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500
    assert r.json()["message"]                       # friendly message for the UI
    assert classify(client, "Book a train to Pune.").status_code == 200


def test_error_names_the_offending_field(client):
    missing = client.post("/classify", json={"note": "x"}).json()["errors"][0]
    assert missing["field"] == "text"
    malformed = client.post("/classify", content=b'{"text": ', headers={"Content-Type": "application/json"})
    assert malformed.json()["errors"][0]["field"] == "body"


def test_thousand_character_note_is_accepted_and_longer_is_rejected(client):
    assert classify(client, "x" * 1000).status_code == 200
    assert 400 <= classify(client, "x" * 1001).status_code < 500


def test_non_ascii_text_is_accepted(client):
    assert classify(client, "Move ₹2,000 into savings before the 1st.").status_code == 200


def test_same_text_gives_the_same_response(client):
    text = "Revise for my biology examination."
    assert classify(client, text).json() == classify(client, text).json()


def test_surrounding_whitespace_does_not_change_the_answer(client):
    assert classify(client, "  Book a train to Pune.  ").json() == classify(client, "Book a train to Pune.").json()


def test_uncertain_flag_comes_from_the_model_threshold(classifier):
    """Same model and text; only the threshold moves, and the flag follows it."""
    text = "Confirm the bus seat for Pune."
    always = TestClient(create_app(with_threshold(classifier, 1.01)))
    never = TestClient(create_app(with_threshold(classifier, 0.0)))
    with always, never:
        assert classify(always, text).json()["uncertain"] is True
        assert classify(never, text).json()["uncertain"] is False


def test_most_out_of_scope_notes_are_flagged_uncertain(client):
    notes = load_texts_csv(OUT_OF_SCOPE_CSV)
    flagged = sum(classify(client, n).json()["uncertain"] for n in notes)
    assert flagged / len(notes) >= 0.5


def test_health_reports_the_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["labels"] and body["uncertainty_threshold"] > 0
