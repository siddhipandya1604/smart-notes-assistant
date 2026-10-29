import pytest

from smartnotes.search.base import Doc
from smartnotes.search.evaluate import evaluate, load_no_match_queries, load_queries, shares_no_words
from smartnotes.search.hybrid import HybridRetriever
from smartnotes.search.keyword import KeywordRetriever
from smartnotes.search.tfidf import TfidfRetriever
from smartnotes.store.starter import read_starter_notes


@pytest.fixture(scope="module")
def docs():
    return [Doc(n["id"], n["text"]) for n in read_starter_notes()]


def ids(hits):
    return [h.doc_id for h in hits]


# --- the evaluation set meets the brief -------------------------------------------------------
def test_query_set_has_enough_queries_and_paraphrases(docs):
    queries = load_queries()
    texts = {d.id: d.text for d in docs}
    assert len(queries) >= 10
    assert all(q["expected"] and set(q["expected"]) <= set(texts) for q in queries)
    zero_overlap = [q for q in queries if shares_no_words(q["query"], [texts[i] for i in q["expected"]])]
    assert len(zero_overlap) >= 3


# --- retrievers -------------------------------------------------------------------------------
def test_keyword_finds_a_shared_word(docs):
    assert ids(KeywordRetriever().fit(docs).search("hotel"))[0] == "N007"


def test_keyword_cannot_bridge_a_paraphrase(docs):
    assert KeywordRetriever().fit(docs).search("money transfer") == []


def test_tfidf_matches_an_inflection(docs):
    assert ids(TfidfRetriever().fit(docs).search("savings"))[0] == "N006"


@pytest.mark.parametrize("query, expected", [
    ("money transfer", "N006"),      # no shared words: found through the Money topic
    ("groceries", "N012"),
    ("when is my exam?", "N001"),
])
def test_hybrid_bridges_paraphrases(classifier, docs, query, expected):
    hits = HybridRetriever(classifier).fit(docs).search(query, limit=3)
    assert expected in ids(hits)
    assert next(h for h in hits if h.doc_id == expected).match == "topic"


def test_hybrid_puts_word_matches_first(classifier, docs):
    hits = HybridRetriever(classifier).fit(docs).search("hotel")
    assert hits[0].doc_id == "N007" and hits[0].match in ("keyword", "both")


@pytest.mark.parametrize("query", ["dentist appointment", "walk the dog", "call my mother"])
def test_hybrid_returns_nothing_when_the_classifier_is_unsure(classifier, docs, query):
    """No confident topic and no shared words: say so instead of listing unrelated notes."""
    assert HybridRetriever(classifier).fit(docs).search(query) == []


def test_search_with_no_documents_is_empty(classifier):
    assert HybridRetriever(classifier).fit([]).search("hotel") == []
    assert TfidfRetriever().fit([]).search("hotel") == []


@pytest.mark.parametrize("text", ["???", "the", "12345", "I am"])
def test_notes_with_nothing_to_index_do_not_crash_search(classifier, text):
    retriever = HybridRetriever(classifier).fit([Doc("N013", text)])
    assert retriever.search("hotel") == []


# --- the numbers reported in RESULTS.md -------------------------------------------------------
def test_hybrid_beats_keyword_on_the_evaluation_set(classifier, docs):
    queries, no_match = load_queries(), load_no_match_queries()
    keyword = evaluate(KeywordRetriever(), docs, queries, no_match)
    hybrid = evaluate(HybridRetriever(classifier), docs, queries, no_match)
    assert hybrid["hit@3"] > keyword["hit@3"]
    assert hybrid["hit@3 zero-overlap"] > keyword["hit@3 zero-overlap"] == 0


# --- HTTP -------------------------------------------------------------------------------------
def test_search_finds_a_starter_note_with_its_id_and_text(client):
    client.post("/api/notes/load-starter")
    body = client.get("/api/search", params={"q": "hotel"}).json()
    top = body["results"][0]
    assert top["note"]["id"] == "N007" and "Jaipur" in top["note"]["text"]
    assert body["message"] is None


def test_search_accepts_a_natural_language_question(client):
    client.post("/api/notes/load-starter")
    found = {r["note"]["id"] for r in client.get("/api/search", params={"q": "what do I need to buy?"}).json()["results"]}
    assert found & {"N004", "N012"}


def test_search_finds_a_note_the_user_just_saved(client):
    saved = client.post("/api/notes", json={"text": "Pick up the dry cleaning on Friday."}).json()
    results = client.get("/api/search", params={"q": "dry cleaning"}).json()["results"]
    assert results[0]["note"]["id"] == saved["id"]


def test_search_with_no_good_match_says_so(client):
    client.post("/api/notes/load-starter")
    body = client.get("/api/search", params={"q": "dentist appointment"}).json()
    assert body["results"] == [] and "No notes match" in body["message"]


def test_search_before_any_note_is_saved(client):
    body = client.get("/api/search", params={"q": "hotel"}).json()
    assert body["results"] == [] and "no saved notes" in body["message"]


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "   "}, {}, {"q": "x" * 201}, {"q": "hotel", "limit": 0}])
def test_bad_search_requests_are_4xx_with_a_message(client, params):
    r = client.get("/api/search", params=params)
    assert 400 <= r.status_code < 500 and r.json()["message"]


def test_a_saved_note_with_no_searchable_words_does_not_break_search(client):
    assert client.post("/api/notes", json={"text": "???"}).status_code == 201
    r = client.get("/api/search", params={"q": "hotel"})
    assert r.status_code == 200 and r.json()["results"] == []


def test_search_query_too_long_names_the_search_limit(client):
    """A 201-character query must report the 200-character search limit, not the note limit."""
    r = client.get("/api/search", params={"q": "a" * 201})
    assert r.status_code == 422
    assert "search query" in r.json()["message"]
    assert "200" in r.json()["message"] and "1000" not in r.json()["message"]
    assert r.json()["errors"][0]["field"] == "q"


def test_search_without_q_names_the_q_field(client):
    r = client.get("/api/search")
    assert r.status_code == 422
    assert "'q'" in r.json()["message"] and "'text'" not in r.json()["message"]
    assert r.json()["errors"][0]["field"] == "q"


def test_note_too_long_still_reports_the_note_limit(client):
    """The fix must not change the original wording for notes."""
    r = client.post("/api/notes", json={"text": "x" * 1001})
    assert r.status_code == 422
    assert r.json()["message"] == "A note can be at most 1000 characters."


def test_search_sees_notes_added_after_the_first_search(client):
    client.post("/api/notes", json={"text": "Book a train to Pune."})
    client.get("/api/search", params={"q": "train"})                    # builds the index
    client.post("/api/notes", json={"text": "Pick up the dry cleaning on Friday."})
    results = client.get("/api/search", params={"q": "dry cleaning"}).json()["results"]
    assert results and results[0]["note"]["text"].startswith("Pick up the dry")
