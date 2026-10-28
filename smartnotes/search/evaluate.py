"""Measure search quality on the starter notes.  Run with:  python eval_search.py

hit rate @3 = share of queries for which at least one expected note is in the top 3 results.
"""
import json
import sys

from smartnotes.config import DATA_DIR
from smartnotes.search.base import Doc, Retriever, content_words
from smartnotes.search.hybrid import CategoryOnlyRetriever, HybridRetriever
from smartnotes.search.keyword import KeywordRetriever
from smartnotes.search.tfidf import TfidfRetriever
from smartnotes.services.classifier import NoteClassifier
from smartnotes.store.starter import read_starter_notes

QUERIES_JSON = DATA_DIR / "search_queries.json"
NO_MATCH_JSON = DATA_DIR / "search_no_match_queries.json"
K = 3


def load_queries() -> list[dict]:
    return json.loads(QUERIES_JSON.read_text(encoding="utf-8"))


def load_no_match_queries() -> list[str]:
    return json.loads(NO_MATCH_JSON.read_text(encoding="utf-8"))


def shares_no_words(query: str, expected_texts: list[str]) -> bool:
    """True if the query has no content word in common with any note it should find."""
    return all(not (content_words(query) & content_words(text)) for text in expected_texts)


def first_rank(result_ids: list[str], expected: list[str]) -> int | None:
    """1-based position of the first expected note in the results, or None."""
    return next((rank for rank, note_id in enumerate(result_ids, 1) if note_id in expected), None)


def evaluate(retriever: Retriever, docs: list[Doc], queries: list[dict], no_match: list[str]) -> dict:
    retriever.fit(docs)
    texts = {d.id: d.text for d in docs}
    rows = []
    for q in queries:
        ids = [h.doc_id for h in retriever.search(q["query"], limit=K)]
        rows.append({
            "query": q["query"], "expected": q["expected"], "got": ids,
            "rank": first_rank(ids, q["expected"]),
            "zero_overlap": shares_no_words(q["query"], [texts[i] for i in q["expected"]]),
        })

    def rate(selected, hit):
        return sum(hit(r) for r in selected) / len(selected) if selected else float("nan")

    zero = [r for r in rows if r["zero_overlap"]]
    return {
        "name": retriever.name,
        "hit@3": rate(rows, lambda r: r["rank"] is not None),
        "hit@3 zero-overlap": rate(zero, lambda r: r["rank"] is not None),
        "hit@1": rate(rows, lambda r: r["rank"] == 1),
        "mrr": rate(rows, lambda r: 1 / r["rank"] if r["rank"] else 0),
        "no-match empty": sum(not retriever.search(q, limit=K) for q in no_match) / len(no_match),
        "rows": rows,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    classifier = NoteClassifier.load()
    docs = [Doc(n["id"], n["text"]) for n in read_starter_notes()]
    queries, no_match = load_queries(), load_no_match_queries()
    zero = [q for q in queries if shares_no_words(q["query"], [d.text for d in docs if d.id in q["expected"]])]
    print(f"{len(queries)} queries over {len(docs)} starter notes; {len(zero)} share no content words with their target; "
          f"{len(no_match)} extra queries that should find nothing\n")

    retrievers = [KeywordRetriever(), TfidfRetriever(), HybridRetriever(classifier), CategoryOnlyRetriever(classifier)]
    results = [evaluate(r, docs, queries, no_match) for r in retrievers]

    header = f"{'approach':<36}{'hit@3':>7}{'hit@3 (no shared words)':>25}{'hit@1':>7}{'MRR':>7}{'no-match empty':>16}"
    print(header)
    for r in results:
        print(f"{r['name']:<36}{r['hit@3']:>7.2f}{r['hit@3 zero-overlap']:>25.2f}{r['hit@1']:>7.2f}{r['mrr']:>7.2f}{r['no-match empty']:>16.2f}")

    print("\nmisses (expected note not in the top 3):")
    for r in results:
        missed = [row for row in r["rows"] if row["rank"] is None]
        print(f"  {r['name']}: " + ("none" if not missed else "; ".join(
            f"{m['query']!r} wanted {m['expected']} got {m['got']}" for m in missed)))

    print("\nhybrid, topic weight sensitivity (hit@3 / hit@1):")
    for weight in (0.25, 0.5, 1.0):
        r = evaluate(HybridRetriever(classifier, topic_weight=weight), docs, queries, no_match)
        print(f"  weight {weight:<5} {r['hit@3']:.2f} / {r['hit@1']:.2f}   no-match empty {r['no-match empty']:.2f}")


if __name__ == "__main__":
    main()
