#!/usr/bin/env python3
"""Send data/validation.csv and data/out_of_scope_examples.csv through a running /classify.

Reproduces the validation accuracy / macro-F1 and the uncertain rates in RESULTS.md from the
live endpoint, with the standard library only.

Usage: python examples/replay_validation.py --url http://localhost:8000/classify
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = ["Study", "Money", "Travel", "Shopping"]


def classify(url, text):
    req = Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=4) as r:
        return json.load(r)


def macro_f1(actual, predicted):
    scores = []
    for label in LABELS:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def read_rows(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/classify")
    url = parser.parse_args().url
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = read_rows("validation.csv")
    results = [classify(url, r["text"]) for r in rows]
    actual = [r["category"] for r in rows]
    predicted = [res["category"] for res in results]
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(rows)
    print(f"validation ({len(rows)} notes): accuracy={accuracy:.3f}  macro-F1={macro_f1(actual, predicted):.3f}  "
          f"flagged uncertain={sum(res['uncertain'] for res in results) / len(rows):.0%}")

    oos = [r["text"] for r in read_rows("out_of_scope_examples.csv")]
    flags = [classify(url, t)["uncertain"] for t in oos]
    print(f"out-of-scope ({len(oos)} notes, 6 of them were used to pick the threshold): "
          f"flagged uncertain={sum(flags) / len(oos):.0%}")


if __name__ == "__main__":
    main()
