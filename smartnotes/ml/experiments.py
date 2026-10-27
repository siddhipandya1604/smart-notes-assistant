"""The "try one fix" experiment.  Run with:  python -m smartnotes.ml.experiments

Target: error analysis showed "I must reach the railway station by 10 AM." predicted Study with
0.99 confidence. The stop word "by" and the time "10 AM" (common in Study notes) outweighed the
Travel word "station". Candidate fixes remove that noise. They are compared with template-grouped
CV on train.csv only; validation.csv is read once at the end to report the effect.
"""
import sys
from dataclasses import replace

from smartnotes.config import TRAIN_CSV, VALIDATION_CSV
from smartnotes.ml.evaluate import score
from smartnotes.ml.model import build_pipeline
from smartnotes.ml.preprocess import load_labeled_csv
from smartnotes.ml.train import CANDIDATES, cv_metrics, fmt, out_of_fold_proba

BASE = next(c for c in CANDIDATES if c.name == "naive bayes word+char")
VARIANTS = [
    BASE,
    replace(BASE, name="nb + numbers->num", normalize_numbers=True),
    replace(BASE, name="nb + drop stopwords", drop_stopwords=True),
    replace(BASE, name="nb + both", normalize_numbers=True, drop_stopwords=True),
]
MIN_GAIN = 0.01  # adopt a variant only if CV macro-F1 improves by at least this much


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    X_train, y_train = load_labeled_csv(TRAIN_CSV)
    X_val, y_val = load_labeled_csv(VALIDATION_CSV)

    print("== Grouped 5-fold CV on train.csv (used to choose) ==")
    results = {v.name: cv_metrics(out_of_fold_proba(v, X_train, y_train), y_train) for v in VARIANTS}
    for name, m in results.items():
        print(f"{name:<24} {fmt(m)}")

    base_f1 = results[BASE.name]["macro_f1"]
    best = max(VARIANTS[1:], key=lambda v: results[v.name]["macro_f1"])
    adopt = results[best.name]["macro_f1"] - base_f1 >= MIN_GAIN
    print(f"\nbest variant: {best.name}  (gain {results[best.name]['macro_f1'] - base_f1:+.3f} macro-F1; "
          f"{'adopt' if adopt else 'not adopted'}, minimum gain {MIN_GAIN})")

    print("\n== Validation, read once to report the effect ==")
    for v in (BASE, best):
        pred = list(build_pipeline(v).fit(X_train, y_train).predict(X_val))
        wrong = [t for t, a, p in zip(X_val, y_val, pred) if a != p]
        print(f"{v.name:<24} {fmt(score(y_val, pred))}   mistakes={len(wrong)}")
        print("   still wrong: " + "; ".join(wrong))


if __name__ == "__main__":
    main()
