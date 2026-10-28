"""Train, compare and save the note classifier.  Run with:  python train.py

Model selection and the uncertainty threshold use train.csv (cross-validated) and half of
the out-of-scope notes. validation.csv and the other half are only read to report numbers.
"""
import sys
import warnings

import numpy as np
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

from smartnotes.config import OUT_OF_SCOPE_CSV, RANDOM_STATE, TRAIN_CSV, VALIDATION_CSV
from smartnotes.ml.baseline import KeywordRulesBaseline, MajorityBaseline
from smartnotes.ml.evaluate import confusion_table, mistakes, per_category_report, score
from smartnotes.ml.grouping import grouped_folds
from smartnotes.ml.model import ModelBundle, ModelConfig, build_pipeline, save_bundle
from smartnotes.ml.preprocess import load_labeled_csv, load_texts_csv
from smartnotes.ml.uncertainty import choose_threshold, flagged_share, max_confidence, split_out_of_scope

CANDIDATES = [
    ModelConfig("logreg word+char", kind="logreg", char_ngrams=True),
    ModelConfig("logreg word only", kind="logreg", char_ngrams=False),
    ModelConfig("naive bayes word+char", kind="nb", char_ngrams=True),
    # The fix chosen in smartnotes/ml/experiments.py (grouped CV on train.csv only).
    ModelConfig("naive bayes + fix", kind="nb", char_ngrams=True, normalize_numbers=True, drop_stopwords=True),
]

# Share of in-scope notes we accept flagging. A flagged note keeps its best-guess category and
# stays visible, so a spurious badge is cheap; a confident wrong label on a note that fits no
# category is the failure the brief cares about. That is why the app ships the highest cap.
IN_SCOPE_FLAG_CAPS = {"5%": 0.05, "10%": 0.10, "20%": 0.20, "30%": 0.30}
SHIPPED_SETTING = "20%"


def out_of_fold_proba(cfg: ModelConfig, texts, labels) -> np.ndarray:
    """Out-of-fold class probabilities on the training data, with template-grouped folds."""
    return cross_val_predict(build_pipeline(cfg), texts, labels, cv=grouped_folds(texts, labels),
                             method="predict_proba")


def cv_metrics(proba: np.ndarray, labels) -> dict:
    classes = sorted(set(labels))
    predicted = [classes[i] for i in np.argmax(proba, axis=1)]
    return {**score(labels, predicted), "log_loss": log_loss(labels, proba, labels=classes)}


def fmt(m: dict) -> str:
    text = f"accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}"
    return text + (f"  log-loss={m['log_loss']:.3f}" if "log_loss" in m else "")


def print_tradeoff(thresholds: dict, in_scope_oof, oos_tune, oos_held, val_conf) -> None:
    print("\n== Uncertainty trade-off (confidence < threshold => uncertain) ==")
    print("threshold chosen from validation + the OOS tune half. 'validation' and 'OOS tune' are therefore")
    print("tuned-on; 'OOS held-out' and 'train OOF' (grouped, pessimistic) are the independent checks.\n")
    columns = [("OOS tune", 10), ("OOS held-out", 14), ("OOS all 12", 12), ("validation", 12), ("train OOF", 11)]
    print(f"{'in-scope cap':<14}{'threshold':>11}" + "".join(f"{name:>{width}}" for name, width in columns))
    oos_all = np.concatenate([oos_tune, oos_held])
    by_threshold: dict[float, list[str]] = {}  # caps that land on the same threshold share a row
    for name, t in thresholds.items():
        by_threshold.setdefault(t, []).append(name)
    for t, names in by_threshold.items():
        shares = [flagged_share(c, t) for c in (oos_tune, oos_held, oos_all, val_conf, in_scope_oof)]
        marker = " <- shipped" if SHIPPED_SETTING in names else ""
        print(f"{'/'.join(names):<14}{t:>11.4f}" + "".join(f"{s:>{w}.0%}" for s, (_, w) in zip(shares, columns)) + marker)


def main() -> ModelBundle:
    # scikit-learn 1.5.2 passes an `iprint` option that newer SciPy ignores with a warning on every
    # logistic-regression fit. Harmless, but it buries the real output, so hide only that message.
    warnings.filterwarnings("ignore", message="Unknown solver options")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # notes contain "₹"; Windows consoles default to cp1252
    X_train, y_train = load_labeled_csv(TRAIN_CSV)
    X_val, y_val = load_labeled_csv(VALIDATION_CSV)
    oos_tune_texts, oos_held_texts = split_out_of_scope(load_texts_csv(OUT_OF_SCOPE_CSV))
    print(f"train rows: {len(X_train)}   validation rows: {len(X_val)}   "
          f"out-of-scope: {len(oos_tune_texts)} tune + {len(oos_held_texts)} held out   "
          f"sklearn {sklearn.__version__}   seed {RANDOM_STATE}\n")

    print("== Baselines (validation) ==")
    for name, baseline in [("majority class", MajorityBaseline()), ("keyword rules", KeywordRulesBaseline())]:
        pred = baseline.fit(X_train, y_train).predict(X_val)
        print(f"{name:<24} {fmt(score(y_val, pred))}")

    print("\n== Model selection: 5-fold template-grouped CV on train.csv only ==")
    oof = {cfg.name: out_of_fold_proba(cfg, X_train, y_train) for cfg in CANDIDATES}
    cv_results = {name: cv_metrics(p, y_train) for name, p in oof.items()}
    for name, m in cv_results.items():
        print(f"{name:<24} {fmt(m)}")
    best = max(CANDIDATES, key=lambda cfg: cv_results[cfg.name]["macro_f1"])
    print(f"selected (highest CV macro-F1): {best.name}")

    pipeline = build_pipeline(best).fit(X_train, y_train)
    pred = list(pipeline.predict(X_val))

    print("\n== Trained model (validation) ==")
    print(f"{best.name:<24} {fmt(score(y_val, pred))}\n")
    print(per_category_report(y_val, pred))
    print(confusion_table(y_val, pred))
    wrong = mistakes(X_val, y_val, pred)
    print(f"\n{len(wrong)} validation mistakes:")
    for text, actual, predicted in wrong:
        print(f"  [{actual} -> {predicted}] {text}")

    # Uncertainty threshold: in-scope side = validation confidence, out-of-scope side = tune half.
    in_scope_oof = oof[best.name].max(axis=1)
    oos_tune = max_confidence(pipeline, oos_tune_texts)
    oos_held = max_confidence(pipeline, oos_held_texts)
    val_conf = max_confidence(pipeline, X_val)
    thresholds = {name: choose_threshold(val_conf, oos_tune, cap) for name, cap in IN_SCOPE_FLAG_CAPS.items()}
    print_tradeoff(thresholds, in_scope_oof, oos_tune, oos_held, val_conf)

    bundle = ModelBundle(
        pipeline=pipeline,
        config=best,
        threshold=thresholds[SHIPPED_SETTING],
        metadata={
            "train_rows": len(X_train),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "cv_macro_f1": round(cv_results[best.name]["macro_f1"], 4),
            "cv_log_loss": round(cv_results[best.name]["log_loss"], 4),
            "uncertainty_setting": SHIPPED_SETTING,
        },
    )
    save_bundle(bundle)
    print(f"\nsaved model (uncertainty threshold {bundle.threshold:.4f}) -> models/notes_model.joblib")
    return bundle


if __name__ == "__main__":
    main()
