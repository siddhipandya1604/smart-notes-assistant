"""Train, compare and save the note classifier.  Run with:  python train.py

Model selection uses 5-fold cross-validation on train.csv only. validation.csv is read
once the model is fixed, purely to report metrics.
"""
import numpy as np
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from smartnotes.config import RANDOM_STATE, TRAIN_CSV, VALIDATION_CSV
from smartnotes.ml.baseline import KeywordRulesBaseline, MajorityBaseline
from smartnotes.ml.evaluate import confusion_table, mistakes, per_category_report, score
from smartnotes.ml.model import ModelBundle, ModelConfig, build_pipeline, save_bundle
from smartnotes.ml.preprocess import load_labeled_csv

CANDIDATES = [
    ModelConfig("logreg word+char", kind="logreg", char_ngrams=True),
    ModelConfig("logreg word only", kind="logreg", char_ngrams=False),
    ModelConfig("naive bayes word+char", kind="nb", char_ngrams=True),
]


def cross_val_scores(cfg: ModelConfig, texts, labels) -> dict:
    """Out-of-fold accuracy, macro-F1 and log-loss on the training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(build_pipeline(cfg), texts, labels, cv=cv, method="predict_proba")
    classes = sorted(set(labels))
    predicted = [classes[i] for i in np.argmax(proba, axis=1)]
    return {**score(labels, predicted), "log_loss": log_loss(labels, proba, labels=classes)}


def fmt(m: dict) -> str:
    text = f"accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}"
    return text + (f"  log-loss={m['log_loss']:.3f}" if "log_loss" in m else "")


def main() -> ModelBundle:
    X_train, y_train = load_labeled_csv(TRAIN_CSV)
    X_val, y_val = load_labeled_csv(VALIDATION_CSV)
    print(f"train rows: {len(X_train)}   validation rows: {len(X_val)}   "
          f"sklearn {sklearn.__version__}   seed {RANDOM_STATE}\n")

    print("== Baselines (validation) ==")
    for name, baseline in [("majority class", MajorityBaseline()), ("keyword rules", KeywordRulesBaseline())]:
        pred = baseline.fit(X_train, y_train).predict(X_val)
        print(f"{name:<24} {fmt(score(y_val, pred))}")

    print("\n== Model selection: 5-fold CV on train.csv only ==")
    cv_results = {cfg.name: cross_val_scores(cfg, X_train, y_train) for cfg in CANDIDATES}
    for name, m in cv_results.items():
        print(f"{name:<24} {fmt(m)}")
    # Accuracy/F1 differ by about one note between candidates, which is noise. Log-loss also
    # rewards well-calibrated probabilities, and the `uncertain` flag is read off those.
    best = min(CANDIDATES, key=lambda cfg: cv_results[cfg.name]["log_loss"])
    print(f"selected (lowest CV log-loss): {best.name}")

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

    bundle = ModelBundle(
        pipeline=pipeline,
        config=best,
        metadata={
            "train_rows": len(X_train),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "cv_macro_f1": round(cv_results[best.name]["macro_f1"], 4),
            "cv_log_loss": round(cv_results[best.name]["log_loss"], 4),
        },
    )
    save_bundle(bundle)
    print("\nsaved model -> models/notes_model.joblib")
    return bundle


if __name__ == "__main__":
    main()
