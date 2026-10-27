"""Evidence for a prediction: which features pushed it, and which training notes look alike.

    python -m smartnotes.ml.explain "I must reach the railway station by 10 AM." --expected Travel
"""
import argparse
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from smartnotes.config import TRAIN_CSV
from smartnotes.ml.model import ModelBundle, load_bundle
from smartnotes.ml.preprocess import load_labeled_csv


def _class_weights(bundle: ModelBundle) -> np.ndarray:
    """Per-class, per-feature weights: logistic coefficients or Naive Bayes log-probabilities."""
    clf = bundle.pipeline.named_steps["clf"]
    return clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_


def feature_push(bundle: ModelBundle, text: str, toward: str, away_from: str, top_k: int = 6) -> list[tuple[str, float]]:
    """Features of `text` that favour class `toward` over `away_from`, strongest first.

    The score is tf-idf value x (weight[toward] - weight[away_from]); the sum over all
    features (plus the class prior) is the model's log-odds between the two classes.
    """
    features = bundle.pipeline.named_steps["features"]
    classes = list(bundle.pipeline.classes_)
    weights = _class_weights(bundle)
    x = features.transform([text]).toarray()[0]
    delta = x * (weights[classes.index(toward)] - weights[classes.index(away_from)])
    names = features.get_feature_names_out()
    order = np.argsort(-np.abs(delta))[:top_k * 3]
    ranked = [(str(names[i]).split("__", 1)[-1], float(delta[i])) for i in order if delta[i] != 0]
    return sorted(ranked, key=lambda p: -p[1])[:top_k]


def nearest_training_notes(bundle: ModelBundle, text: str, k: int = 5) -> list[tuple[float, str, str]]:
    """The k most similar training notes as (cosine similarity, label, text)."""
    X_train, y_train = load_labeled_csv(TRAIN_CSV)
    features = bundle.pipeline.named_steps["features"]
    sims = cosine_similarity(features.transform([text]), features.transform(X_train))[0]
    return [(float(sims[i]), y_train[i], X_train[i]) for i in np.argsort(-sims)[:k]]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--expected", help="the label you think is right, to see what pushed away from it")
    args = parser.parse_args()

    bundle = load_bundle()
    proba = bundle.pipeline.predict_proba([args.text])[0]
    classes = list(bundle.pipeline.classes_)
    predicted = classes[int(np.argmax(proba))]
    print(f"note: {args.text}")
    print("probabilities: " + ", ".join(f"{c} {p:.3f}" for c, p in sorted(zip(classes, proba), key=lambda cp: -cp[1])))
    print(f"predicted: {predicted}   uncertain: {proba.max() < bundle.threshold}  (threshold {bundle.threshold:.4f})")

    if args.expected and args.expected != predicted:
        print(f"\nfeatures pushing toward {predicted} over {args.expected}:")
        for name, value in feature_push(bundle, args.text, predicted, args.expected):
            print(f"  {name!r:<22} {value:+.3f}")
        print(f"\nfeatures pushing toward {args.expected} over {predicted}:")
        for name, value in feature_push(bundle, args.text, args.expected, predicted):
            print(f"  {name!r:<22} {value:+.3f}")

    print("\nnearest training notes:")
    for sim, label, text in nearest_training_notes(bundle, args.text):
        print(f"  {sim:.2f}  [{label}] {text}")


if __name__ == "__main__":
    main()
