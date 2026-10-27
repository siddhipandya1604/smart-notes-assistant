"""Model pipeline factory, and saving/loading of the model file the app serves."""
from dataclasses import dataclass, field
from functools import partial

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from smartnotes.config import LABELS, MODEL_PATH, RANDOM_STATE
from smartnotes.ml.preprocess import normalize


@dataclass(frozen=True)
class ModelConfig:
    """One candidate setup. Frozen so it can be printed, compared and stored with the model."""

    name: str
    kind: str = "logreg"          # "logreg" or "nb"
    char_ngrams: bool = True      # add character n-grams next to word n-grams
    C: float = 10.0               # logistic regression inverse regularisation
    normalize_numbers: bool = False
    drop_stopwords: bool = False  # English stop words are removed from the word features only


def build_pipeline(cfg: ModelConfig) -> Pipeline:
    prep = partial(normalize, numbers=cfg.normalize_numbers)  # module-level -> picklable
    features = [("word", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), sublinear_tf=True,
                                         stop_words="english" if cfg.drop_stopwords else None))]
    if cfg.char_ngrams:
        features.append(("char", TfidfVectorizer(preprocessor=prep, analyzer="char_wb",
                                                 ngram_range=(2, 5), sublinear_tf=True)))
    if cfg.kind == "logreg":
        clf = LogisticRegression(C=cfg.C, max_iter=1000, random_state=RANDOM_STATE)
    elif cfg.kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"unknown model kind: {cfg.kind}")
    return Pipeline([("features", FeatureUnion(features)), ("clf", clf)])


@dataclass
class ModelBundle:
    """Everything the app needs, saved as a single file."""

    pipeline: Pipeline
    config: ModelConfig
    labels: tuple = LABELS
    threshold: float = 0.0        # confidence below this => uncertain (set by smartnotes.ml.uncertainty)
    metadata: dict = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH) -> ModelBundle:
    return joblib.load(path)
