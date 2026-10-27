"""Group near-duplicate notes so cross-validation does not leak them across folds.

train.csv is built from templates ("Look for a sale on <item> this <day>"). With plain random
folds, a held-out note almost always has a near-twin in the training folds, so CV scores and
CV confidences look far better than they do on notes the model has not seen the template of.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from smartnotes.config import RANDOM_STATE

# Cosine similarity above which two notes count as the same template. 0.6 keeps exact-template
# siblings together (52 groups of at most 4 notes). 0.4 also merged related templates and
# removed whole vocabularies from the training folds (CV accuracy ~0.5, far below validation);
# plain random folds leak twins (CV accuracy ~0.97). 0.5 / 0.7 gave the same ranking of models.
SIMILARITY = 0.6


def template_groups(texts: list[str], similarity: float = SIMILARITY) -> np.ndarray:
    """Connected components of the 'cosine similarity >= threshold' graph over content words."""
    vectors = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}").fit_transform(texts)
    close = (vectors @ vectors.T).toarray() >= similarity

    group = np.full(len(texts), -1)
    next_id = 0
    for start in range(len(texts)):
        if group[start] != -1:
            continue
        stack = [start]
        group[start] = next_id
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(close[i] & (group == -1)):
                group[j] = next_id
                stack.append(j)
        next_id += 1
    return group


def grouped_folds(texts: list[str], labels: list[str], n_splits: int = 5):
    """Stratified folds in which a template never appears on both sides of a split."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(texts, labels, groups=template_groups(texts)))
