import re

from smartnotes.config import TRAIN_CSV, VALIDATION_CSV
from smartnotes.ml.preprocess import load_labeled_csv
from smartnotes.store.starter import read_starter_notes
from smartnotes.text import STOPWORDS


def test_function_words_are_stopped():
    assert {"the", "my", "is", "a", "when", "what", "by", "am", "pm"} <= STOPWORDS


def test_category_cues_are_not_stopped():
    """scikit-learn's English list drops these; they are some of the best category cues."""
    assert not {"bill", "move", "due", "find", "put", "call", "amount", "cost", "interest"} & STOPWORDS


def test_no_stop_word_is_the_only_strong_cue_of_one_category():
    """Guard against re-adding content words: any stopped word that appears in 4+ training notes
    must occur in more than one category."""
    texts, labels = load_labeled_csv(TRAIN_CSV)
    seen: dict[str, set] = {}
    count: dict[str, int] = {}
    for text, label in zip(texts, labels):
        for word in set(re.findall(r"[a-z0-9]+", text.lower())) & STOPWORDS:
            seen.setdefault(word, set()).add(label)
            count[word] = count.get(word, 0) + 1
    one_sided = {w for w, c in count.items() if c >= 4 and len(seen[w]) == 1}
    # "our" (only Travel in this tiny sample: "our trip") and "at" are function words, so allowed.
    assert one_sided <= {"our", "at", "into", "after", "do", "and", "if", "in", "am"}
