from smartnotes.config import TRAIN_CSV
from smartnotes.ml.grouping import grouped_folds, template_groups
from smartnotes.ml.preprocess import load_labeled_csv


def test_template_siblings_share_a_group():
    texts, _ = load_labeled_csv(TRAIN_CSV)
    groups = template_groups(texts)
    # Same template, only the day differs (both are in train.csv).
    a = groups[texts.index("Review my bank statement on Tuesday.")]
    b = groups[texts.index("Review my bank statement on Monday.")]
    other = groups[texts.index("Move ₹900 into my savings account.")]
    assert a == b
    assert a != other


def test_groups_never_mix_labels_in_the_training_data():
    texts, labels = load_labeled_csv(TRAIN_CSV)
    groups = template_groups(texts)
    for g in set(groups):
        assert len({l for l, gi in zip(labels, groups) if gi == g}) == 1


def test_grouped_folds_keep_templates_on_one_side():
    texts, labels = load_labeled_csv(TRAIN_CSV)
    groups = template_groups(texts)
    for train_idx, test_idx in grouped_folds(texts, labels):
        assert {groups[i] for i in train_idx}.isdisjoint({groups[i] for i in test_idx})
