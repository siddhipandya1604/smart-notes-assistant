import numpy as np

from smartnotes.ml.uncertainty import choose_threshold, flagged_share, is_uncertain, split_out_of_scope


def test_split_is_deterministic_and_disjoint():
    texts = [f"note {i}" for i in range(12)]
    tune, held_out = split_out_of_scope(texts)
    assert (tune, held_out) == split_out_of_scope(texts)
    assert len(tune) == 6 and len(held_out) == 6
    assert set(tune).isdisjoint(held_out)
    assert set(tune) | set(held_out) == set(texts)


def test_flagged_share_uses_strict_less_than():
    conf = np.array([0.4, 0.5, 0.9])
    assert flagged_share(conf, 0.5) == 1 / 3
    assert is_uncertain(0.49, 0.5) and not is_uncertain(0.5, 0.5)


def test_choose_threshold_respects_in_scope_cap():
    in_scope = np.array([0.95] * 9 + [0.6])       # one in-scope note is genuinely unsure
    oos = np.array([0.3, 0.4, 0.5, 0.97])
    t = choose_threshold(in_scope, oos, max_in_scope_flagged=0.10)
    assert flagged_share(in_scope, t) <= 0.10
    assert flagged_share(oos, t) == 0.75           # catches the three low-confidence ones


def test_choose_threshold_tighter_cap_flags_less():
    in_scope = np.linspace(0.5, 1.0, 20)
    oos = np.array([0.45, 0.55, 0.65])
    loose = choose_threshold(in_scope, oos, 0.30)
    tight = choose_threshold(in_scope, oos, 0.05)
    assert tight <= loose
