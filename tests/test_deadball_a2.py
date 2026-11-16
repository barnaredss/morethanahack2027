"""A2 dead-ball model: feature / inference plumbing on the synthetic half from test_deadball."""

import numpy as np

from possession import deadball
from tests.test_deadball import make_half, OUT_START, OUT_END, KICKOFF


def _toy_model(F, depth=2):
    """One stump on feature `missing`: goes right (dead) when the ball is missing."""
    I = 2 ** depth - 1
    feat = np.zeros((1, I), dtype=np.int16)
    thr = np.full((1, I), np.inf, dtype=np.float32)
    feat[0, 0], thr[0, 0] = deadball.A2_FEATURE_NAMES.index("missing"), 0.5
    leaf = np.zeros((1, 2 ** depth))
    leaf[0, 0], leaf[0, 1 << (depth - 1)] = -3.0, 3.0      # heap: root -> right child -> leftmost leaf below it
    return dict(feat=feat, thr=thr, leaf=leaf, lr=np.float64(1.0), base=np.float64(0.0))


def test_gbdt_predict_matches_manual():
    X = np.zeros((4, len(deadball.A2_FEATURE_NAMES)))
    X[:, deadball.A2_FEATURE_NAMES.index("missing")] = [0, 1, 0, 1]
    z = deadball.gbdt_predict(_toy_model(X.shape[1]), X)
    assert z[0] < 0 < z[1] and z[2] == z[0] and z[3] == z[1]


def test_features_finite_and_aligned():
    match, pad = make_half()
    X = deadball.frame_features2(match)
    assert X.shape == (match.n, len(deadball.A2_FEATURE_NAMES))
    assert np.isnan(X[:pad]).all() and np.isfinite(X[pad:-pad]).all()
    i = deadball.A2_FEATURE_NAMES.index
    # the ball goes missing at the throw-in: the next-gap features see it coming
    just_before = pad + OUT_START + 15 - 50
    assert X[just_before, i("nxt_dt")] < 3.0
    # collective stop: speed before the whistle is higher than after
    assert X[pad + OUT_START + 10, i("r_delta")] < 0.0 < X[pad + OUT_END - 10, i("r_delta")] + 1.0


def test_p_dead_a2_contract_and_pre_kickoff():
    match, pad = make_half()
    p = deadball.p_dead_a2(match, model=_toy_model(len(deadball.A2_FEATURE_NAMES)))
    assert p.shape == (match.n,) and p.dtype == np.float32
    assert np.isnan(p[:pad]).all() and not np.isnan(p[pad:-pad]).any()
    assert p[pad + 100:pad + KICKOFF - 100].min() > 0.9
    assert np.array_equal(p, deadball.p_dead_a2(match, model=_toy_model(len(deadball.A2_FEATURE_NAMES))),
                          equal_nan=True)


def test_p_dead_a_unchanged_by_a2_code():
    match, pad = make_half()
    p = deadball.p_dead_a(match)
    assert p[pad + OUT_START + 20:pad + OUT_END - 20].min() > 0.7
