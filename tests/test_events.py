"""Event layer: targets, peak picking, matching, extra-column symmetry (no model files needed)."""

import numpy as np

from possession import events as Ev
from possession.data import A, D, H
from possession.synthetic import toy_match


def test_onset_targets_mark_each_team_change():
    gt = np.array([3, 3, 0, 0, 0, 1, 1, 2, 2, 0, 0, 3], dtype=np.int8)
    yH, yA, on = Ev.onset_targets(gt, k=1)
    assert list(on[H]) == [9] and list(on[A]) == [5]      # first graded frame is not an onset
    assert yA[4:7].tolist() == [1, 1, 1] and yA.sum() == 3
    assert yH[8:11].tolist() == [1, 1, 1]


def test_pick_events_finds_peaks_and_respects_threshold():
    m, _ = toy_match("H" * 200)
    sH = np.zeros(m.n, dtype=np.float32)
    sA = np.zeros(m.n, dtype=np.float32)
    g = np.flatnonzero(m.graded)
    sH[g[50]] = 0.8
    sH[g[150]] = 0.1
    sA[g[100]] = 0.5
    rows = Ev.pick_events(m, sH, sA, thr=0.2, smooth_s=0.0)
    assert [(int(f), int(t)) for f, t, _ in rows] == [(g[50], H), (g[100], A)]
    assert Ev.pick_events(m, sH, sA, thr=0.9).shape == (0, 3)


def test_match_events_one_to_one():
    rows = np.array([[100, H, 0.9], [105, H, 0.5], [300, A, 0.7]])
    res = Ev.match_events(rows, {H: np.array([102]), A: np.array([400])}, tol=12)
    assert res[H] == (1, 2, 1) and res[A] == (0, 1, 1)


def test_team_extra_is_antisymmetric_under_swap():
    m, _ = toy_match("H" * 100 + "A" * 100)
    n = m.n
    rng = np.random.default_rng(1)
    sH, sA = rng.random(n).astype(np.float32) * 0.5, rng.random(n).astype(np.float32) * 0.5
    sH[m.graded & (np.arange(n) % 40 == 0)] = 0.9
    e1, o1 = Ev.team_extra(m, sH, sA)
    e2, o2 = Ev.team_extra(m, sA, sH)
    assert np.allclose(e1, e2) and np.allclose(o1, -o2, atol=1e-9)


def test_event_features_shapes_and_view_symmetry():
    m, _ = toy_match("H" * 80 + "A" * 80 + "H" * 80)
    even, own, odd = Ev.event_features(m)
    assert even.shape[1] + 2 * own[0].shape[1] + odd.shape[1] == len(Ev.feature_names())
    xh, xa = Ev.view(even, own, odd, H), Ev.view(even, own, odd, A)
    assert xh.shape == xa.shape and xh.shape[1] == len(Ev.feature_names())
