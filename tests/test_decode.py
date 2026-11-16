"""HSMM decoder tests (synthetic only; no match data needed)."""

import numpy as np
import pytest

from possession import decode, pipelines
from possession.data import A, D, H, X
from possession.decode_synth import make_match
from possession.scoring import score
from possession.synthetic import toy_match


def _clean(labels, nan=False):
    match, gt = toy_match(labels, pad=20)
    pd_ = np.where(gt == D, 0.97, 0.03).astype(np.float32)
    ph = np.where(gt == H, 0.95, np.where(gt == A, 0.05, 0.5)).astype(np.float32)
    if nan:
        pd_[:] = np.nan
        ph[:] = np.nan
    return match, gt, pd_, ph


def test_clean_evidence_gives_perfect_spells():
    match, gt, pd_, ph = _clean("D" * 100 + "H" * 300 + "A" * 250 + "D" * 120 + "A" * 200)
    pred = decode.hsmm_decode(match, pd_, ph)
    assert (pred[~match.graded] == X).all() and (pred[match.graded] != X).all()
    assert score(gt, pred)["spell_f1_mean"] == pytest.approx(1.0)
    assert (pred == gt).mean() > 0.99


def test_nan_is_uninformative_not_half():
    ll = decode.evidence_ll(np.full(5, np.nan), np.full(5, np.nan), pd0=0.4)
    assert np.allclose(ll, 0.0)
    # known p_home with unknown p_dead must not favour D
    ll = decode.evidence_ll(np.full(3, np.nan), np.full(3, 0.9), pd0=0.4)
    assert (ll[:, H] > 0).all() and (ll[:, A] < 0).all() and np.allclose(ll[:, D], 0.0)


def test_all_nan_returns_single_constant_segment():
    match, _, pd_, ph = _clean("H" * 200 + "A" * 200, nan=True)
    pred = decode.hsmm_decode(match, pd_, ph)
    g = pred[match.graded]
    assert len(set(g.tolist())) == 1


def test_flicker_is_suppressed_and_real_short_spell_kept():
    match, gt, pd_, ph = _clean("H" * 400 + "A" * 40 + "H" * 400)   # 40 frames = 1.6 s, real
    pred = decode.hsmm_decode(match, pd_, ph)
    assert score(gt, pred)["n_pred_spells"] == 3
    # 4-frame blips of the wrong team inside a long spell are removed
    match, gt, pd_, ph = _clean("H" * 800)
    ph[match.graded] = 0.95
    ph[100:104] = 0.02
    ph[300:305] = 0.02
    assert (decode.hsmm_decode(match, pd_, ph)[match.graded] == H).all()


def test_boundary_lands_near_the_truth():
    match, gt, pd_, ph = _clean("H" * 500 + "A" * 500)
    pred = decode.hsmm_decode(match, pd_, ph)
    assert abs(int(np.flatnonzero(pred == A)[0]) - int(np.flatnonzero(gt == A)[0])) <= 1


def test_refine_snaps_to_plateau_midpoint():
    codes = np.array([H] * 100 + [A] * 100, dtype=np.int8)
    ll = np.zeros((200, 3))
    ll[:60, H] = 1.0                      # H evidence up to 60, A evidence from 140, nothing between
    ll[140:, A] = 1.0
    out = decode.refine_boundaries(codes, ll, 80, np.array([5, 5, 5]))
    assert out[:100].tolist().count(H) in range(98, 102) and abs(int(np.flatnonzero(out == A)[0]) - 100) <= 1


def test_change_points_move_an_ambiguous_boundary():
    match, gt, pd_, ph = _clean("H" * 400 + "A" * 400)
    pd_[:] = np.nan
    ph[match.graded] = np.nan             # no evidence at all between 100 and 700
    ph[match.graded][:50] = 0.95
    first = int(np.flatnonzero(match.graded)[0])
    ph[first:first + 50] = 0.95
    ph[first + 750:first + 800] = 0.05
    cp = [(first + 560, A, 1.0)]
    pred = decode.hsmm_decode(match, pd_, ph, change_points=cp, refine_window_s=0)
    boundary = int(np.flatnonzero(pred == A)[0])
    assert abs(boundary - (first + 560)) <= 12


def test_unknown_param_rejected():
    match, _, pd_, ph = _clean("H" * 100)
    with pytest.raises(TypeError):
        decode.hsmm_decode(match, pd_, ph, bogus=1)


def test_deterministic_and_beats_majority_on_synthetic():
    r_h, r_m = [], []
    for seed in range(3):
        m, gt, pd_, ph = make_match(n_halves_frames=8000, seed=seed, noise=1.0)
        a = decode.hsmm_decode(m, pd_, ph)
        assert np.array_equal(a, decode.hsmm_decode(m, pd_, ph))
        r_h.append(score(gt, a)["spell_f1_mean"])
        r_m.append(score(gt, decode.majority_decode(m, pd_, ph))["spell_f1_mean"])
    assert np.mean(r_h) > np.mean(r_m) + 0.1


def test_pipeline_registered_and_valid():
    assert "decode_c_v1" in pipelines.names()
    match, gt = toy_match("H" * 300 + "D" * 100 + "A" * 300)
    out = pipelines.get("decode_c_v1")(match)
    assert out.shape == (match.n,) and out.dtype == np.int8
    assert (out[~match.graded] == X).all()
