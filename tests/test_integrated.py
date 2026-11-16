"""Integration: integrated_v1 pipeline, dead_weight decoder param, warning-free team_speed."""

import warnings

import numpy as np

from possession import deadball, decode, emissions, pipelines
from possession.data import A, D, H, X
from possession.synthetic import toy_match
from tests.test_deadball import make_half


def test_integrated_registered_and_valid_and_deterministic():
    match, pad = make_half()
    fn = pipelines.get("integrated_v1")
    a, b = fn(match), fn(match)
    assert a.shape == (match.n,) and a.dtype == np.int8
    assert (a[:pad] == X).all() and (a[pad:-pad] != X).all()
    assert (a == b).all()


def test_integrated_is_the_composition_of_the_three_components():
    match, _ = make_half()
    manual = decode.hsmm_decode(match, deadball.p_dead_a(match), emissions.p_home_b(match))
    assert (pipelines.get("integrated_v1")(match) == manual).all()


def test_integrated_passes_decoder_params_through():
    match, _ = make_half()
    fn = pipelines.get("integrated_v1")
    assert (fn(match, evidence=0.2) == decode.hsmm_decode(
        match, deadball.p_dead_a(match), emissions.p_home_b(match), evidence=0.2)).all()


def test_dead_weight_default_is_the_plain_posterior():
    pd = np.array([0.1, 0.5, 0.9, np.nan], dtype=np.float32)
    ph = np.array([0.7, 0.2, 0.5, np.nan], dtype=np.float32)
    ll = decode.evidence_ll(pd, ph, 0.4, 0.02, 1.0)
    pdc, phc = np.clip(pd, 0.02, 0.98), np.clip(ph, 0.02, 0.98)
    half_prior = 0.3
    assert np.allclose(ll[0, H], np.log((1 - pdc[0]) * phc[0] / half_prior))
    assert np.allclose(ll[1, A], np.log((1 - pdc[1]) * (1 - phc[1]) / half_prior))
    assert np.allclose(ll[2, D], np.log(pdc[2] / 0.4))
    assert np.allclose(ll[3], 0.0)                               # NaN stays uninformative
    assert np.allclose(decode.evidence_ll(pd, ph, 0.4, 0.02, 1.0, dead_weight=1.0), ll)


def test_dead_weight_scales_only_the_dead_alive_part():
    pd = np.array([0.1, 0.9], dtype=np.float32)
    ph = np.array([0.7, 0.5], dtype=np.float32)
    l1 = decode.evidence_ll(pd, ph, 0.4, 0.02, 1.0, dead_weight=1.0)
    l2 = decode.evidence_ll(pd, ph, 0.4, 0.02, 1.0, dead_weight=2.0)
    assert np.allclose(l2[:, D], 2 * l1[:, D])
    # the team contrast H - A is untouched
    assert np.allclose(l2[:, H] - l2[:, A], l1[:, H] - l1[:, A])
    assert np.allclose(l2[:, H] - 2 * np.log((1 - pd.astype(float)) / 0.6),
                       l1[:, H] - np.log((1 - pd.astype(float)) / 0.6), atol=1e-6)


def test_decoder_accepts_dead_weight_and_clean_toy_stays_perfect():
    match, gt = toy_match("H" * 300 + "D" * 200 + "A" * 300 + "D" * 150 + "H" * 250)
    pdead = np.where(gt == D, 0.95, 0.05).astype(np.float32)
    phome = np.where(gt == H, 0.9, 0.1).astype(np.float32)
    phome[gt == D] = np.nan
    for dw in (0.5, 1.0, 2.0):
        out = decode.hsmm_decode(match, pdead, phome, dead_weight=dw)
        assert (out == gt).mean() > 0.97


def test_team_speed_is_silent_on_empty_frames():
    match, _ = make_half()
    match.speed[:50] = np.nan                                    # frames with no usable player
    match.players[:50] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp = deadball.team_speed(match)
    assert np.isnan(sp[:20]).all() and np.isfinite(sp[1000])
