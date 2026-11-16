"""Synthetic possession sequences + noisy emissions, for developing/tuning the decoder.

Not a tracking simulator: it fabricates ground-truth H/A/D sequences with football-like durations
and then the two per-frame estimator outputs (p_dead, p_home) with realistic failure modes:
  * temporally correlated noise (AR(1) in logit space, correlation time `tau_s`),
  * burst errors (a stretch where the estimator confidently says the wrong thing),
  * NaN gaps (ball missing -> p_home NaN, about `gap_frac` of frames; p_dead NaN in some gaps).
`noise` scales noise sd and burst rate together. Deterministic given the seed.
"""

import numpy as np

from .data import A, D, H, Match


def _lognormal_frames(rng, median_s, sigma, fps, lo_s, hi_s):
    return int(round(np.clip(rng.lognormal(np.log(median_s), sigma), lo_s, hi_s) * fps))


def gen_truth(n, rng, fps=25, alive_median_s=8.0, alive_sigma=1.0, short_frac=0.12,
              dead_median_s=10.0, dead_sigma=0.5, p_turn=0.5):
    """Ground-truth codes for one half: starts dead (pre-kickoff), alternates spells and stoppages;
    `short_frac` of spells are genuine short spells (0.7-2.5 s)."""
    out = np.empty(n, dtype=np.int8)
    t, state = 0, D
    team = int(rng.integers(2))
    while t < n:
        if state == D:
            d = _lognormal_frames(rng, dead_median_s, dead_sigma, fps, 5.0, 30.0)
        elif rng.random() < short_frac:
            d = int(rng.uniform(0.7, 2.5) * fps)
        else:
            d = _lognormal_frames(rng, alive_median_s, alive_sigma, fps, 2.5, 90.0)
        out[t:t + d] = state
        t += d
        if state == D:
            state, team = team, int(rng.integers(2))
        elif rng.random() < p_turn:
            state = team = 1 - state
        else:
            state = D
    return out


def _ar1(rng, n, sd, tau):
    rho = np.exp(-1.0 / max(tau, 1e-6))
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def _runs_mask(rng, n, p_median, g_median, sigma=1.2):
    """True = gap. Alternating present/gap runs with log-normal lengths."""
    mask = np.zeros(n, dtype=bool)
    t, gap = int(rng.integers(0, int(p_median))), False
    while t < n:
        d = max(1, int(rng.lognormal(np.log(g_median if gap else p_median), sigma)))
        if gap:
            mask[t:t + d] = True
        t += d
        gap = not gap
    return mask


def _bursts(rng, n, rate_per_1000, median_len):
    mask = np.zeros(n, dtype=bool)
    for s in rng.integers(0, n, rng.poisson(rate_per_1000 * n / 1000.0)):
        mask[s:s + max(5, int(rng.lognormal(np.log(median_len), 0.6)))] = True
    return mask


def gen_emissions(truth, rng, fps=25, noise=1.0, margin_dead=2.0, margin_team=2.0, gap_frac=0.41,
                  burst_rate=1.2, burst_len=40, tau_s=1.0, nan_pd_in_gap=0.3):
    """(p_dead, p_home) float32 for one half given its truth codes."""
    n = len(truth)
    tau = tau_s * fps
    sgn_dead = np.where(truth == D, 1.0, -1.0)
    sgn_home = np.where(truth == H, 1.0, np.where(truth == A, -1.0, 0.0))
    z_dead = margin_dead * sgn_dead + _ar1(rng, n, 1.2 * noise, tau)
    z_home = margin_team * sgn_home + _ar1(rng, n, 1.5 * noise, tau)
    z_dead[_bursts(rng, n, burst_rate * noise, burst_len)] *= -1.0
    flip = _bursts(rng, n, burst_rate * noise, burst_len)
    z_home[flip & (truth != D)] *= -1.0
    z_home[truth == D] = rng.normal(0, 2.0, int((truth == D).sum()))   # team meaningless when dead
    p_dead = (1.0 / (1.0 + np.exp(-z_dead))).astype(np.float32)
    p_home = (1.0 / (1.0 + np.exp(-z_home))).astype(np.float32)

    # gap_frac of frames in gaps: with sigma 1.2 the mean ratio equals the median ratio
    gap = _runs_mask(rng, n, 90.0, 90.0 * gap_frac / (1.0 - gap_frac))
    p_home[gap] = np.nan
    pd_gap = gap & _runs_mask(rng, n, 1000.0, 1000.0 * nan_pd_in_gap / (1.0 - nan_pd_in_gap))
    p_dead[pd_gap] = np.nan
    return p_dead, p_home


def make_match(n_halves_frames=20000, seed=0, fps=25, pad=500, **kw):
    """(match, gt_codes, p_dead, p_home) with two graded halves; X (code 3) outside them.
    Keyword args: truth params (gen_truth) and emission params (gen_emissions)."""
    rng = np.random.default_rng(seed)
    truth_keys = {"alive_median_s", "alive_sigma", "short_frac", "dead_median_s", "dead_sigma",
                  "p_turn"}
    tkw = {k: v for k, v in kw.items() if k in truth_keys}
    ekw = {k: v for k, v in kw.items() if k not in truth_keys}
    L = n_halves_frames
    n = 2 * L + 3 * pad
    gt = np.full(n, 3, dtype=np.int8)
    half = np.zeros(n, dtype=np.int8)
    p_dead = np.full(n, np.nan, dtype=np.float32)
    p_home = np.full(n, np.nan, dtype=np.float32)
    for h, start in ((1, pad), (2, 2 * pad + L)):
        tr = gen_truth(L, rng, fps, **tkw)
        pdd, phh = gen_emissions(tr, rng, fps, **ekw)
        gt[start:start + L], half[start:start + L] = tr, h
        p_dead[start:start + L], p_home[start:start + L] = pdd, phh
    match = Match(name="synth", n=n, fps=fps, pitch=(105.0, 68.0), ball=np.zeros((n, 2)),
                  players=np.zeros((n, 2, 2)), speed=np.zeros((n, 2)),
                  slot_team=np.array([H, A], dtype=np.int8), half=half,
                  home_attacks_right={1: True, 2: False})
    return match, gt, p_dead, p_home
