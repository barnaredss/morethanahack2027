"""Decoder: per-frame evidence -> clean possession labels.

Contract: a decoder is `fn(match, p_dead, p_home, **params) -> int8[n]` of label codes (H, A, D, X),
where p_dead and p_home are float32[n] in [0, 1] with NaN for "no opinion". It decodes each half
independently (match.half_blocks()) and writes X on every frame outside the halves.

Decoders here:
  * majority_decode : the starter baseline's majority vote, applied per half (placeholder).
  * hsmm_decode     : explicit-duration HMM + boundary refinement (decode_c_v1).

HSMM model (per half):
  * emissions: per-frame log-likelihood ratio vs the prior, built from p_dead / p_home. NaN means
    "no opinion" and contributes exactly 0 to every state (see `evidence_ll`), so a gap is neither
    evidence for D nor for either team.
  * durations: log-normal per state class (H and A share one, D has its own) plus a hard minimum.
    This kills flicker (a 3-frame spell has no prior mass) while still allowing genuine short
    spells (the labelled external games have alive spells down to ~0.6-1 s, 5th percentile 1.4 s).
  * transitions: H<->A direct ("turnover") costs `switch_penalty` extra nats; D->H/A is a restart,
    H/A->D a stoppage. Same-state "transitions" are allowed at a fixed cost so segments longer
    than the duration window (dmax) stay representable; they merge in the output.
  * optional change points (frame, team, conf) from the possession estimator give a bonus to a segment of that
    team starting near that frame.
  A refinement pass then slides every boundary inside +/- refine_window to the maximum-likelihood
  split point under sharper (unscaled) evidence.
"""

import math

import numpy as np
from numba import njit

from .data import A, D, H, X


def majority_decode(match, p_dead, p_home, window=50):
    """Placeholder. A frame votes H/A when p_dead < 0.5 and p_home is known; majority over
    +/- window frames within the half; D where no frame in the window votes."""
    out = np.full(match.n, X, dtype=np.int8)
    vote = np.full(match.n, -1, dtype=np.int8)
    alive = (p_dead < 0.5) & ~np.isnan(p_home)
    vote[alive] = np.where(p_home[alive] >= 0.5, H, A)
    for _, first, last in match.half_blocks():
        v = vote[first:last + 1]
        hc = np.concatenate(([0], np.cumsum(v == H)))
        ac = np.concatenate(([0], np.cumsum(v == A)))
        i = np.arange(len(v))
        lo, hi = np.maximum(0, i - window), np.minimum(len(v), i + window + 1)
        home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
        seg = np.full(len(v), D, dtype=np.int8)
        seen = (home + away) > 0
        seg[seen] = np.where(home[seen] >= away[seen], H, A)
        out[first:last + 1] = seg
    return out


# --------------------------------------------------------------------------------------------
# HSMM

HSMM_DEFAULTS = dict(
    evidence=0.10,          # weight on per-frame log-lik (frames are strongly autocorrelated)
    eps=0.02,               # clip probabilities to [eps, 1-eps]: one frame is never decisive
    dead_weight=1.0,        # scale of the alive-vs-dead evidence relative to the team evidence
    pd0=0.30,              # prior dead fraction (extval labels are 30-33 % dead); integration sweep: 0.3 beats 0.4 by +0.011
    alive_median_s=8.0,     # log-normal duration prior, H and A (mean = 8*exp(s^2/2) ~ 14.6 s)
    alive_sigma=1.0,
    dead_median_s=13.0,     # D stretches: median 13.4-15 s in the Metrica extval labels (sweep 6/9/13: identical)
    dead_sigma=0.7,
    dmin_alive_s=1.0,       # hard minimum spell: extval labels' shortest alive spell is 0.6-1.0 s, 5th pct 1.4 s; = CHALLENGE 'touch' vs 1 s control (sweep 0.6-1.5: identical)
    dmin_dead_s=3.0,
    dmax_s=80.0,            # duration window; longer segments chain via same-state continuation
    p_turn=0.5,             # share of alive exits that go straight to the other team
    switch_penalty=1.0,     # extra nats on a direct H<->A change
    cont_cost=6.0,          # nats for a same-state continuation (only matters past dmax)
    refine_window_s=4.0,    # boundary search radius; 0 disables refinement
    refine_eps=0.02,
    cp_weight=3.0,          # nats per unit confidence for a change-point bonus
    cp_sigma_s=0.4,         # temporal spread of a change-point bonus
)

_NEG = -1e18


def evidence_ll(p_dead, p_home, pd0=0.25, eps=0.02, weight=1.0, dead_weight=1.0):
    """[T, 3] log-likelihood of (H, A, D) per frame, relative to the prior (0 = uninformative).

    The 3-way posterior q = ((1-pd)*ph, (1-pd)*(1-ph), pd) divided by the prior
    ((1-pd0)/2, (1-pd0)/2, pd0). NaN p_dead -> pd := pd0, NaN p_home -> ph := 0.5, so a frame with
    no opinion yields exactly 0 for all three states and cannot bias towards any of them.
    `dead_weight` scales only the alive-vs-dead part of the evidence (log((1-pd)/(1-pd0)) for H/A,
    log(pd/pd0) for D) relative to the team part log(ph/0.5); 1.0 is the plain posterior."""
    pd = np.asarray(p_dead, dtype=np.float64)
    ph = np.asarray(p_home, dtype=np.float64)
    pd = np.clip(np.where(np.isnan(pd), pd0, pd), eps, 1 - eps)
    ph = np.clip(np.where(np.isnan(ph), 0.5, ph), eps, 1 - eps)
    alive = dead_weight * np.log((1 - pd) / (1 - pd0))
    ll = np.empty((len(pd), 3))
    ll[:, H] = alive + np.log(ph / 0.5)
    ll[:, A] = alive + np.log((1 - ph) / 0.5)
    ll[:, D] = dead_weight * np.log(pd / pd0)
    return weight * ll


def _log_lognormal(d, median, sigma):
    return (-np.log(d) - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            - np.log(d / median) ** 2 / (2 * sigma ** 2))


@njit(cache=True)
def _hsmm_core(ll, logdur, dmin, dmax, logA, init, bonus):
    T = ll.shape[0]
    C = np.zeros((T + 1, 3))
    for t in range(T):
        for s in range(3):
            C[t + 1, s] = C[t, s] + ll[t, s]
    delta = np.full((T, 3), -1e18)
    bd = np.zeros((T, 3), np.int32)
    Bv = np.full((T, 3), -1e18)       # best score with a segment of state s STARTING at t
    Bp = np.zeros((T, 3), np.int8)    # that segment's predecessor state
    for t in range(T):
        for s in range(3):
            if t == 0:
                Bv[0, s] = init[s]
                Bp[0, s] = -1
            else:
                best = -np.inf
                arg = 0
                for sp in range(3):
                    v = delta[t - 1, sp] + logA[sp, s]
                    if sp != s:
                        v += bonus[t, s]
                    if v > best:
                        best = v
                        arg = sp
                Bv[t, s] = best
                Bp[t, s] = arg
        dm = min(dmax, t + 1)
        for s in range(3):
            best = -np.inf
            bdur = 0
            for d in range(dmin[s], dm + 1):
                t0 = t + 1 - d
                v = Bv[t0, s] + C[t + 1, s] - C[t0, s] + logdur[s, d]
                if v > best:
                    best = v
                    bdur = d
            delta[t, s] = best
            bd[t, s] = bdur
    s = 0
    for k in range(1, 3):
        if delta[T - 1, k] > delta[T - 1, s]:
            s = k
    out = np.empty(T, np.int8)
    t = T - 1
    while t >= 0:
        d = bd[t, s]
        if d <= 0:                       # half shorter than dmin: no valid segmentation
            for i in range(t + 1):
                out[i] = s
            break
        for i in range(t - d + 1, t + 1):
            out[i] = s
        sp = Bp[t - d + 1, s]
        t = t - d
        s = sp
    return out


def _bonus_array(T, first, change_points, fps, p):
    bonus = np.zeros((T, 3))
    if change_points is None:
        return bonus
    sig = max(1.0, p["cp_sigma_s"] * fps)
    for frame, team, conf in np.asarray(change_points, dtype=np.float64).reshape(-1, 3):
        rel = frame - first
        lo, hi = int(max(0, rel - 4 * sig)), int(min(T, rel + 4 * sig + 1))
        if hi <= lo or int(team) not in (H, A):
            continue
        t = np.arange(lo, hi)
        bonus[lo:hi, int(team)] += p["cp_weight"] * conf * np.exp(-0.5 * ((t - rel) / sig) ** 2)
    return bonus


def _tables(p, fps):
    dmax = int(round(p["dmax_s"] * fps))
    dmin = np.array([max(1, int(round(p["dmin_alive_s"] * fps)))] * 2
                    + [max(1, int(round(p["dmin_dead_s"] * fps)))], dtype=np.int64)
    d = np.arange(1, dmax + 1, dtype=np.float64)
    logdur = np.full((3, dmax + 1), -np.inf)
    alive = _log_lognormal(d / fps, p["alive_median_s"], p["alive_sigma"])
    dead = _log_lognormal(d / fps, p["dead_median_s"], p["dead_sigma"])
    for s, tab in ((H, alive), (A, alive), (D, dead)):
        logdur[s, 1:] = tab
        logdur[s, :dmin[s]] = -np.inf
    t = p["p_turn"]
    logA = np.full((3, 3), -np.inf)
    logA[H, A] = logA[A, H] = math.log(t) - p["switch_penalty"]
    logA[H, D] = logA[A, D] = math.log(1 - t)
    logA[D, H] = logA[D, A] = math.log(0.5)
    for s in range(3):
        logA[s, s] = -p["cont_cost"]
    init = np.array([math.log(0.3), math.log(0.3), math.log(0.4)])
    return logdur, dmin, dmax, logA, init


def _segments(codes):
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    b = np.concatenate(([0], br, [len(codes)]))
    return [[int(b[i]), int(b[i + 1]), int(codes[b[i]])] for i in range(len(b) - 1)]


def refine_boundaries(codes, ll, window, dmin):
    """Slide each boundary within +/- window to the split maximising ll(left state)+ll(right
    state), keeping every segment >= its dmin. A flat (unobserved) plateau takes its midpoint."""
    if window <= 0:
        return codes
    seg = _segments(codes)
    if len(seg) < 2:
        return codes
    C = np.vstack([np.zeros((1, 3)), np.cumsum(ll, axis=0)])
    for i in range(len(seg) - 1):
        a, b, sl = seg[i]
        e, sr = seg[i + 1][1], seg[i + 1][2]
        lo = max(a + int(dmin[sl]), b - window)
        hi = min(e - int(dmin[sr]), b + window)
        if hi < lo:
            continue
        pos = np.arange(lo, hi + 1)
        score = (C[pos, sl] - C[a, sl]) + (C[e, sr] - C[pos, sr])
        # Unobserved stretches (NaN gaps) make the score exactly flat; the boundary is then
        # unidentifiable, and the midpoint of the plateau minimises the expected timing error.
        flat = score >= score.max() - 1e-9
        k = int(np.argmin(np.abs(pos - b) + (~flat) * 10 ** 9))        # plateau cell nearest b
        left, right = k, k
        while left > 0 and flat[left - 1]:
            left -= 1
        while right < len(pos) - 1 and flat[right + 1]:
            right += 1
        seg[i][1] = seg[i + 1][0] = int(pos[(left + right) // 2])
    out = np.empty_like(codes)
    for a, b, s in seg:
        out[a:b] = s
    return out


def hsmm_decode(match, p_dead, p_home, change_points=None, **params):
    """HSMM decoder (see module docstring). `change_points`: optional array-like of rows
    (frame, team 0/1, confidence 0-1) on the video-frame index. Returns int8[n], X outside halves.
    Parameters and defaults: HSMM_DEFAULTS."""
    p = dict(HSMM_DEFAULTS)
    unknown = set(params) - set(p)
    if unknown:
        raise TypeError("unknown decoder params {}".format(sorted(unknown)))
    p.update(params)
    fps = match.fps
    logdur, dmin, dmax, logA, init = _tables(p, fps)
    out = np.full(match.n, X, dtype=np.int8)
    for _, first, last in match.half_blocks():
        pd_, ph_ = p_dead[first:last + 1], p_home[first:last + 1]
        ll = evidence_ll(pd_, ph_, p["pd0"], p["eps"], p["evidence"], p["dead_weight"])
        bonus = _bonus_array(len(ll), first, change_points, fps, p)
        codes = _hsmm_core(ll, logdur, dmin, dmax, logA, init, bonus)
        w = int(round(p["refine_window_s"] * fps))
        if w > 0:
            ll_sharp = evidence_ll(pd_, ph_, p["pd0"], p["refine_eps"], 1.0, p["dead_weight"])
            codes = refine_boundaries(codes, ll_sharp, w, dmin)
        out[first:last + 1] = codes
    return out
