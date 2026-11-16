"""Dead ball: is the ball out of play?

Contract: every public estimator is `fn(match, **params) -> float32[n]` giving P(dead) in [0, 1],
NaN where it has no opinion (here: outside the graded halves). Reads only the Match.

`p_dead_a` is the real detector. Per frame it builds a handful of physical / football signals
(see `frame_features`), combines them with a logistic model (`DEFAULT_WEIGHTS`, fitted on external
validation games, see docs/NOTES.md) and sharpens stretch edges with `refine_edges`.
`p_dead_ball_missing` is the Phase-0 placeholder ("no ball = dead"), kept for the baselines.
"""

import warnings

import numpy as np
import pandas as pd

from .data import A, H


def p_dead_ball_missing(match):
    """Placeholder: 1 where the ball is not tracked, 0 where it is."""
    return np.where(match.ball_ok, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------
# helpers


def _roll_mean(x, w):
    """Centred NaN-aware rolling mean over w frames (NaN where the window has no value)."""
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=1).mean().to_numpy()


def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def dist_to_line(match, xy_m):
    """Signed distance in metres from the pitch outline, positive inside, negative outside."""
    L, W = match.pitch
    x, y = xy_m[..., 0], xy_m[..., 1]
    return np.minimum(np.minimum(x, L - x), np.minimum(y, W - y))


def team_speed(match, smooth_s=1.0):
    """[n] median player speed (m/s) over tracked players, smoothed. Slots with a NaN speed fall
    back to the speed implied by position differences (the ATD leaves ~10 % of speeds empty)."""
    L, W = match.pitch
    pos = match.players * np.array([L, W])
    d = np.full(match.speed.shape, np.nan)
    d[1:-1] = np.sqrt(((pos[2:] - pos[:-2]) ** 2).sum(-1)) * match.fps / 2.0
    sp = np.where(np.isnan(match.speed), d, match.speed)
    cnt = (~np.isnan(sp)).sum(1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # rows with no tracked player make nanmedian warn "All-NaN slice"; they are masked by cnt
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.where(cnt >= 4, np.nanmedian(sp, axis=1), np.nan)
    return _roll_mean(med, max(1, int(round(smooth_s * match.fps))))


def _baseline(x, fps, win_s, step=25):
    """Rolling median of x over win_s seconds, computed on a subsample and interpolated."""
    idx = np.arange(0, len(x), step)
    sub = pd.Series(x[idx]).rolling(max(3, int(win_s * fps / step)), center=True,
                                    min_periods=3).median().to_numpy()
    ok = ~np.isnan(sub)
    if not ok.any():
        return np.full(len(x), np.nan)
    return np.interp(np.arange(len(x)), idx[ok], sub[ok])


# --------------------------------------------------------------------------------------------
# kickoff detection (algorithmic: no frame numbers)


def own_half_fraction(match, team, frames):
    """Fraction of observed players of `team` standing in their own half on each of `frames`."""
    L = match.pitch[0]
    out = np.full(len(frames), np.nan)
    sel = match.slot_team == team
    x = match.players[frames][:, sel, 0] * L
    right = np.array([match.home_attacks_right[int(match.half[f])] for f in frames])
    own_left = right if team == H else ~right        # that team's own half is x < L/2
    inside = np.where(own_left[:, None], x < L / 2, x > L / 2)
    cnt = (~np.isnan(x)).sum(1)
    with np.errstate(all="ignore"):
        out = np.where(cnt >= 5, (inside & ~np.isnan(x)).sum(1) / np.maximum(cnt, 1), np.nan)
    return out


def detect_kickoffs(match, play_ratio=1.25, play_window_s=10.0, own_half_min=0.60,
                    backtrack_s=8.0, ramp_rise=0.5, search_s=600.0, min_pre_s=10.0, spd=None):
    """{half: kickoff_frame}. The kickoff is where sustained play begins: the first moment the
    team speed over the next `play_window_s` stays above `play_ratio` x the half's median speed,
    taken back (at most `backtrack_s` seconds) to the last frame still within `ramp_rise` m/s of the
    preceding lull before the peak of the next `play_window_s`, and where, at the preceding lull, both
    teams stand mostly in their own halves (`own_half_min`). Halves where nothing qualifies
    within `search_s` of the first graded frame, or where the kickoff would be less than `min_pre_s`
    after it (too short to tell from a lull), get their first graded frame.

    Parameter origins (docs/NOTES.md, "Kickoff parameters"): football priors, with `own_half_min` and
    `min_pre_s` checked on the external Metrica games only (scripts/kickoff_extval.py); nothing here
    was fitted or adjusted on BAR-ATH."""
    spd = team_speed(match) if spd is None else spd
    out = {}
    fps = match.fps
    w = int(play_window_s * fps)
    for h, first, last in match.half_blocks():
        seg = spd[first:last + 1]
        if np.isnan(seg).all():
            out[h] = first
            continue
        med = np.nanmedian(seg)
        fwd = pd.Series(seg[::-1]).rolling(w, min_periods=w // 2).mean().to_numpy()[::-1]
        stop = min(len(seg), int(search_s * fps))
        hot = np.flatnonzero(fwd[:stop] >= play_ratio * med)
        found = first
        prev_end = -1
        if len(hot) and hot[0] == 0:            # play is already under way on the first frame
            out[h] = first
            continue
        for t in hot:
            if t <= prev_end:
                continue
            lo = max(0, t - int(backtrack_s * fps))
            top_i = t + int(np.nanargmax(np.nan_to_num(seg[t:t + w], nan=-np.inf)))
            low = np.nanmin(seg[lo:t + 1])
            level = low + ramp_rise                       # where the ramp has just begun
            below = np.flatnonzero(np.nan_to_num(seg[lo:top_i + 1], nan=np.inf) <= level)
            k = lo + int(below[-1]) if len(below) else t
            fr = np.array([first + lo + int(np.argmin(np.nan_to_num(seg[lo:t + 1], nan=np.inf)))])
            ok_h = own_half_fraction(match, H, fr)[0]
            ok_a = own_half_fraction(match, A, fr)[0]
            if (np.isnan(ok_h) or ok_h >= own_half_min) and (np.isnan(ok_a) or ok_a >= own_half_min):
                found = first + k
                break
            prev_end = t + w
        out[h] = int(found) if found - first >= min_pre_s * fps else first
    return out


# --------------------------------------------------------------------------------------------
# per-frame features

FEATURE_NAMES = ("bias", "missing", "gap_len", "gap_short", "gap_edge_last", "gap_edge_next",
                 "gap_tag", "gap_slow", "oob", "slow", "still", "pre_kickoff")


def gap_table(match):
    """Per ball-missing run inside a half: (start, end_exclusive, d_last, d_next) where d_* is the
    distance in metres of the last / next observed ball position from the pitch outline (NaN when
    the run touches the half edge)."""
    L, W = match.pitch
    bm = match.ball * np.array([L, W])
    ok = match.ball_ok
    rows = []
    for h, first, last in match.half_blocks():
        for s, e in _runs(~ok[first:last + 1]):
            s += first
            e += first
            dl = dist_to_line(match, bm[s - 1]) if s > first else np.nan
            dn = dist_to_line(match, bm[e]) if e <= last else np.nan
            rows.append((s, e, dl, dn))
    return rows


def _tag_overlap(match, gaps_s, gaps_e, codes=("SET PIECES", "GOALS"), pad_s=10.0):
    """For each gap: 1 if a set-piece / goal tag overlaps [start - pad, end + pad]."""
    pad = int(pad_s * match.fps)
    iv = [(t.start_frame, t.end_frame) for t in match.tags if t.code in codes]
    out = np.zeros(len(gaps_s))
    for i, (s, e) in enumerate(zip(gaps_s, gaps_e)):
        out[i] = any(a < e + pad and b > s - pad for a, b in iv)
    return out


def frame_features(match, edge_scale_m=6.0, oob_margin_m=0.5, slow_win_s=60.0, short_gap_s=2.0,
                   still_win_s=1.0, still_thr_m=1.5, spd=None, kickoffs=None):
    """[n, len(FEATURE_NAMES)] features, NaN rows outside the graded halves."""
    n, fps = match.n, match.fps
    L, W = match.pitch
    X = np.zeros((n, len(FEATURE_NAMES)))
    X[:, 0] = 1.0
    ok = match.ball_ok
    bm = match.ball * np.array([L, W])

    spd = team_speed(match) if spd is None else spd
    base = _baseline(np.nan_to_num(spd, nan=np.nanmedian(spd)), fps, slow_win_s)
    ratio = spd / np.maximum(base, 0.3)

    # missing + gap structure
    X[:, 1] = ~ok
    gaps = gap_table(match)
    if gaps:
        gs = np.array([g[0] for g in gaps])
        ge = np.array([g[1] for g in gaps])
        tag = _tag_overlap(match, gs, ge)
        for (s, e, dl, dn), tg in zip(gaps, tag):
            len_s = (e - s) / fps
            X[s:e, 2] = np.log1p(len_s)
            X[s:e, 3] = np.exp(-len_s / short_gap_s)
            X[s:e, 4] = np.exp(-max(dl, 0) / edge_scale_m) if not np.isnan(dl) else 0.0
            X[s:e, 5] = np.exp(-max(dn, 0) / edge_scale_m) if not np.isnan(dn) else 0.0
            X[s:e, 6] = tg
            r = np.nanmean(ratio[s:e]) if np.isfinite(ratio[s:e]).any() else 1.0
            X[s:e, 7] = np.clip(1.0 - r, -1.0, 1.0)

    # ball observed outside the pitch, sustained a few frames
    d = np.full(n, np.nan)
    d[ok] = dist_to_line(match, bm[ok])
    out = ok & (d < -oob_margin_m)
    keep = np.zeros(n, dtype=bool)
    for s, e in _runs(out):
        if e - s >= 5:
            keep[s:e] = True
    X[:, 8] = keep

    # players slow down relative to the local norm (whistle)
    X[:, 9] = np.nan_to_num(1.0 - np.minimum(np.clip(ratio, 0.0, 2.0), 1.0), nan=0.0)

    # ball stationary (observed at both ends of +/- still_win_s)
    k = max(1, int(still_win_s * fps))
    a, b = np.roll(bm, k, axis=0), np.roll(bm, -k, axis=0)
    mv = np.maximum(np.sqrt(((bm - a) ** 2).sum(1)), np.sqrt(((bm - b) ** 2).sum(1)))
    still = ok & ~np.isnan(mv) & (mv < still_thr_m)
    X[:, 10] = still

    # before the half's kickoff
    kickoffs = detect_kickoffs(match, spd=spd) if kickoffs is None else kickoffs
    for h, first, last in match.half_blocks():
        X[first:kickoffs.get(h, first), 11] = 1.0

    X[~match.graded] = np.nan
    return X


# Logistic weights fitted on extval (Metrica 1-3 x seeds 0-2, IRLS, L2 = 1), see docs/NOTES.md,
# section A. Order = FEATURE_NAMES. pre_kickoff is set by prior: extval halves start at the kickoff,
# so there is nothing to fit it on.
DEFAULT_WEIGHTS = np.array([-2.8, 3.9, -1.0, -6.0, 2.6, 0.0, 1.2, 2.7, 1.4, 1.4, 1.4, 5.0])
assert len(DEFAULT_WEIGHTS) == len(FEATURE_NAMES)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def clean_stretches(p, match, close_s=1.5, min_dead_s=1.5, hi=0.85, lo=0.15):
    """Close alive holes shorter than `close_s` inside dead stretches and drop dead islands shorter
    than `min_dead_s` (inside each half). Frames that change are set to `hi` / `lo`."""
    out = p.copy()
    for _, first, last in match.half_blocks():
        seg = out[first:last + 1]
        dead = seg > 0.5
        for s, e in _runs(~dead):
            if 0 < s and e < len(seg) and (e - s) < close_s * match.fps:
                seg[s:e] = hi
        dead = seg > 0.5
        for s, e in _runs(dead):
            if (e - s) < min_dead_s * match.fps:
                seg[s:e] = lo
    return out


def p_dead_a(match, weights=None, clean=True, **params):
    """P(ball out of play) per frame; NaN outside the graded halves. `params` go to
    `frame_features` (edge_scale_m, oob_margin_m, slow_win_s, short_gap_s, still_win_s, ...)."""
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights)
    X = frame_features(match, **params)
    p = np.full(match.n, np.nan, dtype=np.float32)
    g = match.graded
    p[g] = sigmoid(X[g] @ w)
    return clean_stretches(p, match) if clean else p


# --------------------------------------------------------------------------------------------
# A2: richer features + a small boosted-tree model (pure numpy inference)
#
# `p_dead_a2` adds what `p_dead_a` lacks: (i) ball VISIBLE but dead (ball held / carried slowly while
# everybody slows down, ball near a line), (ii) collective-stop timing (team speed before vs after
# each frame, fraction of players walking), (iii) the restart touch (ball still before, moving
# after), (iv) context: how far the nearest gap is and what it looks like, whether the whole tracking
# is in an outage. The model is a depth-4 boosted-tree ensemble (`deadball_a2_model.npz`,
# regenerated by scripts/train_deadball_a2.py).

import os as _os

A2_MODEL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "deadball_a2_model.npz")

A2_EXTRA_NAMES = (
    "vis_edge", "lmv_f1", "lmv_b1", "lmv_c4", "vis_f", "vis_b", "near_player",
    "frac_slow2", "r2", "r6", "r_before4", "r_after4", "r_delta",
    "miss_ctx3", "miss_ctx10", "nobj", "nobj_ctx3",
    "nxt_dt", "nxt_len", "nxt_edge", "prv_dt", "prv_len", "prv_edge", "gap_in", "gap_out",
)
A2_FEATURE_NAMES = FEATURE_NAMES[1:] + A2_EXTRA_NAMES      # bias dropped (trees)


def _win_mean(x, lo, hi):
    """Mean of x over frames [t+lo, t+hi] (NaN-aware, window clipped at the array ends)."""
    n = len(x)
    v = np.nan_to_num(x)
    c = np.concatenate(([0.0], np.cumsum(v)))
    k = np.concatenate(([0.0], np.cumsum(~np.isnan(x))))
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    cnt = k[b] - k[a]
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, (c[b] - c[a]) / np.maximum(cnt, 1), np.nan)


def _player_speeds(match):
    L, W = match.pitch
    pos = match.players * np.array([L, W])
    d = np.full(match.speed.shape, np.nan)
    d[1:-1] = np.sqrt(((pos[2:] - pos[:-2]) ** 2).sum(-1)) * match.fps / 2.0
    return np.where(np.isnan(match.speed), d, match.speed)


def frame_features2(match, slow_win_s=60.0, edge_scale_m=6.0, ctx_s=10.0):
    """[n, len(A2_FEATURE_NAMES)] finite features (NaN rows outside the graded halves)."""
    n, fps = match.n, match.fps
    L, W = match.pitch
    X1 = frame_features(match)
    ok = match.ball_ok
    bm = match.ball * np.array([L, W])
    spd = team_speed(match)
    base = _baseline(np.nan_to_num(spd, nan=np.nanmedian(spd)), fps, slow_win_s)
    ratio = np.clip(spd / np.maximum(base, 0.3), 0.0, 2.0)
    slow = 1.0 - np.minimum(ratio, 1.0)
    cols = {}

    # ball visible: line proximity, speed before / after / around, held by a player
    d = np.full(n, np.nan)
    d[ok] = dist_to_line(match, bm[ok])
    cols["vis_edge"] = np.where(ok, np.exp(-np.maximum(np.nan_to_num(d), 0) / edge_scale_m), 0.0)

    def disp(k):
        """Displacement (m) between frame t and t+k (k may be negative); NaN if either missing."""
        other = np.full_like(bm, np.nan)
        if k > 0:
            other[:-k] = bm[k:]
        else:
            other[-k:] = bm[:k]
        return np.sqrt(((other - bm) ** 2).sum(1))

    f1, b1 = disp(fps), disp(-fps)
    cols["lmv_f1"] = np.nan_to_num(np.log1p(f1))
    cols["lmv_b1"] = np.nan_to_num(np.log1p(b1))
    c4 = np.sqrt(((np.roll(bm, -2 * fps, 0) - np.roll(bm, 2 * fps, 0)) ** 2).sum(1))
    cols["lmv_c4"] = np.nan_to_num(np.log1p(c4))
    cols["vis_f"] = ~np.isnan(f1)
    cols["vis_b"] = ~np.isnan(b1)
    pos = match.players * np.array([L, W])
    with np.errstate(all="ignore"):
        dist = np.sqrt(((pos - bm[:, None, :]) ** 2).sum(-1))
        near = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    cols["near_player"] = np.where(ok & np.isfinite(near), np.exp(-near / 1.5), 0.0)

    # collective stop: share of players walking, speed ratio at several scales, before vs after
    sp = _player_speeds(match)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # frames with no tracked player give NaN ("Mean of empty slice"), handled by nan_to_num below
        warnings.simplefilter("ignore", RuntimeWarning)
        walking = np.where(np.isnan(sp), np.nan, (sp < 0.8).astype(float))
        frac = np.nanmean(walking, axis=1)
    cols["frac_slow2"] = np.nan_to_num(_win_mean(frac, -fps, fps), nan=0.0)
    cols["r2"] = np.nan_to_num(_win_mean(slow, -fps, fps))
    cols["r6"] = np.nan_to_num(_win_mean(slow, -3 * fps, 3 * fps))
    rb, ra = _win_mean(slow, -4 * fps, 0), _win_mean(slow, 0, 4 * fps)
    cols["r_before4"], cols["r_after4"] = np.nan_to_num(rb), np.nan_to_num(ra)
    cols["r_delta"] = np.nan_to_num(rb - ra)

    # context: ball gaps nearby, tracking outage
    miss = (~ok).astype(float)
    cols["miss_ctx3"] = _win_mean(miss, -3 * fps, 3 * fps)
    cols["miss_ctx10"] = _win_mean(miss, -int(ctx_s * fps), int(ctx_s * fps))
    nobj = (~np.isnan(match.players[:, :, 0])).sum(1) / 22.0
    cols["nobj"] = np.clip(nobj, 0, 1.2)
    cols["nobj_ctx3"] = np.clip(_win_mean(nobj, -3 * fps, 3 * fps), 0, 1.2)

    far = ctx_s
    for k in ("nxt_dt", "prv_dt"):
        cols[k] = np.full(n, far)
    for k in ("nxt_len", "nxt_edge", "prv_len", "prv_edge", "gap_in", "gap_out"):
        cols[k] = np.zeros(n)
    reach = int(ctx_s * fps)
    gaps = gap_table(match)
    for s, e, dl, dn in reversed(gaps):                       # nearest next gap wins
        a = max(0, s - reach)
        t = np.arange(a, s)
        cols["nxt_dt"][a:s] = (s - t) / fps
        cols["nxt_len"][a:s] = np.log1p((e - s) / fps)
        cols["nxt_edge"][a:s] = np.exp(-max(dl, 0) / edge_scale_m) if not np.isnan(dl) else 0.0
    for s, e, dl, dn in gaps:                                 # nearest previous gap wins
        b = min(n, e + reach)
        t = np.arange(e, b)
        cols["prv_dt"][e:b] = (t - e + 1) / fps
        cols["prv_len"][e:b] = np.log1p((e - s) / fps)
        cols["prv_edge"][e:b] = np.exp(-max(dn, 0) / edge_scale_m) if not np.isnan(dn) else 0.0
        cols["gap_in"][s:e] = np.log1p(np.arange(e - s) / fps)
        cols["gap_out"][s:e] = np.log1p(np.arange(e - s, 0, -1) / fps)

    X = np.column_stack([X1[:, 1:]] + [np.asarray(cols[k], dtype=float) for k in A2_EXTRA_NAMES])
    X = np.nan_to_num(X, nan=0.0)
    X[~match.graded] = np.nan
    return X


def gbdt_predict(model, X):
    """Logit of a depth-d boosted ensemble. model: dict with feat[T, I] int, thr[T, I], leaf[T, 2^d],
    lr, base. Trees are complete heaps; `X > thr` goes right."""
    feat, thr, leaf = model["feat"], model["thr"], model["leaf"]
    T, I = feat.shape
    depth = int(np.log2(I + 1))
    z = np.full(len(X), float(model["base"]))
    rows = np.arange(len(X))
    for t in range(T):
        idx = np.zeros(len(X), dtype=np.int64)
        for _ in range(depth):
            go = X[rows, feat[t, idx]] > thr[t, idx]
            idx = 2 * idx + 1 + go
        z += float(model["lr"]) * leaf[t, idx - I]
    return z


_A2_MODEL = {}


def load_a2_model(path=None):
    path = path or A2_MODEL_PATH
    if path not in _A2_MODEL:
        z = np.load(path)
        _A2_MODEL[path] = {k: z[k] for k in z.files}
    return _A2_MODEL[path]


def p_dead_a2(match, model=None, clean=True, **params):
    """P(ball out of play) per graded frame from the A2 boosted-tree model; NaN outside the halves.
    Does not touch `p_dead_a`."""
    model = load_a2_model() if model is None else model
    X = frame_features2(match, **params)
    p = np.full(match.n, np.nan, dtype=np.float32)
    g = match.graded
    p[g] = sigmoid(gbdt_predict(model, X[g]))
    pre = np.zeros(match.n, dtype=bool)
    pre[g] = X[g][:, A2_FEATURE_NAMES.index("pre_kickoff")] > 0.5     # prior, nothing to learn it on
    p[pre] = np.maximum(p[pre], 0.98)
    return clean_stretches(p, match) if clean else p
