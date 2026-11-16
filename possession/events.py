"""Event layer: detect "team X gains controlled possession" onsets from tracking.

The external labels are made from event streams (control event of a new team that holds for ~1 s =
possession change, see extval/labels.py). This module learns the onset detector directly:

    event_scores(match, model)  -> (sH, sA)  float32[n]   P(onset by team X within +/- k frames)
    pick_events(match, sH, sA)  -> [m, 3]    rows (frame, team, confidence)   (decoder change_points)
    team_extra(...)             -> (even, odd) extra columns for the v3 team model (option b)

Same shipping pattern as `learned.py`: a LightGBM tree ensemble exported to npz and evaluated in
numpy (`learned.predict_raw`). Features are computed from a `Match` only. Symmetry: a row is built
from the point of view of one team ("own" / "opp" columns, odd columns negated), so the home and
away detectors are the same model on swapped views and there is no home bias by construction.
"""

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import emissions as E
from . import learned as Ln
from .data import A, H
from .features import to_metres

EV_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "event_gbm.npz")
TEAM3_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "team_gbm_ev.npz")
K_TARGET = 8                # label = an onset within +/- K_TARGET frames (0.32 s)
NAN_FILL = Ln.NAN_FILL

SHARED_NAMES = ["sp_p8", "sp_f8", "sp_p25", "sp_f25", "acc8", "dsp8", "cos8", "turn8", "acc16", "ball_miss17"]
OWN_NAMES = ["d0", "d_m12", "d_m6", "d_p6", "d_p12", "d_p25", "dmin_p12", "dmin_f12", "rel0", "rel_p10",
             "rel_m10", "n5", "n3", "closing"]
ODD_EV_NAMES = ["dd_m12", "dd_m6", "dd_p6", "dd_p12", "dd_p25", "dd_m25", "dd_min_f12", "dd_min_p12"]


def _sh(x, k):
    """x[t + k] with NaN fill, along axis 0."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    n = len(x)
    if k > 0:
        out[:n - k] = x[k:]
    elif k < 0:
        out[-k:] = x[:n + k]
    else:
        out[:] = x
    return out


def _win_min(x, lo, hi, big=60.0):
    """min of x over [t+lo, t+hi] (NaN ignored; NaN where nothing is observed in the window)."""
    n = len(x)
    W = max(abs(lo), abs(hi))
    xx = np.concatenate([np.full(W, big), np.where(np.isnan(x), big, x), np.full(W, big)])
    win = sliding_window_view(xx, hi - lo + 1)[W + lo:W + lo + n]
    m = win.min(axis=1)
    return np.where(m >= big, np.nan, m)


def _speed_over(ball, a, b, fps):
    """Mean velocity vector over frames [t+a, t+b] from positions: (p[t+b]-p[t+a])/(b-a)*fps."""
    return (_sh(ball, b) - _sh(ball, a)) * (fps / float(b - a))


def event_features(match, base=None, params=None):
    """(even [n, ke], own (2, [n, ko]) per team, odd [n, kd]) float64, NaN where undefined.
    `base` = precomputed (even, odd) of `learned.features_eo(match)` (shared, avoids recomputing)."""
    p = {**E.PARAMS, **(params or {})}
    n, fps = match.n, match.fps
    f = lambda s: int(round(s * fps))
    be, bo = base if base is not None else Ln.features_eo(match, params)
    prep = E._prepare(match, p)
    ball = prep["ball_i"]
    # ball kinematics around t (interpolated ball): past / future mean velocity, impulse, turn
    vp, vf = _speed_over(ball, -8, 0, fps), _speed_over(ball, 0, 8, fps)
    vp2, vf2 = _speed_over(ball, -25, -8, fps), _speed_over(ball, 8, 25, fps)
    spp, spf = np.linalg.norm(vp, axis=1), np.linalg.norm(vf, axis=1)
    sp2p, sp2f = np.linalg.norm(vp2, axis=1), np.linalg.norm(vf2, axis=1)
    acc = np.linalg.norm(vf - vp, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(vp * vf, axis=1) / (spp * spf)
        cos = np.where((spp > 1.5) & (spf > 1.5), cos, np.nan)
    acc16 = np.linalg.norm(vf2 - vp2, axis=1)
    miss = Ln._win_mean((~match.ball_ok).astype(float), -f(0.34), f(0.34))
    shared = np.stack([spp, spf, sp2p, sp2f, acc, spf - spp, cos, np.arccos(np.clip(cos, -1, 1)), acc16, miss], axis=1)
    # per-team nearest-player kinematics
    player_m = prep["player_m"]
    dist_all = np.sqrt(((player_m - ball[:, None, :]) ** 2).sum(-1))
    own = []
    d_clip = []
    for team, (dist, rel) in zip((H, A), prep["geo_int"]):
        d = np.clip(dist, 0, 30)
        cols = match.slot_team == team
        dt = dist_all[:, cols]
        n5 = np.nansum(dt < 5.0, axis=1).astype(float)
        n3 = np.nansum(dt < 3.0, axis=1).astype(float)
        n5[np.isnan(ball[:, 0])] = np.nan
        n3[np.isnan(ball[:, 0])] = np.nan
        r = np.clip(rel, 0, 20)
        own.append(np.stack([
            d, _sh(d, -f(0.5)), _sh(d, -f(0.25)), _sh(d, f(0.25)), _sh(d, f(0.5)), _sh(d, f(1.0)),
            _win_min(d, -f(0.5), 0), _win_min(d, 0, f(0.5)),
            r, Ln._win_mean(r, 0, f(0.4)), Ln._win_mean(r, -f(0.4), 0), n5, n3,
            _sh(d, -f(0.5)) - _sh(d, f(0.5))], axis=1))
        d_clip.append(d)
    dH, dA = d_clip
    dd = lambda k: _sh(dA, k) - _sh(dH, k)
    odd = np.stack([dd(-f(0.5)), dd(-f(0.25)), dd(f(0.25)), dd(f(0.5)), dd(f(1.0)), dd(-f(1.0)),
                    _win_min(dA, 0, f(0.5)) - _win_min(dH, 0, f(0.5)),
                    _win_min(dA, -f(0.5), 0) - _win_min(dH, -f(0.5), 0)], axis=1)
    even = np.concatenate([be, shared], axis=1)
    return even, own, np.concatenate([bo, odd], axis=1)


def view(even, own, odd, team, sel=None, swap_sign=None):
    """Design matrix for 'onset by `team`': [even, own[team], own[other], +/-odd] (rounded, NaN filled)."""
    s = slice(None) if sel is None else sel
    sign = 1.0 if team == H else -1.0
    X = np.concatenate([even[s], own[team][s], own[1 - team][s], sign * odd[s]], axis=1)
    X = np.round(X, 6) + 0.0
    return np.where(np.isnan(X), NAN_FILL, X)


def feature_names():
    return (Ln.EVEN_NAMES + SHARED_NAMES + ["own_" + x for x in OWN_NAMES] + ["opp_" + x for x in OWN_NAMES]
            + Ln.ODD_NAMES + ODD_EV_NAMES)


# --------------------------------------------------------------------------------------------
# targets (training only; derived from labels by the training script)


def onset_targets(gt, k=K_TARGET):
    """gt int8 codes -> (yH, yA) float arrays: 1 where a spell of that team starts within +/- k frames
    (any previous state: a restart or a turnover), only inside the graded halves; plus onset frames."""
    n = len(gt)
    on = {H: [], A: []}
    for team in (H, A):
        isx = gt == team
        prev = np.concatenate(([False], isx[:-1]))
        valid = np.concatenate(([False], gt[:-1] != 3))        # not the first frame of the block
        on[team] = np.flatnonzero(isx & ~prev & valid)
    ys = []
    for team in (H, A):
        y = np.zeros(n)
        for t in on[team]:
            y[max(0, t - k):t + k + 1] = 1.0
        ys.append(y)
    return ys[0], ys[1], on


# --------------------------------------------------------------------------------------------
# inference


def load_model(path=EV_MODEL_PATH):
    return Ln.load_model(path)


def event_scores(match, model=None, base=None, feats=None):
    """(sH, sA) float32[n]: probability that team X gains control within +/- K_TARGET frames.
    NaN outside the graded halves. `feats` = precomputed `event_features` output."""
    model = model if model is not None else load_model()
    even, own, odd = feats if feats is not None else event_features(match, base)
    g = np.flatnonzero(match.graded)
    out = []
    for team in (H, A):
        z = Ln.predict_raw(model, view(even, own, odd, team, g))
        s = np.full(match.n, np.nan, dtype=np.float32)
        s[g] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        out.append(s)
    return out[0], out[1]


def pick_events(match, sH, sA, thr=0.2, nms_s=0.5, smooth_s=0.12):
    """Peak-pick the two score tracks to discrete events. A peak is the maximum of the smoothed score
    within +/- nms_s and above thr. Returns [m, 3] float rows (frame, team, confidence = score)."""
    fps = match.fps
    w = max(1, int(round(nms_s * fps)))
    sm = max(0, int(round(smooth_s * fps)))
    rows = []
    for team, s in ((H, sH), (A, sA)):
        for _, a, b in match.half_blocks():
            x = np.nan_to_num(np.asarray(s[a:b + 1], dtype=np.float64), nan=0.0)
            if sm:
                x = Ln._win_mean(x, -sm, sm)
            win = sliding_window_view(np.concatenate([np.zeros(w), x, np.zeros(w)]), 2 * w + 1)
            peak = (x >= win.max(axis=1)) & (x >= thr)
            # a plateau gives several equal maxima: keep the first
            idx = np.flatnonzero(peak)
            last = -10 ** 9
            for i in idx:
                if i - last > w:
                    rows.append((a + i, team, float(x[i])))
                    last = i
    rows.sort()
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def match_events(pred_rows, true_on, tol):
    """Greedy one-to-one matching of picked events to true onsets of the same team within +/- tol.
    pred_rows [m,3]; true_on {team: array of frames}. Returns (tp, n_pred, n_true) per team dict."""
    res = {}
    for team in (H, A):
        pf = pred_rows[pred_rows[:, 1] == team][:, 0] if len(pred_rows) else np.zeros(0)
        conf = pred_rows[pred_rows[:, 1] == team][:, 2] if len(pred_rows) else np.zeros(0)
        tf = np.asarray(true_on[team])
        used = np.zeros(len(tf), dtype=bool)
        tp = 0
        for i in np.argsort(-conf):
            if not len(tf):
                break
            d = np.abs(tf - pf[i]).astype(float)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tol:
                used[j] = True
                tp += 1
        res[team] = (tp, len(pf), len(tf))
    return res


# --------------------------------------------------------------------------------------------
# option (b): event-derived columns for the team model


EXTRA_EVEN = ["since_any", "until_any", "conf_prev_any", "conf_next_any", "s_sum_win2"]
EXTRA_ODD = ["last_sign", "next_sign", "since_diff", "until_diff", "cs_diff", "s_diff_m05", "s_diff_p05",
             "s_diff_past2", "s_diff_next2"]


def team_extra(match, sH, sA, rows=None):
    """(even [n, 5], odd [n, 9]) columns derived from the event scores and the picked events.
    odd columns flip sign under a home<->away swap (sH <-> sA); even do not."""
    n, fps = match.n, match.fps
    f = lambda s: int(round(s * fps))
    rows = pick_events(match, sH, sA) if rows is None else rows
    big = 30.0
    t = np.arange(n)
    since = {H: np.full(n, big), A: np.full(n, big)}
    until = {H: np.full(n, big), A: np.full(n, big)}
    cprev = {H: np.zeros(n), A: np.zeros(n)}
    cnext = {H: np.zeros(n), A: np.zeros(n)}
    prox = {H: np.zeros(n), A: np.zeros(n)}
    for team in (H, A):
        r = rows[rows[:, 1] == team] if len(rows) else rows
        fr = r[:, 0].astype(int) if len(r) else np.zeros(0, int)
        cf = r[:, 2] if len(r) else np.zeros(0)
        if len(fr):
            i = np.searchsorted(fr, t, side="right") - 1          # last event at or before t
            ok = i >= 0
            since[team][ok] = np.minimum(big, (t[ok] - fr[i[ok]]) / fps)
            cprev[team][ok] = cf[i[ok]]
            j = np.searchsorted(fr, t, side="left")               # first event at or after t
            ok = j < len(fr)
            until[team][ok] = np.minimum(big, (fr[j[ok]] - t[ok]) / fps)
            cnext[team][ok] = cf[j[ok]]
            for a, c in zip(fr, cf):
                lo, hi = max(0, a - f(1.5)), min(n, a + f(1.5) + 1)
                prox[team][lo:hi] = np.maximum(prox[team][lo:hi], c * np.exp(-np.abs(np.arange(lo, hi) - a) / f(0.5)))
    last_sign = np.where(since[H] < since[A], 1.0, np.where(since[A] < since[H], -1.0, 0.0))
    next_sign = np.where(until[H] < until[A], 1.0, np.where(until[A] < until[H], -1.0, 0.0))
    sh, sa = np.nan_to_num(sH.astype(np.float64)), np.nan_to_num(sA.astype(np.float64))
    sd = sh - sa
    since_any, until_any = np.minimum(since[H], since[A]), np.minimum(until[H], until[A])
    cp_any = np.where(since[H] < since[A], cprev[H], cprev[A])
    cn_any = np.where(until[H] < until[A], cnext[H], cnext[A])
    even = np.stack([since_any, until_any, cp_any, cn_any, Ln._win_mean(sh + sa, -f(2), f(2))], axis=1)
    odd = np.stack([last_sign, next_sign, np.clip(since[A] - since[H], -big, big),
                    np.clip(until[A] - until[H], -big, big), prox[H] - prox[A],
                    Ln._win_mean(sd, -f(0.5), 0), Ln._win_mean(sd, 0, f(0.5)),
                    Ln._win_mean(sd, -f(2), 0), Ln._win_mean(sd, 0, f(2))], axis=1)
    return even, odd


def p_home_e3(match, ev_model=None, team_model=None, base=None, feats=None, rows_out=None):
    """Option (b): the v3 team model = learned team features + event-derived columns. P(home | alive)."""
    ev_model = ev_model if ev_model is not None else load_model()
    team_model = team_model if team_model is not None else Ln.load_model(TEAM3_MODEL_PATH)
    even, odd = base if base is not None else Ln.features_eo(match)
    sH, sA = event_scores(match, ev_model, base=(even, odd), feats=feats)
    xe, xo = team_extra(match, sH, sA)
    g = np.flatnonzero(match.graded)
    e2, o2 = np.concatenate([even, xe], axis=1), np.concatenate([odd, xo], axis=1)
    z1 = Ln.predict_raw(team_model, Ln.assemble(e2[g], o2[g], False))
    z2 = Ln.predict_raw(team_model, Ln.assemble(e2[g], o2[g], True))
    out = np.full(match.n, np.nan, dtype=np.float32)
    out[g] = (1.0 / (1.0 + np.exp(-0.5 * (z1 - z2)))).astype(np.float32)
    return out
