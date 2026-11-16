"""Possession: given the ball is alive, which team has it?

Contract: every public estimator is `fn(match, **params) -> float32[n]` giving P(home | alive) in
[0, 1], NaN where it has no opinion. Reads only the Match (so it runs on external games too).

Public API
    p_home_nearest(match)          Phase-0 placeholder (nearest tracked object), kept for baselines
    p_home_b(match, **params)      the real estimator (see below)
    control_events(match, **p)     [m, 3] rows (frame, team, confidence): onsets of CONTROLLED
                                   possession by a team other than the previous controller
                                   (candidate change points for the decoder)

Model of `p_home_b`. Possession is defined by control, not by touches: a team keeps the ball until
the opponent *controls* it (CHALLENGE.md). So the estimator is a "last controller" chain:

 1. Ball track -> short gaps interpolated (<= `interp_max_s`), used only to LOCATE a hand-over.
 2. Per frame, per team: distance of the nearest player to the ball and that player's velocity
    relative to the ball. A frame is control evidence for team T when T's nearest player is within
    `r_control` m, the opponent's is at least `margin` m farther, and ball and player move together
    (relative speed < `v_rel`). Contested frames (both teams close) are not evidence for anyone.
 3. Runs of control evidence (bridging blips of <= `join_frames`) lasting >= `min_ctrl_s` are
    control episodes. Anything shorter is a touch / pass-by and is ignored.
 4. Between episodes the owner is the previous controller. Confidence comes from a two-state
    Markov model of possession flips with mean spell length `T_poss_s`:
        p(d) = 0.5 + 0.5 exp(-2 d / T)        P(unchanged after d seconds)
    * ball visible but loose  -> owner = previous controller, confidence p(time since control)
    * ball missing, same team on both sides of the gap -> bridge: p(a)p(b)/p(a+b)
    * ball missing, different teams -> P(prev) = p(a)(1-p(b))/(1-p(a+b)), i.e. a smooth
      hand-over somewhere inside the gap; for gaps short enough to interpolate, the hand-over is
      placed where the interpolated ball first reaches the new team's players.
 5. Log-odds fusion with weak independent evidence: soft nearest-team distance (ball visible, not
    in a fast flight), team-shape (block height, its forward velocity, compactness, width, speed;
    weights fitted on external games, odd features only so there is no home bias) and the Smart
    Tagging phase team (weight set from its agreement with the control chain on unlabelled data).
"""

import numpy as np

from .data import A, H
from .features import attack_sign, nearest_slot, to_metres


# --------------------------------------------------------------------------------------------
# placeholder kept for the baselines


def p_home_nearest(match):
    """Placeholder: 1.0 if the nearest tracked object to the ball is home, 0.0 if away, NaN if
    there is no ball."""
    slot, _ = nearest_slot(match)
    out = np.full(match.n, np.nan, dtype=np.float32)
    ok = slot >= 0
    out[ok] = (match.slot_team[slot[ok]] == H).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------
# parameters

PARAMS = dict(
    interp_max_s=3.0,     # longest ball gap that is interpolated (locating hand-overs only)
    r_control=3.0,        # m, nearest player must be this close for control evidence
    margin=1.0,           # m, opponent must be this much farther
    v_rel=4.0,            # m/s, ball and player must move together
    min_ctrl_s=0.4,       # s, shortest control episode: 10 frames, <= the extval 1 s control rule; sweep 0.2/0.4/0.8 within noise
    join_frames=5,        # blips shorter than this inside one team's control run are bridged
    T_poss_s=13.0,        # s, mean possession length (football prior)
    c_in=0.92,            # confidence inside a control episode
    c_cap=0.97,           # hard cap on chain confidence
    w_near=0.6,           # soft nearest-team evidence weight (log-odds per unit)
    near_scale=3.0,       # m
    near_vmax=15.0,       # m/s, ball faster than this is in flight: no distance evidence
    w_shape=1.0,          # multiplier on the fitted shape logit
    w_tag=1.0,            # log-odds of a tag (logit of 0.76 agreement on unlabelled BAR-ATH)
    shape_win_s=4.0,      # s, smoothing of shape features
)

# Odd shape features (see _shape_features) -> logit weights, fitted on external games (see
# scripts/emissions_b_fit.py and docs/NOTES.md). Order: S, dS, comp, width, speed.
SHAPE_W = np.array([0.03059, 0.32065, 0.05654, 0.18352, -1.34034])

# tag code -> sign relative to the tag's team (+1: that team has the ball, -1: the other has it)
TAG_SIGN = {
    "BUILD UP": 1, "PROGRESSION": 1, "CREATING CHANCES": 1, "ATTACKING TRANSITION": 1,
    "BALL IN FINAL THIRD": 1, "BALL IN THE BOX": 1,
    "DEFENDING IN DEFENSIVE THIRD": -1, "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENDING IN ATTACKING THIRD": -1, "DEFENSIVE TRANSITION": -1,
}


# --------------------------------------------------------------------------------------------
# helpers


def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _interp_ball(ball_m, max_gap):
    """Linear interpolation across interior NaN gaps of at most max_gap frames. Returns
    (ball, interpolated_mask)."""
    out = ball_m.copy()
    obs = ~np.isnan(ball_m[:, 0])
    filled = np.zeros(len(out), dtype=bool)
    for s, e in _runs(~obs):
        if s == 0 or e >= len(out) or e - s > max_gap:
            continue
        w = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
        out[s:e] = ball_m[s - 1] * (1 - w)[:, None] + ball_m[e] * w[:, None]
        filled[s:e] = True
    return out, filled


def _velocity(xy, fps, k=2):
    """Central difference over +/- k frames, NaN where either end is missing. [..., 2] in m/s."""
    v = np.full_like(xy, np.nan)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2.0 * k))
    return v


def _team_geometry(match, ball_m, ball_v, player_m, player_v):
    """Per team: nearest-player distance [n], that player's index, relative speed [n]."""
    out = []
    for team in (H, A):
        cols = np.flatnonzero(match.slot_team == team)
        d = np.sqrt(((player_m[:, cols] - ball_m[:, None, :]) ** 2).sum(-1))
        d_f = np.where(np.isnan(d), np.inf, d)
        idx = np.argmin(d_f, axis=1)
        dist = d_f[np.arange(len(d_f)), idx]
        dist[~np.isfinite(dist)] = np.nan
        pv = player_v[np.arange(len(d_f)), cols[idx]]
        rel = np.sqrt(((pv - ball_v) ** 2).sum(-1))
        out.append((dist, rel))
    return out


def _p_same(d, T):
    return 0.5 + 0.5 * np.exp(-2.0 * np.maximum(d, 0.0) / T)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------------
# control episodes


def _prepare(match, p):
    fps = match.fps
    ball_m = to_metres(match, match.ball)
    player_m = to_metres(match, match.players)
    ball_i, interp = _interp_ball(ball_m, int(round(p["interp_max_s"] * fps)))
    ball_v = _velocity(ball_i, fps)
    player_v = _velocity(player_m, fps)
    geo_raw = _team_geometry(match, ball_m, _velocity(ball_m, fps), player_m, player_v)
    geo_int = _team_geometry(match, ball_i, ball_v, player_m, player_v)
    return dict(ball_m=ball_m, ball_i=ball_i, interp=interp, player_m=player_m, geo_raw=geo_raw,
                geo_int=geo_int, ball_v=_velocity(ball_m, fps))


def _control_frames(geo, p, valid):
    """[n] int8: team with control evidence (0/1) or -1."""
    (dh, rh), (da, ra) = geo
    with np.errstate(invalid="ignore"):
        ok_h = (dh < p["r_control"]) & (da > dh + p["margin"]) & ~(rh >= p["v_rel"])
        ok_a = (da < p["r_control"]) & (dh > da + p["margin"]) & ~(ra >= p["v_rel"])
    out = np.full(len(dh), -1, dtype=np.int8)
    out[ok_h & valid] = H
    out[ok_a & valid] = A
    return out


def _episodes(ctrl, p, fps):
    """[(team, start, end_exclusive)] control episodes from the per-frame control vector."""
    c = ctrl.copy()
    # bridge short blips between runs of the same team
    runs = []
    for team in (H, A):
        runs += [(s, e, team) for s, e in _runs(c == team)]
    runs.sort()
    merged = []
    for s, e, team in runs:
        if merged and merged[-1][2] == team and s - merged[-1][1] <= p["join_frames"] \
                and not np.any(c[merged[-1][1]:s] == 1 - team):
            merged[-1] = (merged[-1][0], e, team)
        else:
            merged.append((s, e, team))
    min_len = max(1, int(round(p["min_ctrl_s"] * fps)))
    return [(team, s, e) for s, e, team in merged
            if (c[s:e] == team).sum() >= min_len]


def _crossing(match, geo_int, prep, ea, sb, team_next, p):
    """Frame in [ea, sb] where an interpolated ball first reaches the new team's players."""
    d_next = geo_int[team_next][0]
    d_prev = geo_int[1 - team_next][0]
    seg = np.arange(ea, sb)
    with np.errstate(invalid="ignore"):
        hit = (d_next[seg] < p["r_control"]) & (d_next[seg] + p["margin"] < d_prev[seg])
    idx = np.flatnonzero(hit & prep["interp"][seg])
    return int(seg[idx[0]]) if len(idx) else sb


# --------------------------------------------------------------------------------------------
# chain of owners with confidence


def _chain(match, p, prep=None):
    """(owner[n] in {0,1,-1}, conf[n], episodes, handovers). conf is P(owner is right)."""
    prep = prep or _prepare(match, p)
    n, fps, T = match.n, match.fps, p["T_poss_s"]
    obs = match.ball_ok
    ctrl = _control_frames(prep["geo_raw"], p, obs)
    eps = _episodes(ctrl, p, fps)
    owner = np.full(n, -1, dtype=np.int8)
    conf = np.full(n, 0.5)
    handovers = []                    # (frame, team, kind) onsets with team != previous controller
    if not eps:
        return owner, conf, eps, handovers
    t = np.arange(n)
    for s, e in ((s, e) for _, s, e in eps):
        owner[s:e] = -2                       # mark as inside (filled below)
    # inside episodes
    for team, s, e in eps:
        owner[s:e] = team
        conf[s:e] = p["c_in"]
    # between / outside
    bounds = [(-1, None)] + [(i, eps[i]) for i in range(len(eps))] + [(len(eps), None)]
    for k in range(len(eps) + 1):
        prev = eps[k - 1] if k > 0 else None
        nxt = eps[k] if k < len(eps) else None
        lo = prev[2] if prev else 0
        hi = nxt[1] if nxt else n
        if hi <= lo:
            continue
        seg = np.arange(lo, hi)
        if prev is None:
            owner[seg] = nxt[0]
            conf[seg] = _p_same((hi - seg) / fps, T)
            continue
        if nxt is None:
            owner[seg] = prev[0]
            conf[seg] = _p_same((seg - lo + 1) / fps, T)
            continue
        a_team, b_team = prev[0], nxt[0]
        miss = ~obs[seg]
        a = (seg - lo + 1) / fps               # seconds since previous control ended
        if a_team == b_team:
            b = (hi - seg) / fps
            two = _p_same(a, T) * _p_same(b, T) / _p_same(a + b, T)
            owner[seg] = a_team
            conf[seg] = np.where(miss, two, _p_same(a, T))
            continue
        # different teams: hand-over at the new team's onset, or earlier if the interpolated ball
        # reaches them inside a short gap
        cross = _crossing(match, prep["geo_int"], prep, lo, hi, b_team, p)
        handovers.append((cross, b_team))
        pre = seg < cross
        owner[seg] = np.where(pre, a_team, b_team)
        b = (hi - seg) / fps
        one_a = _p_same(a, T)
        one_b = _p_same(np.maximum(b, 0), T)
        # long ball-less stretch: smooth two-sided posterior that the previous owner still has it
        tot = a + b
        p_prev = _p_same(a, T) * (1 - _p_same(b, T)) / np.maximum(1 - _p_same(tot, T), 1e-6)
        short = (hi - lo) <= p["interp_max_s"] * fps + 1
        use_post = miss & ~short
        pc = np.where(pre, one_a, one_b)
        conf[seg] = np.where(use_post, np.maximum(p_prev, 1 - p_prev), pc)
        owner[seg] = np.where(use_post, np.where(p_prev >= 0.5, a_team, b_team), owner[seg])
    conf = np.minimum(conf, p["c_cap"])
    # onsets of every change in controller
    prev_team = -1
    out = []
    for team, s, e in eps:
        if team != prev_team and prev_team != -1:
            out.append((s, team))
        prev_team = team
    handovers = out
    return owner, conf, eps, handovers


# --------------------------------------------------------------------------------------------
# other evidence


def _nan_smooth(x, w):
    """Centred moving average over 2w+1 frames ignoring NaN. x [n] or [n, k]."""
    x = np.asarray(x, dtype=np.float64)
    shp = x.shape
    x2 = x.reshape(len(x), -1)
    ok = ~np.isnan(x2)
    cs = np.vstack([np.zeros((1, x2.shape[1])), np.cumsum(np.where(ok, x2, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, x2.shape[1])), np.cumsum(ok, axis=0)])
    i = np.arange(len(x2))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x2), i + w + 1)
    num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(den > 0, num / den, np.nan)
    return res.reshape(shp)


def shape_features(match, win_s=4.0):
    """[n, 5] odd (home <-> away antisymmetric) features of team shape in the home attack frame:
    S  = block height (mean of both teams' median length-position) - 0.5   (+: play in away half)
    dS = forward velocity of that block over +/- win_s  (m/s)
    comp  = length spread of away minus home (the more compact team is pressing)  (m)
    width = lateral spread of home minus away (m)
    speed = mean speed of home minus away outfield players (m/s).
    Smoothed over +/- win_s/2. NaN where a team has < 3 visible players."""
    s = attack_sign(match).astype(np.float64)
    L, W = match.pitch
    x, y = match.players[:, :, 0], match.players[:, :, 1]
    u = np.where(s[:, None] > 0, x, 1 - x) * L
    feats = {}
    for team, name in ((H, "h"), (A, "a")):
        cols = match.slot_team == team
        uu, yy, sp = u[:, cols], y[:, cols] * W, match.speed[:, cols]
        cnt = (~np.isnan(uu)).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(uu, axis=1)
                sdu = np.nanstd(uu, axis=1)
                sdy = np.nanstd(yy, axis=1)
                spm = np.nanmean(sp, axis=1)
        bad = cnt < 3
        for arr in (med, sdu, sdy, spm):
            arr[bad] = np.nan
        feats[name] = (med, sdu, sdy, spm)
    w = max(1, int(round(win_s * match.fps / 2)))
    S = _nan_smooth((feats["h"][0] + feats["a"][0]) / 2 - L / 2, w)
    ww = int(round(win_s * match.fps))
    dS = np.full_like(S, np.nan)
    dS[ww:-ww] = (S[2 * ww:] - S[:-2 * ww]) / (2 * ww / match.fps)
    comp = _nan_smooth(feats["a"][1] - feats["h"][1], w)
    width = _nan_smooth(feats["h"][2] - feats["a"][2], w)
    speed = _nan_smooth(feats["h"][3] - feats["a"][3], w)
    return np.stack([S, dS, comp, width, speed], axis=1)


def shape_logit(match, p=None, weights=None):
    p = {**PARAMS, **(p or {})}
    w = SHAPE_W if weights is None else weights
    f = shape_features(match, p["shape_win_s"])
    z = np.nansum(f * w[None, :], axis=1)
    z[np.all(np.isnan(f), axis=1)] = 0.0
    return z * p["w_shape"]


def tag_evidence(match, p=None):
    """[n] float: +1 where a phase tag says home has the ball, -1 away, 0 no tag. Overlapping
    tags are averaged."""
    num = np.zeros(match.n)
    cnt = np.zeros(match.n)
    for t in match.tags:
        sign = TAG_SIGN.get(t.code)
        if sign is None or t.team not in (H, A):
            continue
        team_has = t.team if sign > 0 else 1 - t.team
        a, b = max(0, t.start_frame), min(match.n, t.end_frame)
        if b > a:
            num[a:b] += 1.0 if team_has == H else -1.0
            cnt[a:b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)


def near_evidence(match, p, prep):
    """[n] soft nearest-team evidence in [-1, 1] (+: home nearer), 0 when ball missing or fast."""
    (dh, _), (da, _) = prep["geo_raw"]
    with np.errstate(invalid="ignore"):
        e = np.tanh((da - dh) / p["near_scale"])
    speed = np.linalg.norm(prep["ball_v"], axis=1)
    e = np.where(np.isnan(e) | (speed > p["near_vmax"]), 0.0, e)
    e[~match.ball_ok] = 0.0
    return e


# --------------------------------------------------------------------------------------------
# public estimators


def p_home_b(match, **params):
    """P(home in possession | ball alive). See module docstring. Never NaN inside the halves."""
    unknown = set(params) - set(PARAMS)
    if unknown:
        raise TypeError("unknown emissions params {}".format(sorted(unknown)))
    p = {**PARAMS, **params}
    prep = _prepare(match, p)
    owner, conf, _, _ = _chain(match, p, prep)
    z = np.zeros(match.n)
    known = owner >= 0
    z[known] = np.where(owner[known] == H, 1.0, -1.0) * _logit(conf[known])
    z += p["w_near"] * near_evidence(match, p, prep)
    z += shape_logit(match, p)
    z += p["w_tag"] * tag_evidence(match, p)
    out = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    out[~match.graded] = np.nan
    return out


def control_events(match, **params):
    """[m, 3] float rows (frame, team, confidence): onsets of controlled possession by a team
    different from the previous controller, on the video-frame index. Onsets inside a ball gap
    are placed where the interpolated ball reaches the new team (short gaps) or at the first
    observed control. Confidence rises with the length of the control run and with how clearly
    the new controller is the nearer team; onsets that follow an unobserved stretch are lower."""
    p = {**PARAMS, **params}
    prep = _prepare(match, p)
    _, _, eps, _ = _chain(match, p, prep)
    fps = match.fps
    (dh, _), (da, _) = prep["geo_raw"]
    rows = []
    prev = None
    for team, s, e in eps:
        if prev is not None and prev[0] != team:
            onset = _crossing(match, prep["geo_int"], prep, prev[2], s, team, p)
            dur = (e - s) / fps
            own, opp = (dh, da) if team == H else (da, dh)
            with np.errstate(invalid="ignore"):
                clear = np.nanmedian(np.tanh((opp[s:e] - own[s:e]) / p["near_scale"]))
            gap = (s - prev[2]) / fps
            c = (1 - np.exp(-dur / 1.0)) * (0.5 + 0.5 * clear) * (0.6 + 0.4 * np.exp(-gap / 5.0))
            rows.append((onset, team, float(np.clip(c, 0.05, 1.0))))
        prev = (team, s, e)
    return np.array(rows, dtype=np.float64).reshape(-1, 3)
