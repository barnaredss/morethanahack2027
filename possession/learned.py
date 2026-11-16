"""Learned team model: a small gradient-boosted classifier stacked on `emissions.p_home_b`.

`p_home_e(match)` -> float32[n] P(home | alive), same contract as every possession estimator.
Features are computed from the Match only (`features_eo`), the model is a tree ensemble stored in
`possession/models/team_gbm.npz` and evaluated in pure numpy (no lightgbm / sklearn at run time),
so the output is byte-deterministic and the submission needs no heavy library or external data.
Training: `scripts/train_learned.py` (LightGBM, fixed seed, one thread, deterministic).

Symmetry. Features are split into EVEN ones (identical when home and away are swapped: ball
visible, gap length, times since control...) and ODD ones (change sign: home-minus-away
differences, signed log-odds, block height in the home attack frame...). Training sees every row
twice (as is, and swapped = odd negated, label flipped) and inference averages the two views, so
there is no home bias by construction.
"""

import os

import numpy as np

from . import emissions as E
from .data import A, H
from .features import attack_sign

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "team_gbm.npz")
NAN_FILL = -999.0

EVEN_NAMES = ["ball_vis", "t_since", "t_until", "in_ep", "gap_s", "d_min", "ball_speed", "n_obj",
              "n_pl_near_sum", "ep_len_prev"]
ODD_NAMES = (["z_b", "z_chain", "prev_sign", "next_sign", "near_ev", "dd", "ctrl_p5", "ctrl_2s",
              "ctrl_5s", "ctrl_past2", "ctrl_lead2", "near_1s", "near_3s",
              "S", "dS", "comp", "width", "speed", "shape_z", "tag", "ball_u", "n_pl_near_diff",
              "rel_diff", "ctrl_past6", "ctrl_lead6"])


def _win_mean(x, lo, hi):
    """Mean of x over frames [t+lo, t+hi] ignoring NaN (lo/hi in frames, may be asymmetric)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0.0], np.cumsum(ok)))
    t = np.arange(len(x))
    a, b = np.clip(t + lo, 0, len(x)), np.clip(t + hi + 1, 0, len(x))
    num, den = cs[b] - cs[a], cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _episode_context(n, eps, fps):
    """Per frame: previous controller team sign (+1 H, -1 A, 0), seconds since its episode ended
    (0 inside), next controller sign, seconds until it starts, inside flag, previous episode length."""
    prev_sign = np.zeros(n)
    nxt_sign = np.zeros(n)
    since = np.full(n, 60.0)
    until = np.full(n, 60.0)
    inside = np.zeros(n)
    plen = np.zeros(n)
    sg = lambda team: 1.0 if team == H else -1.0
    for k, (team, s, e) in enumerate(eps):
        nxt_s = eps[k + 1][1] if k + 1 < len(eps) else n
        prev_sign[s:nxt_s] = sg(team)
        since[s:e] = 0.0
        if nxt_s > e:
            since[e:nxt_s] = np.minimum(60.0, (np.arange(e, nxt_s) - e + 1) / fps)
        plen[s:nxt_s] = (e - s) / fps
        inside[s:e] = 1.0
        prv_e = eps[k - 1][2] if k > 0 else 0
        nxt_sign[prv_e:s] = sg(team)
        until[prv_e:s] = np.minimum(60.0, (s - np.arange(prv_e, s)) / fps)
    return prev_sign, since, nxt_sign, until, inside, plen


def features_eo(match, params=None):
    """(even [n, ke], odd [n, ko]) float64 feature matrices, NaN where undefined."""
    p = {**E.PARAMS, **(params or {})}
    n, fps = match.n, match.fps
    prep = E._prepare(match, p)
    owner, conf, eps, _ = E._chain(match, p, prep)
    (dh, rh), (da, ra) = prep["geo_raw"]
    ctrl = E._control_frames(prep["geo_raw"], p, match.ball_ok)
    chome, caway = (ctrl == H).astype(float), (ctrl == A).astype(float)
    cdiff = chome - caway
    near = E.near_evidence(match, p, prep)
    shape = E.shape_features(match, p["shape_win_s"])
    shape_z = np.where(np.all(np.isnan(shape), axis=1), 0.0, np.nansum(shape * E.SHAPE_W[None, :], 1))
    z_chain = np.zeros(n)
    kn = owner >= 0
    z_chain[kn] = np.where(owner[kn] == H, 1.0, -1.0) * E._logit(conf[kn])
    z_b = np.zeros(n) + z_chain + p["w_near"] * near + shape_z * p["w_shape"] + p["w_tag"] * E.tag_evidence(match, p)
    prev_sign, since, nxt_sign, until, inside, plen = _episode_context(n, eps, fps)
    # missing-run length for each frame
    gap = np.zeros(n)
    for s, e in E._runs(~match.ball_ok):
        gap[s:e] = (e - s) / fps
    sgn = attack_sign(match).astype(float)
    L = match.pitch[0]
    bu = np.where(sgn > 0, match.ball[:, 0], 1 - match.ball[:, 0]) * L - L / 2
    # numbers near the ball (within 10 m of the observed/interpolated ball)
    bi = prep["ball_i"]
    pm = prep["player_m"]
    dist = np.sqrt(((pm - bi[:, None, :]) ** 2).sum(-1))
    near10 = dist < 10.0
    nh = (near10 & (match.slot_team == H)[None, :]).sum(1).astype(float)
    na = (near10 & (match.slot_team == A)[None, :]).sum(1).astype(float)
    nh[np.isnan(bi[:, 0])] = np.nan
    na[np.isnan(bi[:, 0])] = np.nan
    vis = (~np.isnan(match.players[:, :, 0])).sum(1).astype(float)
    dmin = np.fmin(dh, da)
    bspeed = np.linalg.norm(prep["ball_v"], axis=1)
    rel = np.where(np.isnan(rh - ra), np.nan, np.clip(ra - rh, -10, 10))
    f = lambda s: int(round(s * fps))
    dd = np.clip(da - dh, -30, 30)
    even = np.stack([match.ball_ok.astype(float), since, until, inside, gap, np.clip(dmin, 0, 40),
                     np.clip(bspeed, 0, 40), vis, nh + na, plen], axis=1)
    odd = np.stack([
        z_b, z_chain, prev_sign, nxt_sign, near, dd,
        _win_mean(cdiff, -f(0.5), f(0.5)), _win_mean(cdiff, -f(2), f(2)), _win_mean(cdiff, -f(5), f(5)),
        _win_mean(cdiff, -f(2), 0), _win_mean(cdiff, 0, f(2)),
        _win_mean(near, -f(1), f(1)), _win_mean(near, -f(3), f(3)),
        shape[:, 0], shape[:, 1], shape[:, 2], shape[:, 3], shape[:, 4], shape_z,
        E.tag_evidence(match, p), bu, nh - na, rel,
        _win_mean(cdiff, -f(6), 0), _win_mean(cdiff, 0, f(6))], axis=1)
    return even, odd


def assemble(even, odd, swap=False):
    X = np.concatenate([even, -odd if swap else odd], axis=1)
    X = np.round(X, 6) + 0.0             # kills float noise (and -0.0) that could flip a split at ~0
    return np.where(np.isnan(X), NAN_FILL, X)


# --------------------------------------------------------------------------------------------
# numpy tree-ensemble evaluator


def load_model(path=MODEL_PATH):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def predict_raw(model, X):
    """Sum of tree outputs + base score. model arrays are concatenated over trees:
    feat/thr/left/right/value of length total_nodes, `root` [T] offsets; leaves have feat = -1
    and `value` holds the leaf output; internal nodes go left when x[feat] <= thr."""
    feat, thr, left, right, value = (model[k] for k in ("feat", "thr", "left", "right", "value"))
    out = np.full(len(X), float(model["base"]), dtype=np.float64)
    for root in model["root"]:
        node = np.full(len(X), root, dtype=np.int64)
        for _ in range(int(model["depth"])):
            f = feat[node]
            internal = f >= 0
            if not internal.any():
                break
            xv = X[np.arange(len(X)), np.where(internal, f, 0)]
            nxt = np.where(xv <= thr[node], left[node], right[node])
            node = np.where(internal, nxt, node)
        out += value[node]
    return out


def p_home_e(match, model=None, params=None, mask_alive_only=False):
    """P(home | alive) from the committed GBM, symmetrised. NaN outside the halves."""
    model = model if model is not None else load_model()
    even, odd = features_eo(match, params)
    g = np.flatnonzero(match.graded)
    z1 = predict_raw(model, assemble(even[g], odd[g], False))
    z2 = predict_raw(model, assemble(even[g], odd[g], True))
    z = 0.5 * (z1 - z2)                  # logit of the swap-averaged view
    out = np.full(match.n, np.nan, dtype=np.float32)
    out[g] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return out
