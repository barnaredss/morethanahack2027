"""Fit the odd team-shape logit weights (SHAPE_W) on EXTERNAL games, with a chain+near offset.
Leave-one-game-out check. Deterministic (ridge IRLS, no randomness). Never touches BAR-ATH GT."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import emissions
from possession.data import A, H
from extval.harness import degraded
from extval.datasets import DEFAULT_GAMES

def collect(g, s):
    m, gt, _ = degraded(g, s)
    z0 = emissions.p_home_b(m, w_shape=0.0, w_tag=0.0).astype(np.float64)
    z0 = np.log(np.clip(z0, 1e-4, 1 - 1e-4) / (1 - np.clip(z0, 1e-4, 1 - 1e-4)))
    f = emissions.shape_features(m)
    alive = ((gt == H) | (gt == A))
    keep = alive & ~np.isnan(f).any(1)
    return f[keep], (gt[keep] == H).astype(float), z0[keep], m.ball_ok[keep]

def fit(F, y, off, lam=1.0, it=25):
    sd = F.std(0) + 1e-9
    X = F / sd
    w = np.zeros(F.shape[1])
    for _ in range(it):
        z = off + X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + lam * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w / sd

def nll(F, y, off, w):
    z = off + F @ w; p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

if __name__ == "__main__":
    data = {g: [collect(g, s) for s in (0, 1, 2)] for g in DEFAULT_GAMES}
    cat = lambda gs: [np.concatenate([d[i] for g in gs for d in data[g]]) for i in range(4)]
    for g in DEFAULT_GAMES:
        tr = cat([x for x in DEFAULT_GAMES if x != g]); te = cat([g])
        w = fit(tr[0], tr[1], tr[2])
        mis = ~te[3]
        print(g, "heldout nll all %.4f -> %.4f | missing %.4f -> %.4f" % (
            nll(te[0], te[1], te[2], 0 * w), nll(te[0], te[1], te[2], w),
            nll(te[0][mis], te[1][mis], te[2][mis], 0 * w), nll(te[0][mis], te[1][mis], te[2][mis], w)),
            "acc missing %.3f -> %.3f" % (
            (((te[2][mis]) >= 0) == (te[1][mis] > .5)).mean(), ((te[2][mis] + te[0][mis] @ w >= 0) == (te[1][mis] > .5)).mean()))
    allw = fit(*cat(DEFAULT_GAMES)[:3])
    print("ALL w =", np.array2string(allw, precision=5, separator=", "))
