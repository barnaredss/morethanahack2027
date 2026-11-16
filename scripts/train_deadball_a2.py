"""Train / evaluate the A2 dead-ball model (boosted trees, numpy only) on EXTERNAL games.

    $PY scripts/train_deadball_a2.py logo [--trees 100 --depth 4 ...]   # leave-one-game-out report
    $PY scripts/train_deadball_a2.py fit                                  # fit on all, write the model

Training rows: every `stride`-th graded frame of Metrica 1-3 x seeds 0-2 (simulator v2), label =
extval truth is D. Deterministic (no randomness). Inference lives in possession/deadball.py.
No BAR-ATH data or labels are read.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from extval.datasets import DEFAULT_GAMES  # noqa: E402
from extval.harness import degraded  # noqa: E402
from possession import deadball as db  # noqa: E402
from possession.data import CACHE_DIR, D  # noqa: E402

SEEDS = (0, 1, 2)
DEFAULTS = dict(trees=100, depth=4, lr=0.1, lam=10.0, min_leaf=100, bins=32, stride=3)


def fit_gbdt(X, y, trees, depth, lr, lam, min_leaf, bins):
    n, F = X.shape
    edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1])) for j in range(F)]
    B = np.stack([np.searchsorted(edges[j], X[:, j], side="left") for j in range(F)], 1).astype(np.int16)
    nb = max(len(e) for e in edges) + 1
    I, Lf = 2 ** depth - 1, 2 ** depth
    p0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    base = np.log(p0 / (1 - p0))
    z = np.full(n, base)
    feat = np.zeros((trees, I), dtype=np.int16)
    thr = np.full((trees, I), np.inf, dtype=np.float32)
    leaf = np.zeros((trees, Lf))
    rows = np.arange(n)
    for t in range(trees):
        p = 1.0 / (1.0 + np.exp(-z))
        g, h = p - y, np.maximum(p * (1 - p), 1e-6)
        idx = np.zeros(n, dtype=np.int64)
        for lev in range(depth):
            nn = 2 ** lev
            loc = idx - (nn - 1)
            best = np.full(nn, -np.inf)
            bj = np.zeros(nn, dtype=np.int64)
            bb = np.full(nn, nb, dtype=np.int64)
            for j in range(F):
                key = loc * nb + B[:, j]
                G = np.bincount(key, g, nn * nb).reshape(nn, nb).cumsum(1)
                Hh = np.bincount(key, h, nn * nb).reshape(nn, nb).cumsum(1)
                C = np.bincount(key, None, nn * nb).reshape(nn, nb).cumsum(1)
                Gt, Ht, Ct = G[:, -1:], Hh[:, -1:], C[:, -1:]
                gain = G ** 2 / (Hh + lam) + (Gt - G) ** 2 / (Ht - Hh + lam) - Gt ** 2 / (Ht + lam)
                gain = np.where((C >= min_leaf) & (Ct - C >= min_leaf), gain, -np.inf)
                gain[:, len(edges[j]):] = -np.inf          # bins beyond this feature's edges
                b = gain.argmax(1)
                v = gain[np.arange(nn), b]
                upd = v > best
                best = np.where(upd, v, best)
                bj = np.where(upd, j, bj)
                bb = np.where(upd, b, bb)
            ok = np.isfinite(best) & (best > 1e-9)
            for k in range(nn):
                node = nn - 1 + k
                if ok[k]:
                    feat[t, node] = bj[k]
                    thr[t, node] = edges[bj[k]][bb[k]]
                    bb_k = bb[k]
                else:
                    feat[t, node], thr[t, node] = 0, np.inf   # everything goes left
            fsel = feat[t, idx]
            go = np.where(np.isinf(thr[t, idx]), False,
                          X[rows, fsel] > thr[t, idx])
            idx = 2 * idx + 1 + go
        loc = idx - I
        G = np.bincount(loc, g, Lf)
        H = np.bincount(loc, h, Lf)
        leaf[t] = -G / (H + lam)
        z += lr * leaf[t][loc]
    return dict(feat=feat, thr=thr, leaf=leaf, lr=np.float64(lr), base=np.float64(base))


def build_items(games=DEFAULT_GAMES, seeds=SEEDS):
    path = os.path.join(CACHE_DIR, "a2_items_{}.pkl".format(hash_src()))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    items = []
    for g in games:
        for s in seeds:
            m, gt, _ = degraded(g, s)
            items.append(dict(game=g, seed=s, graded=m.graded, gt=gt, X=db.frame_features2(m)))
    with open(path, "wb") as fh:
        pickle.dump(items, fh, protocol=4)
    return items


def hash_src():
    import hashlib
    h = hashlib.sha1()
    for f in ("deadball.py",):
        h.update(open(os.path.join(ROOT, "possession", f), "rb").read())
    return h.hexdigest()[:10]


def rows(items, stride):
    Xs, ys = [], []
    for it in items:
        g = np.flatnonzero(it["graded"])[::stride]
        Xs.append(it["X"][g])
        ys.append((it["gt"][g] == D).astype(float))
    return np.vstack(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["logo", "fit"])
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, type=type(v), default=v)
    ap.add_argument("--out", default=db.A2_MODEL_PATH)
    a = ap.parse_args()
    hp = {k: getattr(a, k) for k in DEFAULTS if k != "stride"}
    items = build_items()
    if a.cmd == "fit":
        X, y = rows(items, a.stride)
        t = time.time()
        model = fit_gbdt(X, y, **hp)
        np.savez(a.out, **model)
        print("fitted on {} rows in {:.0f}s -> {}".format(len(y), time.time() - t, a.out))
        return
    preds = {}
    from possession import decode, emissions  # noqa: E402
    from possession.scoring import score  # noqa: E402
    for held in DEFAULT_GAMES:
        tr = [i for i in items if i["game"] != held]
        te = [i for i in items if i["game"] == held]
        model = fit_gbdt(*rows(tr, a.stride), **hp)
        res = []
        for it in te:
            m, _, _ = degraded(it["game"], it["seed"])
            p = np.full(m.n, np.nan, dtype=np.float32)
            gm = m.graded
            p[gm] = db.sigmoid(db.gbdt_predict(model, it["X"][gm]))
            preds[(it["game"], it["seed"])] = p.copy()
            y = it["gt"] == D
            f1 = 2 * ((p[gm] > .5) & y[gm]).sum() / ((p[gm] > .5).sum() + y[gm].sum())
            pc = db.clean_stretches(p, m)
            pred = decode.hsmm_decode(m, pc, emissions.p_home_b(m))
            sc = score(it["gt"], pred, bootstrap_n=0)
            res.append((f1, sc["spell_f1_mean"], sc["macro_f1"], sc["alive_dead_f1"]))
        with open(os.path.join(CACHE_DIR, "a2_logo_preds.pkl"), "wb") as fh:
            pickle.dump(preds, fh)
        print(held, "frameF1 %.3f spell %.4f macro %.4f alive %.4f" % tuple(np.mean(res, 0)), flush=True)


if __name__ == "__main__":
    main()
