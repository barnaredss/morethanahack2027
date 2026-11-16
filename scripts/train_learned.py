"""Train / evaluate the learned team model (possession/learned.py). EXTERNAL data only.

    python scripts/train_learned.py logo     # leave-one-game-out on Metrica 1-3 x seeds 0-2
    python scripts/train_learned.py final    # fit on all 3 games x seeds 0-2, write the npz
    python scripts/train_learned.py check    # retrain twice, compare model files byte for byte

Needs lightgbm (requirements-train.txt). Run-time inference needs only numpy.
Deterministic: fixed seeds, one thread, deterministic=True, row subsampling by stride.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import deadball, decode, emissions as E, learned as Ln  # noqa: E402
from possession.data import D, H, CACHE_DIR  # noqa: E402
from possession.scoring import score  # noqa: E402
from extval import degrade as dg  # noqa: E402
from extval.harness import degraded  # noqa: E402
from extval.datasets import DEFAULT_GAMES  # noqa: E402

GAMES = list(DEFAULT_GAMES)
DEV_SEEDS = (0, 1, 2)
STRIDE = 4
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=24, max_depth=5, min_child_samples=300,
           feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
           seed=20271117, bagging_seed=1, feature_fraction_seed=2, data_random_seed=3,
           deterministic=True, force_row_wise=True, num_threads=1, use_missing=False,
           zero_as_missing=False, max_bin=63)
N_TREES = 250


def _key(name, seed, tag):
    files = (Ln.__file__, E.__file__, deadball.__file__, dg.__file__, dg.CAL_PATH, dg.FIT_PATH)
    parts = [name, seed, tag] + [os.path.getmtime(f) for f in files]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:12]


def item(name, seed, match=None, gt=None, tag="deg"):
    """Features + labels + baseline probabilities for one run (cached on disk)."""
    path = os.path.join(CACHE_DIR, "learned_feats", "{}_{}_{}_{}.npz".format(tag, name, seed, _key(name, seed, tag)))
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    if match is None:
        match, gt, _ = degraded(name, seed)
    even, odd = Ln.features_eo(match)
    it = dict(even=even.astype(np.float32), odd=odd.astype(np.float32), gt=gt.astype(np.int8),
              p_dead=deadball.p_dead_a(match), p_b=E.p_home_b(match))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **it)
    return it


def dataset(items, stride=STRIDE):
    Xs, ys = [], []
    for it in items:
        idx = np.flatnonzero(it["gt"] <= 1)[::stride]
        e, o = it["even"][idx].astype(np.float64), it["odd"][idx].astype(np.float64)
        y = (it["gt"][idx] == H).astype(float)
        Xs += [Ln.assemble(e, o, False), Ln.assemble(e, o, True)]
        ys += [y, 1 - y]
    return np.concatenate(Xs), np.concatenate(ys)


def train(items, n_trees=N_TREES, params=None):
    import lightgbm as lgb
    X, y = dataset(items)
    return lgb.train({**LGB, **(params or {})}, lgb.Dataset(X, y), num_boost_round=n_trees)


def p_from_booster(bst, n, it):
    e, o = it["even"].astype(np.float64), it["odd"].astype(np.float64)
    sel = np.flatnonzero(it["gt"] != 3)           # graded frames (gt X outside halves)
    z1 = bst.predict(Ln.assemble(e[sel], o[sel], False), raw_score=True)
    z2 = bst.predict(Ln.assemble(e[sel], o[sel], True), raw_score=True)
    out = np.full(n, np.nan, dtype=np.float32)
    out[sel] = (1 / (1 + np.exp(-0.5 * (z1 - z2)))).astype(np.float32)
    return out


def team_acc(it, p):
    alive = it["gt"] <= 1
    truth = it["gt"] == H
    vis = it["even"][:, 0] > 0.5
    ok = ~np.isnan(p)

    def a(m):
        return float((((p >= .5) == truth)[m & ok]).mean())
    return a(alive & vis), a(alive & ~vis)


def eval_run(match, it, p_home, oracle_dead=False, cp=None, **dp):
    pd_ = (it["gt"] == D).astype(np.float32) if oracle_dead else it["p_dead"]
    pred = decode.hsmm_decode(match, pd_, p_home, change_points=cp, **dp)
    r = score(it["gt"], pred)
    return dict(spell=r["spell_f1_mean"], macro=r["macro_f1"], alive=r["alive_dead_f1"],
                n_pred=r["n_pred_spells"], n_gt=r["n_gt_spells"])


def mean_rows(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}


def export(bst, path):
    dump = bst.dump_model()
    feat, thr, left, right, value, root = [], [], [], [], [], []
    depth = 0

    def walk(node, d):
        nonlocal depth
        idx = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(idx)
        right.append(idx)
        value.append(0.0)
        if "leaf_value" in node:
            value[idx] = node["leaf_value"]
            depth = max(depth, d)
            return idx
        assert node["decision_type"] == "<=", node["decision_type"]
        feat[idx] = node["split_feature"]
        thr[idx] = node["threshold"]
        left[idx] = walk(node["left_child"], d + 1)
        right[idx] = walk(node["right_child"], d + 1)
        return idx

    for t in dump["tree_info"]:
        root.append(len(feat))
        walk(t["tree_structure"], 0)
    np.savez(path, feat=np.array(feat, np.int32), thr=np.array(thr, np.float64), left=np.array(left, np.int32),
             right=np.array(right, np.int32), value=np.array(value, np.float64), root=np.array(root, np.int32),
             base=np.array(0.0), depth=np.array(depth), names=np.array(Ln.EVEN_NAMES + Ln.ODD_NAMES))
    return os.path.getsize(path)


def importance(bst):
    names = Ln.EVEN_NAMES + Ln.ODD_NAMES
    gain = bst.feature_importance("gain")
    order = np.argsort(-gain)
    return [(names[i], round(float(gain[i] / gain.sum()), 3)) for i in order[:12]]


def main(cmd):
    runs = {(g, s): degraded(g, s) for g in GAMES for s in DEV_SEEDS}
    its = {k: item(k[0], k[1], runs[k][0], runs[k][1]) for k in runs}
    if cmd == "logo":
        rows = {"v1": [], "ho": [], "ins": []}
        accs = {"v1": [], "ho": []}
        keys = list(its)
        bst_all = train(list(its.values()))
        print("importance (all-game fit):", importance(bst_all))
        for g in GAMES:
            bst = train([its[k] for k in keys if k[0] != g])
            for s in DEV_SEEDS:
                m, it = runs[(g, s)][0], its[(g, s)]
                p_ho = p_from_booster(bst, m.n, it)
                p_in = p_from_booster(bst_all, m.n, it)
                rows["v1"].append(eval_run(m, it, it["p_b"]))
                rows["ho"].append(eval_run(m, it, p_ho))
                rows["ins"].append(eval_run(m, it, p_in))
                accs["v1"].append(team_acc(it, it["p_b"]))
                accs["ho"].append(team_acc(it, p_ho))
        for k in rows:
            print(k, mean_rows(rows[k]))
        for k in accs:
            print("team acc alive vis/missing", k, np.round(np.mean(accs[k], 0), 4))
        for gi, g in enumerate(GAMES):
            sel = range(gi * len(DEV_SEEDS), (gi + 1) * len(DEV_SEEDS))
            print(g, {k: mean_rows([rows[k][i] for i in sel])["spell"] for k in rows})
    elif cmd == "final":
        bst = train(list(its.values()))
        size = export(bst, Ln.MODEL_PATH)
        print("wrote", Ln.MODEL_PATH, size, "bytes", hashlib.sha256(open(Ln.MODEL_PATH, "rb").read()).hexdigest())
    elif cmd == "check":
        import tempfile
        hs = []
        for i in range(2):
            p = os.path.join(tempfile.gettempdir(), "tm{}.npz".format(i))
            export(train(list(its.values())), p)
            hs.append(hashlib.sha256(open(p, "rb").read()).hexdigest())
        print("identical" if hs[0] == hs[1] else "DIFFERENT", hs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logo")
