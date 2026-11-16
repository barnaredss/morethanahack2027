"""Train / evaluate the event layer (possession/events.py). EXTERNAL data only, no BAR-ATH labels.

    python scripts/train_events.py det     # leave-one-game-out event-detection precision / recall
    python scripts/train_events.py logo    # full LOGO: v2 vs (a) change points vs (b) event team model
    python scripts/train_events.py final   # fit event model + v3 team model on all 3 games, write npz
    python scripts/train_events.py check   # retrain twice, compare files byte for byte

Needs lightgbm (requirements-train.txt). Deterministic: fixed seeds, one thread, stride sampling.
Stacking: the v3 team model is trained on OUT-OF-FOLD event scores (each training game scored by an
event model trained on the other games), so it never sees in-sample detector output.
"""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_learned as T  # noqa: E402
from possession import events as Ev, learned as Ln  # noqa: E402
from possession.data import A, H, CACHE_DIR  # noqa: E402

LGB_EV = {**T.LGB, "min_child_samples": 200, "num_leaves": 16, "max_depth": 5}
N_TREES_EV = 150
STRIDE_EV = 2
TOL = 12                      # +/- 0.5 s


def ev_item(name, seed, it, match, tag="deg"):
    """Event feature matrices for one run (cached): even, own0, own1, odd as float32."""
    key = T._key(name, seed, tag)
    f = lambda m: os.path.getmtime(m.__file__)
    key2 = "{}_{}".format(key, int(f(Ev) * 1000))
    path = os.path.join(CACHE_DIR, "learned_feats", "ev_{}_{}_{}_{}.npz".format(tag, name, seed, key2))
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    even, own, odd = Ev.event_features(match, base=(it["even"].astype(np.float64), it["odd"].astype(np.float64)))
    out = dict(even=even.astype(np.float32), own0=own[0].astype(np.float32), own1=own[1].astype(np.float32),
               odd=odd.astype(np.float32))
    np.savez(path, **out)
    return out


def feats_of(ev):
    return ev["even"].astype(np.float64), [ev["own0"].astype(np.float64), ev["own1"].astype(np.float64)], \
        ev["odd"].astype(np.float64)


def ev_dataset(runs, stride=STRIDE_EV):
    Xs, ys = [], []
    for r in runs:
        gt = r["it"]["gt"]
        yH, yA, _ = Ev.onset_targets(gt)
        idx = np.flatnonzero(gt != 3)[::stride]
        e, o, d = feats_of(r["ev"])
        for team, y in ((H, yH), (A, yA)):
            Xs.append(Ev.view(e, o, d, team, idx))
            ys.append(y[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def train_ev(runs, n_trees=N_TREES_EV, params=None):
    import lightgbm as lgb
    X, y = ev_dataset(runs)
    return lgb.train({**LGB_EV, **(params or {})}, lgb.Dataset(X, y), num_boost_round=n_trees)


def scores_from(bst, r):
    m = r["match"]
    e, o, d = feats_of(r["ev"])
    g = np.flatnonzero(r["it"]["gt"] != 3)
    out = []
    for team in (H, A):
        z = bst.predict(Ev.view(e, o, d, team, g), raw_score=True)
        s = np.full(m.n, np.nan, dtype=np.float32)
        s[g] = (1 / (1 + np.exp(-z))).astype(np.float32)
        out.append(s)
    return out


def load_runs(seeds=T.DEV_SEEDS, games=T.GAMES, tag="deg", degr=None):
    runs = {}
    for g in games:
        for s in seeds:
            m, gt, _ = (degr or T.degraded)(g, s)
            it = T.item(g, s, m, gt, tag=tag)
            runs[(g, s)] = dict(match=m, it=it, ev=ev_item(g, s, it, m, tag))
    return runs


def pr_table(runs, scores, thr, tol=TOL):
    """Precision / recall at +/- tol over the given run keys, split into all / turnover onsets."""
    tp = npred = ntrue = 0
    tp_t = ntrue_t = 0
    for k, r in runs.items():
        sH, sA = scores[k]
        m = r["match"]
        rows = Ev.pick_events(m, sH, sA, thr=thr)
        _, _, on = Ev.onset_targets(r["it"]["gt"])
        res = Ev.match_events(rows, on, tol)
        for team in (H, A):
            a, b, c = res[team]
            tp, npred, ntrue = tp + a, npred + b, ntrue + c
        gt = r["it"]["gt"]
        # turnover onsets: previous frame was the OTHER team (not a dead-ball restart)
        on_t = {t: np.array([x for x in on[t] if gt[x - 1] == 1 - t]) for t in (H, A)}
        res_t = Ev.match_events(rows, on_t, tol)
        for team in (H, A):
            tp_t += res_t[team][0]
            ntrue_t += res_t[team][2]
    return dict(thr=thr, precision=tp / max(npred, 1), recall=tp / max(ntrue, 1), n_pred=npred, n_true=ntrue,
                recall_turnover=tp_t / max(ntrue_t, 1), n_turn=ntrue_t)


# --------------------------------------------------------------------------------------------
# dead-ball (A2) fold models and the full LOGO


_A2 = {}


def a2_fold(held):
    """A2 model fitted on the other games (seeds 0-2), as train_deadball_a2.py logo does."""
    import train_deadball_a2 as A2
    if "items" not in _A2:
        _A2["items"] = A2.build_items()
    if held not in _A2:
        tr = [i for i in _A2["items"] if i["game"] != held]
        hp = {k: v for k, v in A2.DEFAULTS.items() if k != "stride"}
        _A2[held] = A2.fit_gbdt(*A2.rows(tr, A2.DEFAULTS["stride"]), **hp)
    return _A2[held]


def a2_pred(model, m):
    from possession import deadball as db
    X = db.frame_features2(m)
    p = np.full(m.n, np.nan, dtype=np.float32)
    g = m.graded
    p[g] = db.sigmoid(db.gbdt_predict(model, X[g]))
    return db.clean_stretches(p, m)


def with_extra(r, sH, sA):
    xe, xo = Ev.team_extra(r["match"], sH, sA)
    it = r["it"]
    return dict(even=np.concatenate([it["even"], xe.astype(np.float32)], 1),
                odd=np.concatenate([it["odd"], xo.astype(np.float32)], 1), gt=it["gt"])


def oof_scores(runs, games):
    """Out-of-fold event scores for every run of `games` (each game scored by a model trained on the others)."""
    res = {}
    for g in games:
        bst = train_ev([r for k, r in runs.items() if k[0] in games and k[0] != g])
        for k, r in runs.items():
            if k[0] == g:
                res[k] = scores_from(bst, r)
    return res


def dec(m, gt, pdead, p_home, cp=None, **dp):
    from possession import decode
    from possession.scoring import score
    r = score(gt, decode.hsmm_decode(m, pdead, p_home, change_points=cp, evidence=0.2, **dp))
    return dict(spell=r["spell_f1_mean"], macro=r["macro_f1"], alive=r["alive_dead_f1"], n_pred=r["n_pred_spells"],
                n_gt=r["n_gt_spells"])


def logo(train_runs, test_runs, variants_cp=((0.2, 3.0), (0.2, 6.0), (0.3, 3.0), (0.3, 6.0))):
    out = {}
    scores = {}
    folds = {}
    for g in T.GAMES:
        others = [x for x in T.GAMES if x != g]
        tr = {k: r for k, r in train_runs.items() if k[0] in others}
        bst_ev = train_ev(list(tr.values()))
        bst_v2 = T.train([r["it"] for r in tr.values()])
        oof = oof_scores(tr, others)
        v3_items = [with_extra(tr[k], *oof[k]) for k in tr]
        bst_v3 = T.train(v3_items)
        a2 = a2_fold(g)
        folds[g] = (bst_ev, bst_v2, bst_v3, a2)
        for k, r in test_runs.items():
            if k[0] != g:
                continue
            m, it = r["match"], r["it"]
            sH, sA = scores_from(bst_ev, r)
            scores[k] = (sH, sA)
            pd = a2_pred(a2, m)
            ph2 = T.p_from_booster(bst_v2, m.n, it)
            rows = {}
            rows["v2"] = dec(m, it["gt"], pd, ph2)
            for thr, w in variants_cp:
                cp = Ev.pick_events(m, sH, sA, thr=thr)
                rows["a thr%.1f w%.0f" % (thr, w)] = dec(m, it["gt"], pd, ph2, cp, cp_weight=w)
            r3 = with_extra(r, sH, sA)
            ph3 = T.p_from_booster(bst_v3, m.n, r3)
            rows["b"] = dec(m, it["gt"], pd, ph3)
            cp = Ev.pick_events(m, sH, sA, thr=0.3)
            rows["b+a thr0.3 w3"] = dec(m, it["gt"], pd, ph3, cp, cp_weight=3.0)
            for name, v in rows.items():
                out.setdefault(name, []).append((g, k[1], v))
        print("fold", g, "done", flush=True)
    return out, scores, folds


def report(out):
    for name, rows in out.items():
        print("==", name)
        for g in T.GAMES:
            print("  ", g, T.mean_rows([v for gg, _, v in rows if gg == g]))
        print("   MEAN", T.mean_rows([v for _, _, v in rows]))


def save_model(bst, path, names):
    T.export(bst, path)
    z = dict(np.load(path))
    z["names"] = np.array(names)
    np.savez(path, **z)
    return os.path.getsize(path)


def fit_final(runs, out_ev, out_team):
    bst_ev = train_ev(list(runs.values()))
    oof = oof_scores(runs, T.GAMES)
    bst_t = T.train([with_extra(runs[k], *oof[k]) for k in runs])
    s1 = save_model(bst_ev, out_ev, Ev.feature_names())
    s2 = save_model(bst_t, out_team, Ln.EVEN_NAMES + Ev.EXTRA_EVEN + Ln.ODD_NAMES + Ev.EXTRA_ODD)
    return s1, s2


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def secondary(runs):
    """Clean Metrica (LOGO fold models, seed-0 tags) and SkillCorner (all-3 fit). Not used for selection."""
    from dataclasses import replace
    from extval.datasets import load_game, sc_match_ids
    from possession import deadball as db
    _, _, folds = logo(runs, {}, variants_cp=())
    print("== clean Metrica, LOGO fold models")
    rows = {"v2": [], "b": []}
    for g in T.GAMES:
        bst_ev, bst_v2, bst_v3, a2 = folds[g]
        deg, gt, _ = T.degraded(g, 0)
        clean = replace(load_game(g).match, tags=deg.tags)
        it = T.item(g, 0, clean, gt, tag="clean")
        r = dict(match=clean, it=it, ev=ev_item(g, 0, it, clean, "clean"))
        sH, sA = scores_from(bst_ev, r)
        pd = a2_pred(a2, clean)
        rows["v2"].append(dec(clean, gt, pd, T.p_from_booster(bst_v2, clean.n, it)))
        rows["b"].append(dec(clean, gt, pd, T.p_from_booster(bst_v3, clean.n, with_extra(r, sH, sA))))
        print(g, {k: round(v[-1]["spell"], 4) for k, v in rows.items()}, flush=True)
    for k, v in rows.items():
        print("clean MEAN", k, T.mean_rows(v))
    print("== SkillCorner, models fit on all 3 degraded Metrica games (committed-style), seed 0")
    bst_ev = train_ev(list(runs.values()))
    oof = oof_scores(runs, T.GAMES)
    bst_v3 = T.train([with_extra(runs[k], *oof[k]) for k in runs])
    bst_v2 = T.train([r["it"] for r in runs.values()])
    rows = {"v2": [], "b": []}
    for i in sc_match_ids()[:4]:
        name = "sc%s" % i
        m, gt, _ = T.degraded(name, 0)
        it = T.item(name, 0, m, gt)
        r = dict(match=m, it=it, ev=ev_item(name, 0, it, m, "deg"))
        sH, sA = scores_from(bst_ev, r)
        pd = db.p_dead_a2(m)
        rows["v2"].append(dec(m, gt, pd, T.p_from_booster(bst_v2, m.n, it)))
        rows["b"].append(dec(m, gt, pd, T.p_from_booster(bst_v3, m.n, with_extra(r, sH, sA))))
        print(name, {k: round(v[-1]["spell"], 4) for k, v in rows.items()}, flush=True)
    for k, v in rows.items():
        print("SC MEAN", k, T.mean_rows(v))


def main(cmd):
    import pickle
    runs = load_runs()
    keys = list(runs)
    if cmd == "logo":
        out, scores, _ = logo(runs, runs)
        report(out)
        pickle.dump((out, scores), open(os.path.join(CACHE_DIR, "events_logo_dev.pkl"), "wb"))
        for thr in (0.2, 0.3):
            print("PR", pr_table(runs, scores, thr))
        return
    if cmd == "test678":
        test = load_runs(seeds=(6, 7, 8))
        out, scores, _ = logo(runs, test)
        report(out)
        for thr in (0.2, 0.3):
            print("PR", pr_table(test, scores, thr))
        return
    if cmd == "notags":
        # the simulator builds its tags from the truth (possession spans), so check that nothing leans on them:
        # every fitted piece is retrained and tested on tag-stripped matches (A2 dead ball stays as fitted)
        from dataclasses import replace

        def strip(g, sd):
            m, gt, info = T.degraded(g, sd)
            return replace(m, tags=[]), gt, info
        nt = load_runs(tag="notag", degr=strip)
        out, scores, _ = logo(nt, nt, variants_cp=((0.2, 6.0),))
        report(out)
        print("PR", pr_table(nt, scores, 0.3))
        return
    if cmd == "secondary":
        secondary(runs)
        return
    if cmd == "final":
        s1, s2 = fit_final(runs, Ev.EV_MODEL_PATH, Ev.TEAM3_MODEL_PATH)
        print("wrote", s1, s2, sha(Ev.EV_MODEL_PATH), sha(Ev.TEAM3_MODEL_PATH))
        return
    if cmd == "check":
        import tempfile
        hs = []
        for i in range(2):
            a = os.path.join(tempfile.gettempdir(), "ev{}.npz".format(i))
            b = os.path.join(tempfile.gettempdir(), "t3_{}.npz".format(i))
            fit_final(runs, a, b)
            hs.append((sha(a), sha(b)))
        print("identical" if hs[0] == hs[1] else "DIFFERENT", hs)
        return
    if cmd == "det":
        res = {}
        for g in T.GAMES:
            bst = train_ev([runs[k] for k in keys if k[0] != g])
            for k in keys:
                if k[0] == g:
                    res[k] = scores_from(bst, runs[k])
            names = Ev.feature_names()
            gain = bst.feature_importance("gain")
            top = np.argsort(-gain)[:12]
            print(g, "importance", [(names[i], round(float(gain[i] / gain.sum()), 3)) for i in top], flush=True)
        for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
            print("all games", {k: round(v, 3) if isinstance(v, float) else v for k, v in pr_table(runs, res, thr).items()})
        for g in T.GAMES:
            sub = {k: runs[k] for k in keys if k[0] == g}
            print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in pr_table(sub, res, 0.2).items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "det")
