"""Step 5: retrain the three learned layers (A2 dead ball, event detector, event-aware team model) on
Metrica 1-3 + DFL 1-7 and evaluate leave-one-game-out / leave-one-provider-out. EXTERNAL data only.

    $PY scripts/dfl_retrain.py build <game> ...      # per-run feature cache (cache/dfl_feats), v1 prediction
    $PY scripts/dfl_retrain.py fold <held-out game | metrica | dfl> [--ev] [--a2m]   # --a2m: dead-ball model fitted on Metrica only
                                                     # fit on the rest, predict held-out runs -> cache/dfl_retrain
    $PY scripts/dfl_retrain.py report                # LOGO + provider tables
    $PY scripts/dfl_retrain.py final [--check]       # fit on all 10 games, write the v4 model files

The recipe is the committed v3 recipe unchanged (hyper-parameters, features, stacking on out-of-fold
event scores); only the training games change. Strides scale with the number of training runs so the
row count stays that of the committed fit (committed: 9 runs, strides A2 3 / event 2 / team 4).
Training labels: DFL tracking flags (games dfl<k>) or, with --ev, the DFL event rules (games dflev<k>,
same tracking, degraded copies with tags built from the event labels). Metrica labels are always
the Metrica event rules. Deterministic: fixed seeds, 1 thread.
"""
import hashlib
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_learned as T  # noqa: E402
import train_events as TE  # noqa: E402
import train_deadball_a2 as A2  # noqa: E402
from possession import deadball as db, decode, emissions as E, events as Ev, learned as Ln  # noqa: E402
from possession.data import A, CACHE_DIR, D, H  # noqa: E402
from possession.scoring import score  # noqa: E402
from extval import labels  # noqa: E402
from extval.datasets import load_game  # noqa: E402
from extval.harness import degraded  # noqa: E402

METRICA = ["metrica1", "metrica2", "metrica3"]
DFL = ["dfl%d" % k for k in range(1, 8)]
DFLEV = ["dflev%d" % k for k in range(1, 8)]      # same tracking, event-rule labels (tags built from them)
ALL = METRICA + DFL
SEEDS = (0, 1, 2)
FEATS = os.path.join(CACHE_DIR, "dfl_feats2")
OUT = os.path.join(CACHE_DIR, "dfl_retrain2")
OOF_GROUPS = 3


def strides(n_runs):
    return dict(a2=max(1, round(3 * n_runs / 9)), ev=max(1, round(2 * n_runs / 9)), team=max(1, round(4 * n_runs / 9)))


# --------------------------------------------------------------------------------------------
# per-run feature cache


def fpath(game, seed):
    return os.path.join(FEATS, "{}_{}.npz".format(game, seed))


def build(game):
    os.makedirs(FEATS, exist_ok=True)
    g = None
    for s in SEEDS:
        p = fpath(game, s)
        if os.path.isfile(p):
            continue
        t = time.time()
        m, gt, _ = degraded(game, s)
        even, odd = Ln.features_eo(m)
        even, odd = even.astype(np.float32), odd.astype(np.float32)
        e2, own, o2 = Ev.event_features(m, base=(even.astype(np.float64), odd.astype(np.float64)))
        if game.startswith("dfl"):
            g = g or load_game(game)
            gt_ev, gt_fl = labels.derive(g, mode="events"), labels.derive(g, mode="flags")
        else:
            gt_ev = gt_fl = gt
        v1 = decode.hsmm_decode(m, db.p_dead_a(m), E.p_home_b(m))
        np.savez(p, even=even, odd=odd, ev_even=e2.astype(np.float32), own0=own[0].astype(np.float32),
                 own1=own[1].astype(np.float32), ev_odd=o2.astype(np.float32),
                 a2X=db.frame_features2(m).astype(np.float32), gt=gt.astype(np.int8),
                 gt_events=gt_ev.astype(np.int8), gt_flags=gt_fl.astype(np.int8), v1=v1.astype(np.int8))
        print(game, s, "built %.0fs" % (time.time() - t), flush=True)


def load_run(game, seed):
    z = np.load(fpath(game, seed))
    return {k: z[k] for k in z.files}


def ev_feats(d):
    return d["ev_even"].astype(np.float64), [d["own0"].astype(np.float64), d["own1"].astype(np.float64)], \
        d["ev_odd"].astype(np.float64)


# --------------------------------------------------------------------------------------------
# training


def ev_rows(d, stride):
    gt = d["gt"]
    yH, yA, _ = Ev.onset_targets(gt)
    idx = np.flatnonzero(gt != 3)[::stride]
    e, o, dd = ev_feats(d)
    X = np.concatenate([Ev.view(e, o, dd, team, idx) for team in (H, A)]).astype(np.float32)
    return X, np.concatenate([yH[idx], yA[idx]])


def lgb_fit(X, y, params, n_trees):
    import lightgbm as lgb
    return lgb.train(params, lgb.Dataset(X.astype(np.float64), y), num_boost_round=n_trees)


def ev_scores(bst, d, n):
    e, o, dd = ev_feats(d)
    g = np.flatnonzero(d["gt"] != 3)
    out = []
    for team in (H, A):
        z = bst.predict(Ev.view(e, o, dd, team, g), raw_score=True)
        s = np.full(n, np.nan, dtype=np.float32)
        s[g] = (1 / (1 + np.exp(-z))).astype(np.float32)
        out.append(s)
    return out


def team_item(d, match, sH, sA):
    xe, xo = Ev.team_extra(match, sH, sA)
    return dict(even=np.concatenate([d["even"], xe.astype(np.float32)], 1),
                odd=np.concatenate([d["odd"], xo.astype(np.float32)], 1), gt=d["gt"])


def team_rows(it, stride):
    idx = np.flatnonzero(it["gt"] <= 1)[::stride]
    e, o = it["even"][idx].astype(np.float64), it["odd"][idx].astype(np.float64)
    y = (it["gt"][idx] == H).astype(float)
    return np.concatenate([Ln.assemble(e, o, False), Ln.assemble(e, o, True)]).astype(np.float32), np.concatenate([y, 1 - y])


def fit_all(train_games, log=print, a2_only_metrica=False):
    """-> dict(a2, ev, team): the three fitted pieces on `train_games` x seeds 0-2."""
    keys = [(g, s) for g in train_games for s in SEEDS]
    st = strides(len(keys))
    t0 = time.time()
    # pass 1: rows for the A2 and event models
    a2X, a2y, EV = [], [], {}
    for k in keys:
        d = load_run(*k)
        if not (a2_only_metrica and not k[0].startswith("metrica")):
            st_a2 = strides(sum(1 for x in keys if x[0].startswith("metrica")))["a2"] if a2_only_metrica else st["a2"]
            sel = np.flatnonzero(d["gt"] != 3)[::st_a2]
            a2X.append(d["a2X"][sel])
            a2y.append((d["gt"][sel] == D).astype(float))
        EV[k] = ev_rows(d, st["ev"])
    hp = {k: v for k, v in A2.DEFAULTS.items() if k != "stride"}
    a2 = A2.fit_gbdt(np.vstack(a2X).astype(np.float64), np.concatenate(a2y), **hp)
    log("  A2 fitted on %d rows (%.0fs)" % (sum(len(y) for y in a2y), time.time() - t0))
    cat = lambda ks: (np.concatenate([EV[k][0] for k in ks]), np.concatenate([EV[k][1] for k in ks]))  # noqa: E731
    bst_ev = lgb_fit(*cat(keys), TE.LGB_EV, TE.N_TREES_EV)
    log("  event model on %d rows (%.0fs)" % (sum(len(EV[k][1]) for k in keys), time.time() - t0))
    # pass 2: out-of-fold event scores for the team model, grouped by game
    games = sorted(train_games)
    k_groups = min(OOF_GROUPS, len(games))
    grp = {g: i % k_groups for i, g in enumerate(games)}
    oof_models = {i: lgb_fit(*cat([k for k in keys if grp[k[0]] != i]), TE.LGB_EV, TE.N_TREES_EV) for i in range(k_groups)}
    log("  %d out-of-fold event models (%.0fs)" % (k_groups, time.time() - t0))
    del EV
    Xt, yt = [], []
    for k in keys:
        d = load_run(*k)
        m, _, _ = degraded(*k)
        sH, sA = ev_scores(oof_models[grp[k[0]]], d, m.n)
        X, y = team_rows(team_item(d, m, sH, sA), st["team"])
        Xt.append(X)
        yt.append(y)
    bst_team = lgb_fit(np.concatenate(Xt), np.concatenate(yt), T.LGB, T.N_TREES)
    log("  team model on %d rows (%.0fs)" % (sum(len(y) for y in yt), time.time() - t0))
    return dict(a2=a2, ev=bst_ev, team=bst_team)


# --------------------------------------------------------------------------------------------
# prediction (fitted-in-memory models, LightGBM inference; the shipped pipeline uses the numpy export)


def predict_run(models, game, seed):
    d = load_run(game, seed)
    m, _, _ = degraded(game, seed)
    g = m.graded
    p = np.full(m.n, np.nan, dtype=np.float32)
    p[g] = db.sigmoid(db.gbdt_predict(models["a2"], d["a2X"][g].astype(np.float64)))
    pre = np.zeros(m.n, dtype=bool)
    pre[g] = d["a2X"][g][:, db.A2_FEATURE_NAMES.index("pre_kickoff")] > 0.5
    p[pre] = np.maximum(p[pre], 0.98)
    p_dead = db.clean_stretches(p, m)
    sH, sA = ev_scores(models["ev"], d, m.n)
    it = team_item(d, m, sH, sA)
    sel = np.flatnonzero(it["gt"] != 3)
    e, o = it["even"].astype(np.float64), it["odd"].astype(np.float64)
    z1 = models["team"].predict(Ln.assemble(e[sel], o[sel], False), raw_score=True)
    z2 = models["team"].predict(Ln.assemble(e[sel], o[sel], True), raw_score=True)
    ph = np.full(m.n, np.nan, dtype=np.float32)
    ph[sel] = (1 / (1 + np.exp(-0.5 * (z1 - z2)))).astype(np.float32)
    return decode.hsmm_decode(m, p_dead, ph, evidence=0.2)


def fold(held, ev=False, a2m=False):
    """held: a game name, or 'metrica' (train DFL, test Metrica) / 'dfl' (train Metrica, test DFL).
    ev=True swaps every DFL game for its dflev twin (same tracking, event-rule labels)."""
    dfl = DFLEV if ev else DFL
    if held == "metrica":
        train, test = dfl, METRICA
    elif held == "dfl":
        train, test = METRICA, dfl
    else:
        h = held.replace("dfl", "dflev") if ev and held.startswith("dfl") else held
        train, test = [g for g in METRICA + dfl if g != h], [h]
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    models = fit_all(train, a2_only_metrica=a2m)
    res = {}
    for g in test:
        for s in SEEDS:
            res[(g, s)] = predict_run(models, g, s)
    tag = held + ("_ev" if ev else "") + ("_a2m" if a2m else "")
    pickle.dump(res, open(os.path.join(OUT, "fold_{}.pkl".format(tag)), "wb"))
    print("fold", tag, "done %.0fs" % (time.time() - t), flush=True)


# --------------------------------------------------------------------------------------------
# reporting


def sc(gt, pred):
    r = score(gt, pred, bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["n_pred_spells"], r["n_gt_spells"]])


def scored(pred_by_run, games):
    """{game: {'native': row, 'other': row}}: mean over seeds against the labels the game was built
    with ('gt': DFL flags for dfl<k>, event rules for dflev<k> and Metrica) and against the other
    DFL convention ('other'; the same as native for Metrica)."""
    out = {}
    for g in games:
        nat, oth = [], []
        for s in SEEDS:
            z = np.load(fpath(g, s))
            alt = z["gt_events"] if g.startswith("dfl") and not g.startswith("dflev") else                 (z["gt_flags"] if "gt_flags" in z.files else z["gt"])
            nat.append(sc(z["gt"], pred_by_run[(g, s)]))
            oth.append(sc(alt, pred_by_run[(g, s)]))
        out[g] = dict(native=np.mean(nat, 0), other=np.mean(oth, 0))
    return out


def fmt(v):
    return "%.3f / %.3f / %.3f, %3.0f of %3.0f" % tuple(v)


def report(ev=False, a2m=False):
    suf = ("_ev" if ev else "") + ("_a2m" if a2m else "")
    dfl = DFLEV if ev else DFL
    allg = METRICA + dfl
    nat_name, oth_name = ("event-rule", "tracking-flag") if ev else ("tracking-flag", "event-rule")
    v1 = {(g, s): np.load(fpath(g, s))["v1"] for g in allg for s in SEEDS}
    s1 = scored(v1, allg)
    logo = {}
    for g in allg:
        h = g.replace("dflev", "dfl")
        p = os.path.join(OUT, "fold_{}.pkl".format(h + suf))
        if os.path.isfile(p):
            logo.update(pickle.load(open(p, "rb")))
    got = [g for g in allg if (g, 0) in logo]
    s3 = scored(logo, got)
    mean = lambda d, key, gs: np.mean([d[g][key] for g in gs], 0)  # noqa: E731
    print("== LOGO over %d games, DFL trained with %s labels (spell / macro / alive, n_pred of n_true); native = %s for DFL, "
          "Metrica event rules for Metrica" % (len(got), nat_name, nat_name))
    for g in got:
        print("%-9s v1 %s | retrained %s" % (g, fmt(s1[g]["native"]), fmt(s3[g]["native"])))
    for name, gs in (("Metrica", [g for g in METRICA if g in got]), ("DFL", [g for g in dfl if g in got]), ("ALL", got)):
        if gs:
            print("%-9s v1 %s | retrained %s   (%d games)" % ("MEAN " + name, fmt(mean(s1, "native", gs)), fmt(mean(s3, "native", gs)), len(gs)))
    gs = [g for g in dfl if g in got]
    if gs:
        print("-- the same DFL runs scored against %s labels: v1 %s | retrained %s" % (oth_name, fmt(mean(s1, "other", gs)), fmt(mean(s3, "other", gs))))
    for held, test in (("metrica", METRICA), ("dfl", dfl)):
        p = os.path.join(OUT, "fold_{}.pkl".format(held + suf))
        if os.path.isfile(p):
            sp = scored(pickle.load(open(p, "rb")), test)
            print("== provider-out: train on %s (%s labels for DFL), test on %s" % (
                "DFL" if held == "metrica" else "Metrica", nat_name, "Metrica" if held == "metrica" else "DFL"))
            print("   native labels: v1 %s | retrained %s" % (fmt(mean(s1, "native", test)), fmt(mean(sp, "native", test))))
            if held == "dfl":
                print("   %s labels: v1 %s | retrained %s" % (oth_name, fmt(mean(s1, "other", test)), fmt(mean(sp, "other", test))))


# --------------------------------------------------------------------------------------------
# final fit, v4 model files


V4 = {"a2": os.path.join(ROOT, "possession", "deadball_a2_model_v4.npz"),
      "ev": os.path.join(ROOT, "possession", "models", "event_gbm_v4.npz"),
      "team": os.path.join(ROOT, "possession", "models", "team_gbm_ev_v4.npz")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def save_models(models, paths):
    np.savez(paths["a2"], **models["a2"])
    TE.save_model(models["ev"], paths["ev"], Ev.feature_names())
    TE.save_model(models["team"], paths["team"], Ln.EVEN_NAMES + Ev.EXTRA_EVEN + Ln.ODD_NAMES + Ev.EXTRA_ODD)


def final(check=False, ev=False):
    import tempfile
    runs = 2 if check else 1
    shas = []
    for i in range(runs):
        paths = V4 if not check else {k: os.path.join(tempfile.gettempdir(), "v4chk{}_{}.npz".format(i, k)) for k in V4}
        save_models(fit_all(METRICA + (DFLEV if ev else DFL)), paths)
        shas.append({k: sha(p) for k, p in paths.items()})
        print("run", i, {k: (v[:12], os.path.getsize(paths[k])) for k, v in shas[-1].items()}, flush=True)
    if check:
        print("identical" if shas[0] == shas[1] else "DIFFERENT")


if __name__ == "__main__":
    cmd = sys.argv[1]
    ev = "--ev" in sys.argv
    if cmd == "build":
        for g in [a for a in sys.argv[2:] if not a.startswith("--")]:
            build(g)
    elif cmd == "fold":
        fold(sys.argv[2], ev, "--a2m" in sys.argv)
    elif cmd == "report":
        report(ev, "--a2m" in sys.argv)
    elif cmd == "final":
        final("--check" in sys.argv, ev)
