"""Combination LOGO: A2 dead ball (fold models from train_deadball_a2.py logo -> cache/a2_logo_preds.pkl)
x learned team model (fold models trained here) + hsmm_decode(evidence=0.2). EXTERNAL data, seeds 0-2.
Committed model files are not touched; nothing is written except cache/."""
import os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_learned as T
from possession import deadball as db, decode
from possession.data import CACHE_DIR
from possession.scoring import score

preds = pickle.load(open(os.path.join(CACHE_DIR, "a2_logo_preds.pkl"), "rb"))
assert len(preds) == 9, len(preds)
runs = {(g, s): T.degraded(g, s) for g in T.GAMES for s in T.DEV_SEEDS}
its = {k: T.item(k[0], k[1], runs[k][0], runs[k][1]) for k in runs}
res = {"v1": [], "combo": [], "combo_ev01": []}
for g in T.GAMES:
    bst = T.train([its[k] for k in its if k[0] != g])
    for s in T.DEV_SEEDS:
        m, gt, _ = runs[(g, s)]
        it = its[(g, s)]
        ph = T.p_from_booster(bst, m.n, it)
        pd = db.clean_stretches(preds[(g, s)], m)
        def sc(pdead, p, **dp):
            r = score(gt, decode.hsmm_decode(m, pdead, p, **dp))
            return dict(spell=r["spell_f1_mean"], macro=r["macro_f1"], alive=r["alive_dead_f1"], n_pred=r["n_pred_spells"], n_gt=r["n_gt_spells"])
        res["v1"].append((g, sc(it["p_dead"], it["p_b"])))
        res["combo"].append((g, sc(pd, ph, evidence=0.2)))
        res["combo_ev01"].append((g, sc(pd, ph)))
    print("fold", g, "done", flush=True)
for k, rows in res.items():
    print("==", k)
    for g in T.GAMES:
        print("  ", g, T.mean_rows([r for gg, r in rows if gg == g]))
    print("   MEAN", T.mean_rows([r for _, r in rows]))
