"""Simulator-artefact check: learned model (trained on degraded Metrica) on CLEAN Metrica (LOGO) and on SkillCorner."""
import os, sys
import numpy as np
from dataclasses import replace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_learned as T
from extval.datasets import load_game, sc_match_ids
from extval.labels import derive

EV = float(os.environ.get("EVIDENCE", 0.2))
runs = {(g, s): T.degraded(g, s) for g in T.GAMES for s in T.DEV_SEEDS}
its = {k: T.item(k[0], k[1], runs[k][0], runs[k][1]) for k in runs}
print("== clean Metrica (LOGO model, tags from degraded seed 0), evidence", EV)
v1, v2, a1, a2 = [], [], [], []
for g in T.GAMES:
    bst = T.train([its[k] for k in its if k[0] != g])
    game = load_game(g)
    deg, gt, _ = T.degraded(g, 0)
    clean = replace(game.match, tags=deg.tags)
    it = T.item(g, 0, clean, gt, tag="clean")
    p2 = T.p_from_booster(bst, clean.n, it)
    v1.append(T.eval_run(clean, it, it["p_b"])); v2.append(T.eval_run(clean, it, p2, evidence=EV))
    a1.append(T.team_acc(it, it["p_b"])); a2.append(T.team_acc(it, p2))
    print(g, "v1", round(v1[-1]["spell"], 4), "learned", round(v2[-1]["spell"], 4))
print("clean MEAN v1", T.mean_rows(v1)); print("clean MEAN learned", T.mean_rows(v2))
print("team acc vis/missing v1", np.round(np.mean(a1, 0), 3), "learned", np.round(np.mean(a2, 0), 3))
print("== SkillCorner (model on all 3 degraded Metrica), seed 0")
bst = T.train(list(its.values()))
ids = sc_match_ids()[:int(os.environ.get("NSC", 4))]
v1, v2, a1, a2 = [], [], [], []
for i in ids:
    name = "sc%s" % i
    m, gt, _ = T.degraded(name, 0)
    it = T.item(name, 0, m, gt)
    p2 = T.p_from_booster(bst, m.n, it)
    v1.append(T.eval_run(m, it, it["p_b"])); v2.append(T.eval_run(m, it, p2, evidence=EV))
    a1.append(T.team_acc(it, it["p_b"])); a2.append(T.team_acc(it, p2))
    print(name, "v1", round(v1[-1]["spell"], 4), "learned", round(v2[-1]["spell"], 4))
print("SC MEAN v1", T.mean_rows(v1)); print("SC MEAN learned", T.mean_rows(v2))
print("team acc vis/missing v1", np.round(np.mean(a1, 0), 3), "learned", np.round(np.mean(a2, 0), 3))
