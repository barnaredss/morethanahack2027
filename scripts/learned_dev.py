"""Dev: cache LOGO held-out probabilities, then sweep decoder params on them (extval seeds 0-2)."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_learned as T
from possession import learned as Ln

runs = {(g, s): T.degraded(g, s) for g in T.GAMES for s in T.DEV_SEEDS}
its = {k: T.item(k[0], k[1], runs[k][0], runs[k][1]) for k in runs}
P = {}
for g in T.GAMES:
    bst = T.train([its[k] for k in its if k[0] != g])
    for s in T.DEV_SEEDS:
        P[(g, s)] = T.p_from_booster(bst, runs[(g, s)][0].n, its[(g, s)])
def ev(dp, cp=False, prob="ho"):
    rows = []
    for k in runs:
        m, it = runs[k][0], its[k]
        p = P[k] if prob == "ho" else it["p_b"]
        rows.append(T.eval_run(m, it, p, **dp))
    return T.mean_rows(rows)
for dp in [dict(), dict(evidence=0.2)]:
    print(dp, ev(dp))
