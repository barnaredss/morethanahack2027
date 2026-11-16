"""Step 4: do the COMMITTED models (trained on Metrica only) transfer to DFL? integrated_v1 / v2 / v3 on
degraded DFL games (seeds 0-2) and on the clean games, scored against BOTH label types (tracking flags
and event rules). Nothing is fitted here. EXTERNAL data only.

    $PY scripts/dfl_transfer.py run dfl1 dfl2 ...   # compute + cache predictions per game
    $PY scripts/dfl_transfer.py report              # tables
"""
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import deadball, decode, emissions, events, learned  # noqa: E402
from possession.data import CACHE_DIR  # noqa: E402
from possession.scoring import score  # noqa: E402
from extval import labels  # noqa: E402
from extval.datasets import load_game  # noqa: E402
from extval.harness import degraded  # noqa: E402

GAMES = ["dfl%d" % k for k in range(1, 8)]
PIPES = ("v1", "v2", "v3")
OUT = os.path.join(CACHE_DIR, "dfl_transfer2")


def predict(m):
    p_dead_a = deadball.p_dead_a(m)
    p_dead_2 = deadball.p_dead_a2(m)
    base = learned.features_eo(m)
    return {"v1": decode.hsmm_decode(m, p_dead_a, emissions.p_home_b(m)),
            "v2": decode.hsmm_decode(m, p_dead_2, learned.p_home_e(m), evidence=0.2),
            "v3": decode.hsmm_decode(m, p_dead_2, events.p_home_e3(m, base=base), evidence=0.2)}


def run(game):
    os.makedirs(OUT, exist_ok=True)
    g = load_game(game)
    gt_ev = labels.derive(g, mode="events")
    res = {"gt_events": gt_ev}
    for s in (0, 1, 2):
        m, gt_fl, _ = degraded(game, s)
        res[("deg", s)] = predict(m)
        res[("gt_flags", s)] = gt_fl
        if s == 0:
            res[("clean", 0)] = predict(replace(g.match, tags=m.tags))
        print(game, "seed", s, "done", flush=True)
    pickle.dump(res, open(os.path.join(OUT, game + ".pkl"), "wb"))


def sc(gt, pred):
    r = score(gt, pred, bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["n_pred_spells"], r["n_gt_spells"]])


def report():
    tab = {}
    for game in GAMES:
        res = pickle.load(open(os.path.join(OUT, game + ".pkl"), "rb"))
        for lab in ("flags", "events"):
            for cond, seeds in (("deg", (0, 1, 2)), ("clean", (0,))):
                for p in PIPES:
                    rows = [sc(res[("gt_flags", s)] if lab == "flags" else res["gt_events"], res[(cond, s)][p]) for s in seeds]
                    tab[(lab, cond, p, game)] = np.mean(rows, 0)
    for lab in ("flags", "events"):
        for cond in ("deg", "clean"):
            print("\n== labels: %s | tracking: %s   (spellF1 / macroF1 / aliveF1, n_pred of n_true)" % (lab, cond))
            print("%-6s | " % "game" + " | ".join("%-34s" % p for p in PIPES))
            for game in GAMES + ["MEAN"]:
                cells = []
                for p in PIPES:
                    v = np.mean([tab[(lab, cond, p, g)] for g in GAMES], 0) if game == "MEAN" else tab[(lab, cond, p, game)]
                    cells.append("%.3f / %.3f / %.3f, %3.0f of %3.0f" % tuple(v))
                print("%-6s | " % game + " | ".join(cells))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for g in sys.argv[2:]:
            run(g)
    else:
        report()
