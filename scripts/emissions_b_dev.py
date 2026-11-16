"""Possession-estimator dev harness on EXTERNAL games only (never BAR-ATH GT).

    python scripts/emissions_b_dev.py [--param k=v ...]

Prints team accuracy on alive frames split by ball visible / missing, and spellF1 using the HSMM
with an ORACLE dead-ball signal (isolates the possession estimator from the dead-ball model)."""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import decode, emissions
from possession.data import A, D, H
from possession.scoring import score
from extval.harness import degraded
from extval.datasets import DEFAULT_GAMES


def evaluate(params=None, games=DEFAULT_GAMES, seeds=(0, 1, 2), fn=None, decode_params=None, cp=False):
    fn = fn or emissions.p_home_b
    rows = []
    for g in games:
        for s in seeds:
            m, gt, _ = degraded(g, s)
            ph = fn(m, **(params or {}))
            alive = (gt == H) | (gt == A)
            vis, mis = alive & m.ball_ok, alive & ~m.ball_ok
            truth = (gt == H)
            ok = ~np.isnan(ph)
            acc = lambda msk: float((((ph >= 0.5) == truth)[msk & ok]).mean())
            pd_ = np.where(gt == D, 1.0, 0.0).astype(np.float32)
            cps = emissions.control_events(m, **(params or {})) if cp else None
            pred = decode.hsmm_decode(m, pd_, ph, change_points=cps, **(decode_params or {}))
            r = score(gt, pred)
            rows.append((g, s, acc(vis), acc(mis), r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    a = np.array([r[2:] for r in rows], dtype=float).mean(0)
    return rows, dict(acc_vis=a[0], acc_mis=a[1], spell=a[2], macro=a[3], n_pred=a[4], n_gt=a[5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--cp", action="store_true")
    ap.add_argument("--nearest", action="store_true")
    a = ap.parse_args()
    params = {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.param)}
    rows, mean = evaluate(params, cp=a.cp, fn=emissions.p_home_nearest if a.nearest else None)
    for r in rows:
        print("%s s%d vis %.3f mis %.3f spell %.3f macro %.3f np %d ng %d" % r)
    print({k: round(v, 3) for k, v in mean.items()})
