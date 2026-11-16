"""Fit the v2 (smooth) ball position error to UNLABELLED BAR-ATH statistics.

Two targets, both from extval/calibration.json:
  dist : quantiles of the ball-to-nearest-tracked-object distance (calibrate.NEAR_Q)
  step : frame-to-frame ball step p50/p90/p99, share > 0.5 m, share > 1.5 m, reversal share
No score of any kind is used.  Loss terms are squared log-ratios (with a floor eps so shares that
are ~0 stay finite); L_dist and L_step are reported separately and as a sum.

    $PY -m extval.fit_ball            # grid search, writes degrade_fit.json (best of L_dist+L_step)
"""

import itertools
import json
import sys
from dataclasses import replace

import numpy as np

from extval import degrade as dg
from extval.calibrate import ball_step_stats, nearest_dist_quantiles
from extval.datasets import load_game

GAMES = ("metrica1", "metrica2", "metrica3")
STEP_KEYS = ("p50", "p90", "p99", "share_gt_0.5", "share_gt_1.5", "reversal_share")
EPS = {"p50": 0.01, "p90": 0.01, "p99": 0.01, "share_gt_0.5": 0.002, "share_gt_1.5": 0.001,
       "reversal_share": 0.001}


def base_matches():
    return [dg.degrade(load_game(g), seed=0, ball_err_sigma=0.0, tags=False)[0] for g in GAMES]


def measure(matches, err_fn):
    """Pooled (mean over games) nearest-distance quantiles and step stats with error err_fn(m)."""
    qs, st = [], []
    for m in matches:
        mm = replace(m, ball=m.ball + err_fn(m)) if err_fn else m
        qs.append(nearest_dist_quantiles(mm))
        st.append(ball_step_stats(mm))
    return np.mean(qs, axis=0), {k: float(np.mean([s[k] for s in st])) for k in STEP_KEYS}


def losses(q, st):
    cal = dg.calibration()["ball"]
    ld = float(np.mean((np.log(q) - np.log(cal["nearest_dist_q"])) ** 2))
    ls = float(np.mean([np.log((st[k] + EPS[k]) / (cal["step"][k] + EPS[k])) ** 2 for k in STEP_KEYS]))
    return ld, ls


def candidates():
    for sc, ss, f, so, ssc in itertools.product((0.0, 0.3, 0.6, 0.9), (30, 90), (0.0, 0.1, 0.2, 0.3),
                                                (5.0, 8.0, 11.0), (40, 80, 160)):
        if f == 0 and (so != 5.0 or ssc != 40):
            continue
        if sc == 0 and ss != 30:
            continue
        yield dict(core_sigma_m=sc, core_scale=ss, out_frac=f, out_sigma_m=so, out_scale=ssc)


def main():
    ms = base_matches()
    rows = []
    q0, st0 = measure(ms, None)
    print("no error: L_dist %.4f L_step %.4f  q %s step %s" % (*losses(q0, st0), np.round(q0, 2).tolist(),
                                                              {k: round(v, 4) for k, v in st0.items()}))
    for c in candidates():
        fn = lambda m, c=c: dg.smooth_error(dg._rng(m, 0, 9), m.n, m.pitch, **c)   # noqa: E731
        q, st = measure(ms, fn)
        ld, ls = losses(q, st)
        rows.append((ld + ls, ld, ls, c, q, st))
        print("%.4f %.4f %.4f %s" % (ld + ls, ld, ls, c), flush=True)
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    by_dist = min(rows, key=lambda r: r[1])
    by_step = min(rows, key=lambda r: r[2])
    for name, r in (("best sum", best), ("best dist", by_dist), ("best step", by_step)):
        print(name, "L_dist %.4f L_step %.4f" % (r[1], r[2]), r[3], "\n   q", np.round(r[4], 2).tolist(),
              "\n   step", {k: round(v, 4) for k, v in r[5].items()})
    out = dict(model="smooth", **best[3], loss_dist=round(best[1], 5), loss_step=round(best[2], 5),
               loss_dist_no_error=round(losses(q0, st0)[0], 5), loss_step_no_error=round(losses(q0, st0)[1], 5),
               origin="grid fit to BAR-ATH unlabelled nearest-object-distance quantiles and ball step "
                      "statistics (squared log ratios), Metrica 1-3; see extval/fit_ball.py")
    with open(dg.FIT_PATH, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write(chr(10))
    json.dump([(r[0], r[1], r[2], r[3]) for r in rows], open("fit_ball_all.json", "w"))


if __name__ == "__main__":
    sys.exit(main())
