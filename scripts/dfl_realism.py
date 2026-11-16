"""Realism of the degraded DFL games vs the unlabelled BAR-ATH statistics (extval/calibration.json),
as NOTES section D does for Metrica. EXTERNAL data only. Also baseline_v0 windows vs the PUBLISHED floors.

    $PY scripts/dfl_realism.py [--calcheck]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extval import calibrate as C  # noqa: E402
from extval.harness import degraded, evaluate_pipeline  # noqa: E402
from possession.data import D  # noqa: E402

cal = json.load(open(C.OUT))
B, P = cal["ball"], cal["players"]
print("BAR-ATH (calibration.json): ball vis %.3f, gaps %d, nearest q %s, step p50/p90/p99 %.3f/%.3f/%.3f, share>0.5 %.3f, "
      "objects %.2f, >22 %.3f, outage share %.3f, births/player/min %.2f" % (
          B["visible_rate"], B["n_gaps"], B["nearest_dist_q"], B["step"]["p50"], B["step"]["p90"], B["step"]["p99"],
          B["step"]["share_gt_0.5"], P["objects_mean"], P["frac_over_22"], P["outage"]["share_of_graded"],
          P["births_per_player_min"]))
games = ["dfl%d" % k for k in range(1, 8)]
rows = []
for g in games:
    for s in (0, 1, 2):
        m, gt, info = degraded(g, s)
        gm = m.graded
        dead = (gt == D) & gm
        miss = ~m.ball_ok & gm
        bs = C.ball_stats(m)
        st = C.ball_step_stats(m)
        ps = C.player_stats(m)
        og = C.outage_stats(m)
        rows.append([bs["visible_rate"], bs["n_gaps"], *C.nearest_dist_quantiles(m), st["p50"], st["p90"], st["share_gt_0.5"],
                     ps["objects_mean"], ps["frac_over_22"], og["share_of_graded"], ps["births_per_player_min"],
                     miss[dead].mean(), miss[gm & ~dead].mean(), info["ball"]["native_visible"]])
    r = np.mean(rows[-3:], 0)
    print("%s vis %.3f gaps %.0f near q10/50/90/99 %.2f/%.2f/%.2f/%.2f step p50/p90 %.3f/%.3f >0.5 %.3f objs %.2f >22 %.3f "
          "outage %.3f births %.2f | P(NaN|dead) %.2f P(NaN|alive) %.2f native vis %.3f" % (
              g, r[0], r[1], r[2], r[4], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17]), flush=True)
r = np.mean(rows, 0)
print("MEAN  vis %.3f gaps %.0f near q10/50/90/99 %.2f/%.2f/%.2f/%.2f step p50/p90 %.3f/%.3f >0.5 %.3f objs %.2f >22 %.3f "
      "outage %.3f births %.2f | P(NaN|dead) %.2f P(NaN|alive) %.2f" % (
          r[0], r[1], r[2], r[4], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16]))
if "--calcheck" in sys.argv:
    pub = {0: (0.0486, 0.6532), 25: (0.1996, 0.6752), 50: (0.2433, 0.6724), 125: (0.2242, 0.6356)}
    print("baseline_v0 window | DFL degraded spell/macro | published | n_pred")
    for w, (ps_, pm_) in pub.items():
        r = evaluate_pipeline("baseline_v0", {"window": w}, games=games, seeds=3, bootstrap=0)["mean"]
        print("%4d | %.4f / %.4f | %.4f / %.4f | %.0f" % (w, r["spell_f1"], r["macro_f1"], ps_, pm_, r["n_pred"]), flush=True)
