"""Where do missing-ball errors come from? Extval, oracle dead, seeds 0-2."""
import os, sys
import numpy as np, json
DP = json.loads(os.environ.get('DEGRADE_PARAMS', '{}'))   # {} = current simulator; LEGACY_V1 dict = old
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from possession import decode, emissions as E
from possession.data import A, D, H
from possession.scoring import score
from extval.harness import degraded
from extval.datasets import DEFAULT_GAMES
from gaps_measure import runs, handovers

tot = dict(err_ho=0, n_ho=0, err_noho=0, n_noho=0, vis_err=0, vis_n=0)
ub = {"truth_vis_only": [], "truth_everywhere": [], "base": [], "truth_miss_noho": []}
for g in DEFAULT_GAMES:
    for s in (0, 1, 2):
        m, gt, _ = degraded(g, s, **DP)
        ph = E.p_home_b(m); pd_ = (gt == D).astype(np.float32)
        alive = gt <= 1; truth = gt == H; wrong = (ph >= .5) != truth
        miss = ~m.ball_ok
        # alive-missing stretches (maximal runs of alive & missing)
        ho = handovers(gt)
        isho = np.zeros(m.n, bool); isho[ho] = True
        ph_nh = ph.copy()
        for a, b in runs(alive & miss):
            has = isho[a:b + 1].any() or (gt[a] != gt[min(b, m.n - 1)] and gt[b] <= 1)
            w = wrong[a:b].sum()
            if has: tot["err_ho"] += w; tot["n_ho"] += b - a
            else:
                tot["err_noho"] += w; tot["n_noho"] += b - a
                sg = np.arange(a, b); ph_nh[sg] = np.where(truth[sg], .9, .1)
        tot["vis_err"] += wrong[alive & ~miss].sum(); tot["vis_n"] += (alive & ~miss).sum()
        ub["base"].append(score(gt, decode.hsmm_decode(m, pd_, ph))["spell_f1_mean"])
        ub["truth_miss_noho"].append(score(gt, decode.hsmm_decode(m, pd_, ph_nh))["spell_f1_mean"])
        p1 = ph.copy(); sel = alive & ~miss; p1[sel] = np.where(truth[sel], .9, .1)
        ub["truth_vis_only"].append(score(gt, decode.hsmm_decode(m, pd_, p1))["spell_f1_mean"])
        p2 = ph.copy(); p2[alive] = np.where(truth[alive], .9, .1)
        ub["truth_everywhere"].append(score(gt, decode.hsmm_decode(m, pd_, p2))["spell_f1_mean"])
print({k: int(v) for k, v in tot.items()})
print("err rate: missing stretches WITH hand-over %.3f (n=%d) | WITHOUT %.3f (n=%d) | visible %.3f" % (
    tot["err_ho"] / max(1, tot["n_ho"]), tot["n_ho"], tot["err_noho"] / tot["n_noho"], tot["n_noho"], tot["vis_err"] / tot["vis_n"]))
print({k: round(float(np.mean(v)), 3) for k, v in ub.items()})
