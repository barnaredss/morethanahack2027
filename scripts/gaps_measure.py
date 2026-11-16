"""Gaps study: error breakdown + upper bound on EXTERNAL degraded games (oracle dead-ball)."""
import os, sys
import numpy as np, json
DP = json.loads(os.environ.get('DEGRADE_PARAMS', '{}'))   # {} = current simulator; LEGACY_V1 dict = old
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import decode, emissions as E
from possession.data import A, D, H
from possession.scoring import score
from extval.harness import degraded
from extval.datasets import DEFAULT_GAMES

def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def handovers(gt):
    """frames t where gt[t] alive team != gt[t-1] alive team (adjacent, no D between)."""
    a = (gt[1:] <= 1) & (gt[:-1] <= 1) & (gt[1:] != gt[:-1])
    return np.flatnonzero(a) + 1

def run(games=DEFAULT_GAMES, seeds=(0, 1, 2)):
    rows = []
    stats = []
    for g in games:
        for s in seeds:
            m, gt, _ = degraded(g, s, **DP)
            ph = E.p_home_b(m)
            pd_ = (gt == D).astype(np.float32)
            base = score(gt, decode.hsmm_decode(m, pd_, ph))
            ho = handovers(gt)
            # gap membership: maximal ball-missing run containing/adjacent the hand-over frame
            miss = ~m.ball_ok
            gaps = runs(miss)
            gap_of = np.full(m.n, -1)
            for i, (a, b) in enumerate(gaps): gap_of[a:b] = i
            lens = []
            ingap = 0
            for t in ho:
                # hand-over "in gap" if ball missing at t or the nearest observed frames both sides are > 0 away in a gap
                i = gap_of[t]
                if i >= 0:
                    ingap += 1; lens.append(gaps[i][1] - gaps[i][0])
            # upper bound A: truth inside every ball-missing alive frame (gap-perfect), conf .9
            alive = gt <= 1
            ph2 = ph.copy()
            sel = miss & alive
            ph2[sel] = np.where(gt[sel] == H, 0.9, 0.1)
            ub_all = score(gt, decode.hsmm_decode(m, pd_, ph2))
            # upper bound B: truth only in gaps that contain a true hand-over
            ph3 = ph.copy()
            for t in ho:
                i = gap_of[t]
                if i >= 0:
                    a, b = gaps[i]; a0, b0 = max(0, a - 1), b
                    seg = np.arange(a0, b0); sg = seg[alive[seg]]
                    ph3[sg] = np.where(gt[sg] == H, 0.9, 0.1)
            ub_ho = score(gt, decode.hsmm_decode(m, pd_, ph3))
            # accuracy on alive frames, visible vs missing
            truth = gt == H
            acc = lambda msk: float(((ph >= .5) == truth)[msk].mean())
            rows.append(dict(g=g, s=s, base=base["spell_f1_mean"], ub_all=ub_all["spell_f1_mean"],
                             ub_ho=ub_ho["spell_f1_mean"], nho=len(ho), ingap=ingap,
                             medlen=float(np.median(lens)) if lens else 0.0,
                             n_pred=base["n_pred_spells"], n_gt=base["n_gt_spells"],
                             acc_vis=acc(alive & m.ball_ok), acc_mis=acc(alive & miss),
                             miss_share=float((alive & miss).sum() / alive.sum())))
            stats.append(lens)
    return rows, np.concatenate([np.array(x, float) for x in stats])

if __name__ == "__main__":
    rows, lens = run()
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    mean = lambda k: np.mean([r[k] for r in rows])
    print("MEAN base %.3f  UB(all gaps truth) %.3f  UB(only gaps w/ hand-over) %.3f" % (mean("base"), mean("ub_all"), mean("ub_ho")))
    print("hand-overs %.0f, in ball-missing %.0f (%.0f%%); acc vis %.3f mis %.3f; missing share of alive %.2f"
          % (mean("nho"), mean("ingap"), 100 * mean("ingap") / mean("nho"), mean("acc_vis"), mean("acc_mis"), mean("miss_share")))
    print("gap length (frames) of hand-over gaps: pct 10/25/50/75/90", np.percentile(lens, [10, 25, 50, 75, 90]).round(0), "<=75f %.2f" % (lens <= 75).mean())
