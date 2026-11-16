"""Final report for the dead-ball v2 work: integrated_v1 vs integrated_v2_dead on
  fresh   Metrica 1-3 x simulator v2 seeds 3-5 (not used for any development decision)
  clean   Metrica 1-3 with NO degradation (native ball dropouts only, no tags)
  sc      SkillCorner games, seed 0 (tags only; natural broadcast dropouts)
plus onset/offset error of the dead stretches. Usage: report.py fresh|clean|sc
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from extval.datasets import DEFAULT_GAMES, list_games, load_game
from extval.harness import degraded
from extval.labels import derive
from possession import decode, deadball, emissions, pipelines
from possession.data import D
from possession.scoring import score

PIPES = ("integrated_v1", "integrated_v2_dead")


def edges(z):
    d = np.diff(np.r_[0, z.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def edge_err(match, gt, p):
    on, off = [], []
    for _, f, l in match.half_blocks():
        pp, yy = np.nan_to_num(p[f:l + 1]) > 0.5, (gt[f:l + 1] == D)
        gs, ge = edges(yy)
        ps, pe = edges(pp)
        on += [ps[np.argmin(abs(ps - a))] - a for a in gs if len(ps)]
        off += [pe[np.argmin(abs(pe - b))] - b for b in ge if len(pe)]
    return np.array(on), np.array(off)


def one(match, gt):
    out = {}
    ph = emissions.p_home_b(match)
    for name, fn in (("integrated_v1", deadball.p_dead_a), ("integrated_v2_dead", deadball.p_dead_a2)):
        p = fn(match)
        sc = score(gt, decode.hsmm_decode(match, p, ph), bootstrap_n=0)
        on, off = edge_err(match, gt, p)
        g = match.graded
        y = gt == D
        f1 = 2 * ((p[g] > .5) & y[g]).sum() / ((p[g] > .5).sum() + y[g].sum())
        out[name] = [sc["spell_f1_mean"], sc["macro_f1"], sc["alive_dead_f1"], f1,
                     np.median(abs(on)), np.median(on), np.median(abs(off)), sc["n_pred_spells"]]
    return out


def main():
    mode = sys.argv[1]
    runs = []
    if mode == "fresh":
        for g in DEFAULT_GAMES:
            for s in (3, 4, 5):
                m, gt, _ = degraded(g, s)
                runs.append(one(m, gt))
    elif mode == "clean":
        for g in DEFAULT_GAMES:
            ext = load_game(g)
            runs.append(one(ext.match, derive(ext)))
    else:
        for g in [x for x in list_games() if x.startswith("sc")][:3]:
            m, gt, _ = degraded(g, 0)
            runs.append(one(m, gt))
    print(mode, len(runs), "runs   spell macro aliveF1 frameDeadF1 |onset| onset_med |offset| n_pred")
    for name in PIPES:
        a = np.mean([r[name] for r in runs], 0)
        print("  %-20s %.4f %.4f %.4f %.4f %5.1f %5.1f %5.1f %6.1f" % ((name,) + tuple(a)))
    d = np.array([[r[PIPES[1]][i] - r[PIPES[0]][i] for i in range(3)] for r in runs])
    print("  per-run delta spell:", np.round(d[:, 0], 3).tolist())


if __name__ == "__main__":
    main()
