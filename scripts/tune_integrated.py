"""Joint tuning of A (dead-ball weights), B (SHAPE_W) and the decoder on EXTERNAL games, with
leave-one-game-out (LOGO) for everything fitted. Never touches BAR-ATH labels.

    $PY scripts/tune_integrated.py fit                 # LOGO table for A weights and B SHAPE_W
    $PY scripts/tune_integrated.py eval [--param k=v]  # held-out table for one decoder config
    $PY scripts/tune_integrated.py sweep               # coordinate search, held-out spellF1

Per (game, seed) the expensive parts are cached once: A's feature matrix, B's logit without the
shape term (p_home_b(w_shape=0), exact because the shape logit is additive) and B's shape
features. A fold then only needs a logistic refit, a matrix product and the decoder.
Deterministic: ridge IRLS, no randomness.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import deadball, decode, emissions  # noqa: E402
from possession.data import A, CACHE_DIR, D, H, X as XC  # noqa: E402
from possession.scoring import score  # noqa: E402
from extval.harness import degraded  # noqa: E402
from extval.datasets import DEFAULT_GAMES  # noqa: E402

SEEDS = (0, 1, 2)
KEEP_BIAS = 0          # column 0 (bias) is not penalised
PRE_KICKOFF = 11       # fixed by prior, not fitted (extval halves begin at the kickoff)


def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build(game, seed):
    """Everything fold-independent for one degraded game/seed."""
    match, gt, _ = degraded(game, seed)
    g = match.graded
    z0 = _logit(emissions.p_home_b(match, w_shape=0.0))
    return dict(game=game, seed=seed, match=match, gt=gt, graded=g,
                Xa=deadball.frame_features(match), z0=z0,
                F=emissions.shape_features(match, emissions.PARAMS["shape_win_s"]))


def load_items(games, seeds=SEEDS):
    key = "{}_{}".format("-".join(games), "".join(map(str, seeds)))
    srcs = [os.path.getmtime(os.path.join(ROOT, "possession", f)) for f in ("deadball.py", "emissions.py")]
    path = os.path.join(CACHE_DIR, "tune_items_{}.pkl".format(key))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob["srcs"] == srcs and blob["games"] == list(games):
            return blob["items"]
    items = [build(g, s) for g in games for s in seeds]
    with open(path, "wb") as fh:
        pickle.dump(dict(srcs=srcs, games=list(games), items=items), fh, protocol=4)
    return items


# ---------------------------------------------------------------- fits (ridge logistic, IRLS)


def irls(Xm, y, off, lam, pen_mask, it=30):
    w = np.zeros(Xm.shape[1])
    P = lam * np.diag(pen_mask.astype(float))
    for _ in range(it):
        z = off + Xm @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        grad = Xm.T @ (p - y) + P @ w
        hess = (Xm * (p * (1 - p))[:, None]).T @ Xm + P + 1e-9 * np.eye(len(w))
        w = w - np.linalg.solve(hess, grad)
    return w


def fit_A(items, lam=1.0):
    """Dead-ball logistic weights: D vs alive on graded frames; pre_kickoff column fixed at prior."""
    Xs = np.concatenate([it["Xa"][it["graded"]] for it in items])
    y = np.concatenate([(it["gt"][it["graded"]] == D) for it in items]).astype(float)
    fixed = deadball.DEFAULT_WEIGHTS[PRE_KICKOFF]
    cols = [c for c in range(Xs.shape[1]) if c != PRE_KICKOFF]
    w = np.zeros(Xs.shape[1])
    pen = np.ones(len(cols))
    pen[KEEP_BIAS] = 0
    w[cols] = irls(Xs[:, cols], y, fixed * Xs[:, PRE_KICKOFF], lam, pen)
    w[PRE_KICKOFF] = fixed
    return w


def fit_B(items, lam=1.0):
    """Shape logit weights on alive frames with the chain+near+tag logit as offset (per raw unit)."""
    Fs, ys, offs = [], [], []
    for it in items:
        alive = (it["gt"] == H) | (it["gt"] == A)
        keep = alive & ~np.isnan(it["F"]).any(1)
        Fs.append(it["F"][keep]); ys.append((it["gt"][keep] == H).astype(float)); offs.append(it["z0"][keep])
    F, y, off = np.concatenate(Fs), np.concatenate(ys), np.concatenate(offs)
    sd = F.std(0) + 1e-9
    w = irls(F / sd, y, off, lam, np.ones(F.shape[1]), it=25)
    return w / sd


# ---------------------------------------------------------------- emissions from fitted weights


def emissions_for(it, wa, wb):
    m, g = it["match"], it["graded"]
    p = np.full(m.n, np.nan, dtype=np.float32)
    p[g] = deadball.sigmoid(it["Xa"][g] @ wa)
    p_dead = deadball.clean_stretches(p, m)
    F = it["F"]
    zs = np.nansum(F * wb[None, :], axis=1)
    zs[np.all(np.isnan(F), axis=1)] = 0.0
    p_home = (1 / (1 + np.exp(-(it["z0"] + zs)))).astype(np.float32)
    p_home[~g] = np.nan
    return p_dead, p_home


def logo_weights(items, fit_fn_a=fit_A, fit_fn_b=fit_B):
    """{game: (wA, wB)} fitted on the other games only."""
    games = sorted({it["game"] for it in items})
    out = {}
    for g in games:
        tr = [it for it in items if it["game"] != g]
        out[g] = (fit_fn_a(tr), fit_fn_b(tr))
    return out


# ---------------------------------------------------------------- scoring


def run_items(items, weights_for, dparams, per_item=False):
    """weights_for(game) -> (wA, wB). Returns per-game mean metrics and the overall mean."""
    rows = []
    for it in items:
        wa, wb = weights_for(it["game"])
        pdead, phome = emissions_for(it, wa, wb)
        pred = decode.hsmm_decode(it["match"], pdead, phome, **dparams)
        r = score(it["gt"], pred, bootstrap_n=0)
        rows.append((it["game"], r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"],
                     r["n_pred_spells"], r["n_gt_spells"]))
    games = sorted({r[0] for r in rows})
    per = {g: np.mean([r[1:] for r in rows if r[0] == g], axis=0) for g in games}
    mean = np.mean([per[g] for g in games], axis=0)
    return per, mean


COLS = "spellF1 macroF1 aliveF1 n_pred n_true".split()


def fmt(per, mean, label=""):
    lines = ["{:22s}".format(label) + " ".join("{:>8s}".format(c) for c in COLS)]
    for g, v in per.items():
        lines.append("  {:20s}".format(g) + " ".join("{:8.3f}".format(x) for x in v))
    lines.append("  {:20s}".format("MEAN") + " ".join("{:8.3f}".format(x) for x in mean))
    return "\n".join(lines)


# ---------------------------------------------------------------- commands


def cmd_fit(items):
    games = sorted({it["game"] for it in items})
    wa_all, wb_all = fit_A(items), fit_B(items)
    print("A weights, all games :", np.array2string(wa_all, precision=2, separator=", "))
    print("A shipped (default)  :", np.array2string(deadball.DEFAULT_WEIGHTS, precision=2, separator=", "))
    print("B SHAPE_W all games  :", np.array2string(wb_all, precision=5, separator=", "))
    print("B SHAPE_W shipped    :", np.array2string(emissions.SHAPE_W, precision=5, separator=", "))
    folds = logo_weights(items)
    for g, (wa, wb) in folds.items():
        print("LOGO", g, "A", np.array2string(wa, precision=1, separator=", "))
        print("LOGO", g, "B", np.array2string(wb, precision=4, separator=", "))
    dp = {}
    tables = {
        "in-sample (fit on all 3)": lambda g: (wa_all, wb_all),
        "LOGO (fit on other 2)": lambda g: folds[g],
        "shipped defaults": lambda g: (deadball.DEFAULT_WEIGHTS, emissions.SHAPE_W),
        "A LOGO + B shipped": lambda g: (folds[g][0], emissions.SHAPE_W),
        "A shipped + B LOGO": lambda g: (deadball.DEFAULT_WEIGHTS, folds[g][1]),
        "B shape off (SHAPE_W=0)": lambda g: (folds[g][0], np.zeros(5)),
    }
    for label, wf in tables.items():
        per, mean = run_items(items, wf, dp)
        print(fmt(per, mean, label))


def _parse(kvs):
    out = {}
    for t in kvs:
        k, v = t.split("=", 1)
        out[k] = json.loads(v)
    return out


GRID = dict(
    evidence=[0.03, 0.06, 0.10, 0.15, 0.25],
    switch_penalty=[0.0, 1.0, 2.0, 3.0, 5.0],
    alive_median_s=[5.0, 8.0, 12.0],
    alive_sigma=[0.7, 1.0, 1.3],
    dmin_alive_s=[0.6, 1.0, 1.5, 2.5],
    pd0=[0.3, 0.4, 0.5],
    dead_median_s=[6.0, 9.0, 13.0],
    dead_weight=[0.5, 1.0, 2.0],
)


def _key(p):
    return json.dumps({k: p[k] for k in sorted(p)}, sort_keys=True)


def cmd_sweep(items, passes, start):
    """Coordinate search. Every evaluation is appended to CACHE_DIR/tune_sweep_log.jsonl and
    re-used on restart, so a killed run resumes without repeating work."""
    folds = logo_weights(items)
    logp = os.path.join(CACHE_DIR, "tune_sweep_log.jsonl")
    memo = {}
    if os.path.isfile(logp):
        for line in open(logp):
            r = json.loads(line)
            memo[r["key"]] = r
    wf = lambda g: folds[g]
    cur = dict(decode.HSMM_DEFAULTS)
    cur = {k: cur[k] for k in GRID}
    cur.update(start)
    log = []
    n_eval = 0

    def ev(p):
        nonlocal n_eval
        k = _key(p)
        if k in memo:
            return None, np.array(memo[k]["mean"])
        n_eval += 1
        per, mean = run_items(items, wf, p)
        memo[k] = dict(key=k, mean=list(mean), per={g: list(v) for g, v in per.items()})
        with open(logp, "a") as fh:
            fh.write(json.dumps(memo[k]) + "\n")
        return per, mean

    per, best = ev(cur)
    print("start", cur, "-> spell {:.4f} macro {:.4f} npred {:.0f}".format(best[0], best[1], best[3]))
    for ps in range(passes):
        improved = False
        for k, vals in GRID.items():
            scores = {}
            for v in vals:
                if v == cur[k]:
                    scores[v] = best
                    continue
                trial = dict(cur); trial[k] = v
                scores[v] = ev(trial)[1]
            print("  {:15s}".format(k) + "  ".join(
                "{}: {:.3f}/{:.3f}/{:.0f}".format(v, s[0], s[1], s[3]) for v, s in scores.items()))
            # a move must beat the current by > 0.01 spellF1 (inside-noise moves are not taken)
            vbest = max(scores, key=lambda v: (scores[v][0], scores[v][1]))
            if scores[vbest][0] > best[0] + 0.01:
                cur[k], best, improved = vbest, scores[vbest], True
        print("pass", ps, "->", cur, "spell {:.4f} macro {:.4f}  ({} evals)".format(best[0], best[1], n_eval))
        if not improved:
            break
    per, mean = run_items(items, wf, cur)
    print(fmt(per, mean, "FINAL " + json.dumps(cur)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval", "sweep"])
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES))
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    items = load_items(a.games.split(","))
    print("loaded {} items ({:.0f}s)".format(len(items), time.time() - t0))
    if a.cmd == "fit":
        cmd_fit(items)
    elif a.cmd == "eval":
        folds = logo_weights(items)
        per, mean = run_items(items, lambda g: folds[g], _parse(a.param))
        print(fmt(per, mean, "LOGO " + " ".join(a.param)))
    else:
        cmd_sweep(items, a.passes, _parse(a.param))
    print("({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
