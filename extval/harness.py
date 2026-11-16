"""External-validation harness: the leaderboard every component is selected on.

    $PY -m extval.harness --pipeline baseline_v0 [--games metrica1,metrica2 --seeds 3]
                          [--param window=75] [--bootstrap 1000] [--json out.json]

Python API for parameter sweeps:

    from extval.harness import evaluate_pipeline
    res = evaluate_pipeline("baseline_v0", {"window": 75})      # or a callable fn(match, **params)
    res["mean"]["spell_f1"], res["ci"]["spell_f1"]               # (low, high)

Each (game, seed) pair is one degraded copy of an external game (extval/degrade.py); the pipeline
sees only a `Match`, the truth is the mechanical H/A/D labelling (extval/labels.py), and scoring is
the official metric code (possession.scoring.score). The per-game figure is the mean over seeds, the
headline is the mean over games. Degraded copies are cached in MTAH_CACHE_DIR/extval_deg.

The CI is a bootstrap over possessions (the official spellF1 bootstrap, resampled jointly over all
runs and averaged per draw). For macroF1 it is the mean of the official per-run bounds, which is
approximate. Remember the real noise band: differences below ~0.04 spellF1 are not distinguishable.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

from possession import pipelines
from possession.data import CACHE_DIR
from possession.scoring import pm, score
from extval import degrade as dg
from extval import labels
from extval.datasets import DEFAULT_GAMES, list_games, load_game

DEFAULT_SEEDS = 3
CACHE_VERSION = 1
METRICS = ("spell_f1", "macro_f1", "alive_dead_f1", "n_pred", "n_gt")


def degraded(name, seed, **dparams):
    """(Match, gt, info) for one game/seed; cached on disk because degrading takes seconds."""
    key = hashlib.sha1(json.dumps([name, seed, dparams, CACHE_VERSION,
                                   os.path.getmtime(dg.CAL_PATH), os.path.getmtime(dg.__file__),
                                   os.path.getmtime(dg.FIT_PATH) if os.path.isfile(dg.FIT_PATH) else 0,
                                   os.path.getmtime(labels.__file__)],
                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, "extval_deg", "{}_{}_{}.pkl".format(name, seed, key))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    res = dg.degrade(load_game(name), seed=seed, **dparams)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    return res


def _spell_flags(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gs), t) for t in pm.SPELL_IOUS])
    return flags, len(gs), len(ps)


def _pooled_bootstrap(flag_sets, n_draws, seed=0):
    """Resample possessions inside every run, average spellF1 over runs, per draw."""
    if not n_draws:
        return (None, None)
    rng = np.random.RandomState(seed)
    scores = np.zeros(n_draws)
    for flags, n_gt, n_pred in flag_sets:
        for d in range(n_draws):
            picks = rng.randint(0, n_gt, n_gt)
            f1s = []
            for row in flags:
                tp = int(row[picks].sum())
                fp, fn = max(0, n_pred - tp), n_gt - tp
                f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
            scores[d] += np.mean(f1s)
    scores /= len(flag_sets)
    return tuple(float(x) for x in np.percentile(scores, pm.BOOTSTRAP_PERCENTILES))


def evaluate_pipeline(fn, params=None, games=None, seeds=DEFAULT_SEEDS, bootstrap=200,
                      degrade_params=None):
    """Run a pipeline on every degraded external game.

    fn: registered pipeline name or callable fn(match, **params) -> int8[n] codes.
    seeds: int (seeds 0..k-1) or iterable of seeds.
    Returns {"games": {name: {metric: mean over seeds}}, "mean": {metric: mean over games},
             "ci": {"spell_f1": (lo, hi), "macro_f1": (lo, hi)}, "runs": [per game/seed dicts]}.
    """
    if isinstance(fn, str):
        fn = pipelines.get(fn)
    params = params or {}
    games = list(games or DEFAULT_GAMES)
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    runs, flag_sets, macro_bounds = [], [], []
    for name in games:
        for seed in seeds:
            match, gt, _ = degraded(name, seed, **(degrade_params or {}))
            pred = np.asarray(fn(match, **params), dtype=np.int8)
            if pred.shape != (match.n,):
                raise ValueError("{} returned shape {}".format(name, pred.shape))
            r = score(gt, pred, bootstrap_n=0)
            runs.append({"game": name, "seed": seed, "spell_f1": r["spell_f1_mean"],
                         "macro_f1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
                         "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
                         "mean_spell_s": r["mean_spell_duration_s"],
                         "gt_mean_spell_s": r["gt_mean_spell_duration_s"]})
            if bootstrap:
                flag_sets.append(_spell_flags(gt, pred))
                rb = score(gt, pred, bootstrap_n=bootstrap)
                macro_bounds.append((rb["bootstrap_low"], rb["bootstrap_high"]))
    per_game = {}
    for name in games:
        rows = [r for r in runs if r["game"] == name]
        per_game[name] = {k: float(np.mean([r[k] for r in rows])) for k in
                          METRICS + ("mean_spell_s", "gt_mean_spell_s")}
    mean = {k: float(np.mean([v[k] for v in per_game.values()])) for k in METRICS}
    ci = {}
    if bootstrap:
        ci["spell_f1"] = _pooled_bootstrap(flag_sets, bootstrap)
        ci["macro_f1"] = (float(np.mean([b[0] for b in macro_bounds])),
                          float(np.mean([b[1] for b in macro_bounds])))
    return {"games": per_game, "mean": mean, "ci": ci, "runs": runs}


def format_report(name, res):
    lines = ["pipeline {}".format(name),
             "{:12s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
                 "game", "spellF1", "macroF1", "aliveF1", "n_pred", "n_gt")]
    for g, v in res["games"].items():
        lines.append("{:12s} {:8.4f} {:8.4f} {:8.4f} {:8.0f} {:8.0f}".format(
            g, v["spell_f1"], v["macro_f1"], v["alive_dead_f1"], v["n_pred"], v["n_gt"]))
    m = res["mean"]
    lines.append("{:12s} {:8.4f} {:8.4f} {:8.4f} {:8.0f} {:8.0f}".format(
        "MEAN", m["spell_f1"], m["macro_f1"], m["alive_dead_f1"], m["n_pred"], m["n_gt"]))
    if res["ci"]:
        lines.append("90% CI  spellF1 [{:.4f}, {:.4f}]  macroF1 [{:.4f}, {:.4f}]  (noise band: "
                     "differences < ~0.04 spellF1 are not real)".format(
                         *res["ci"]["spell_f1"], *res["ci"]["macro_f1"]))
    return "\n".join(lines)


def _parse_param(text):
    k, v = text.split("=", 1)
    try:
        return k, json.loads(v)
    except ValueError:
        return k, v


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a pipeline on degraded external games.")
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES), help="comma list, or 'all'")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--param", action="append", default=[], help="key=value (JSON) pipeline param")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(list_games()))
        return 0
    games = list_games() if args.games == "all" else args.games.split(",")
    params = dict(_parse_param(p) for p in args.param)
    name = args.pipeline or pipelines.DEFAULT
    t0 = time.time()
    res = evaluate_pipeline(name, params, games, args.seeds, args.bootstrap)
    print(format_report(name, res))
    print("({:.0f}s)".format(time.time() - t0))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
