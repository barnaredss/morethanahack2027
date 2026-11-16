"""Decoder synthetic study: spellF1 of the HSMM decoder vs noise level and parameters.

    python scripts/decode_c_study.py sweep      # one-at-a-time sweeps around the defaults
    python scripts/decode_c_study.py report     # spellF1 vs noise, hsmm vs majority_decode

Uses ONLY synthetic data (possession/decode_synth.py); never touches BAR-ATH labels.
Dev seeds 0-5 pick parameters; seeds 100-105 are held out for the final report.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import decode  # noqa: E402
from possession.decode_synth import make_match  # noqa: E402
from possession.scoring import score  # noqa: E402

DEV, HELD = range(0, 6), range(100, 106)
NOISES = (0.5, 1.0, 1.5)
# generator whose duration statistics differ from the decoder's prior (guards against circularity)
MISMATCH = dict(alive_median_s=5.0, alive_sigma=1.3, short_frac=0.2, dead_median_s=7.0)

_cache = {}


def data(seed, noise, mismatch):
    key = (seed, noise, mismatch)
    if key not in _cache:
        _cache[key] = make_match(seed=seed, noise=noise, **(MISMATCH if mismatch else {}))
    return _cache[key]


def evaluate(params, seeds=DEV, noises=NOISES, mismatch=False, fn=None):
    fn = fn or decode.hsmm_decode
    out = []
    for noise in noises:
        vals = []
        for seed in seeds:
            m, gt, pd_, ph = data(seed, noise, mismatch)
            vals.append(score(gt, fn(m, pd_, ph, **params))["spell_f1_mean"])
        out.append(float(np.mean(vals)))
    return out


def objective(params):
    a = evaluate(params)
    b = evaluate(params, mismatch=True)
    return float(np.mean(a + b))


SWEEPS = dict(
    evidence=[0.1, 0.2, 0.35, 0.5, 0.8, 1.2],
    switch_penalty=[0.0, 1.0, 3.0, 6.0],
    alive_median_s=[4.0, 6.0, 8.0, 12.0],
    alive_sigma=[0.8, 1.1, 1.5],
    dmin_alive_s=[0.3, 0.6, 1.0, 1.5],
    dmin_dead_s=[1.0, 2.0, 4.0],
    refine_window_s=[0.0, 1.0, 2.0, 4.0],
    eps=[0.005, 0.02, 0.1],
    pd0=[0.1, 0.25, 0.4],
)


def sweep():
    print("base objective", round(objective({}), 4))
    for name, vals in SWEEPS.items():
        print(name, [(v, round(objective({name: v}), 4)) for v in vals])


def report(params=None):
    params = params or {}
    print("{:>6} {:>10} {:>10} {:>10} | mismatch: {:>8} {:>8}".format(
        "noise", "hsmm", "majority", "no-refine", "hsmm", "majority"))
    for noise in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        row = []
        for mm in (False, True):
            h = evaluate(params, HELD, (noise,), mm)[0]
            mj = evaluate({}, HELD, (noise,), mm, fn=lambda m, pd_, ph, **k: decode.majority_decode(m, pd_, ph))[0]
            nr = evaluate(dict(params, refine_window_s=0.0), HELD, (noise,), mm)[0]
            row.append((h, mj, nr))
        print("{:>6} {:>10.3f} {:>10.3f} {:>10.3f} | {:>16.3f} {:>8.3f}".format(
            noise, row[0][0], row[0][1], row[0][2], row[1][0], row[1][1]))


if __name__ == "__main__":
    {"sweep": sweep, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
