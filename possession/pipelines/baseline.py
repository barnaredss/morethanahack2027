"""Reference pipelines built from the starter baseline. They prove the plumbing end to end.

`baseline_smooth50_exact` reproduces evaluation/results/baseline_smooth50.csv byte for byte
(labels on every frame, ungraded included). `baseline_v0` is the same thing written the way our
own pipelines write: X outside the graded halves, smoothing confined to each half.
"""

import numpy as np

from possession import decode, deadball, emissions
from possession.data import A, D, H, X
from possession.pipelines import register


def _majority(codes, window):
    """starter/baseline.py smooth(): centred majority vote, ties to H, D where nobody seen."""
    if window <= 0:
        return codes.copy()
    hc = np.concatenate(([0], np.cumsum(codes == H)))
    ac = np.concatenate(([0], np.cumsum(codes == A)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def _raw_codes(match):
    p_home = emissions.p_home_nearest(match)
    codes = np.full(match.n, D, dtype=np.int8)
    ok = ~np.isnan(p_home)
    codes[ok] = np.where(p_home[ok] >= 0.5, H, A)
    return codes


@register("baseline_smooth50_exact")
def baseline_smooth50_exact(match, window=50):
    return _majority(_raw_codes(match), window)


@register("baseline_v0")
def baseline_v0(match, window=50):
    p_dead = deadball.p_dead_ball_missing(match)
    p_home = emissions.p_home_nearest(match)
    return decode.majority_decode(match, p_dead, p_home, window=window)
