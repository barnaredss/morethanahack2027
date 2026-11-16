"""Tiny synthetic matches for unit tests. No data files needed.

`toy_match(labels)` builds a Match whose ball sits on a home player during H frames, on an away
player during A frames, and is missing during D frames. Good enough to test plumbing and decoders;
NOT a realistic simulator (that is extval/degrade.py).
"""

import numpy as np

from .data import A, D, H, Match


def toy_match(labels, n_slots=4, fps=25, pad=10, seed=0):
    """labels: sequence of 'H'/'A'/'D' chars for one graded half. `pad` ungraded frames on each
    side. Returns (match, gt_codes) with gt X on the padding."""
    rng = np.random.default_rng(seed)
    lab = np.array([{"H": H, "A": A, "D": D}[c] for c in labels], dtype=np.int8)
    n = len(lab) + 2 * pad
    gt = np.full(n, 3, dtype=np.int8)
    gt[pad:pad + len(lab)] = lab

    slot_team = np.array([H] * (n_slots // 2) + [A] * (n_slots - n_slots // 2), dtype=np.int8)
    players = np.empty((n, n_slots, 2))
    players[:, :, 0] = np.linspace(0.2, 0.8, n_slots)[None, :] + rng.normal(0, 0.005, (n, n_slots))
    players[:, :, 1] = 0.5 + rng.normal(0, 0.005, (n, n_slots))
    speed = np.abs(rng.normal(2.0, 0.5, (n, n_slots)))

    ball = np.full((n, 2), np.nan)
    home_slot, away_slot = 0, n_slots - 1
    for i in range(n):
        if gt[i] == H:
            ball[i] = players[i, home_slot] + 0.001
        elif gt[i] == A:
            ball[i] = players[i, away_slot] + 0.001

    half = np.zeros(n, dtype=np.int8)
    half[pad:pad + len(lab)] = 1
    match = Match(name="toy", n=n, fps=fps, pitch=(105.0, 68.0), ball=ball, players=players,
                  speed=speed, slot_team=slot_team, half=half,
                  home_attacks_right={1: True, 2: False})
    return match, gt
