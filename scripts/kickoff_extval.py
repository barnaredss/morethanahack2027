"""Kick-off detector check on EXTERNAL games only (Metrica 1-3). Never reads BAR-ATH.

Truth = the first KICK OFF event of each half (extval event logs). Three tests per parameter set:
  real    the real tracking of the whole period (from the first tracked frame of the period), so the
          detector must return the logged kick-off (metrica3 half 1 has 14 s of real pre-kick-off,
          the other halves 0-2 s);
  synth   the same halves with a seeded synthetic walking spell (30/60/120 s, ~0.5 m/s wander around
          the kick-off formation) prepended, i.e. the situation of a long tracked pre-kick-off;
  start   own-half fraction of both teams at the logged kick-off (football fact: all players in own
          half except the kicker), used to set `own_half_min`.
Usage: python scripts/kickoff_extval.py
"""
import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extval.datasets import _speed, load_game  # noqa: E402
from possession.data import A, H  # noqa: E402
from possession.deadball import detect_kickoffs, own_half_fraction  # noqa: E402

SETS = {
    "current": dict(),
    "prior": dict(own_half_min=0.60, min_pre_s=10.0),
}


def kickoffs(game):
    return {h: min(e.start for e in game.events
                   if e.type == "SET PIECE" and "KICK OFF" in e.sub and e.period == h)
            for h in (1, 2)}


def period_match(game, ko):
    """Match whose half blocks start at the first tracked frame of the period (kick-off not cropped)."""
    m = game.match
    first_tracked = np.flatnonzero(~np.isnan(m.players[:, :, 0]).all(1))
    half = m.half.copy()
    for h in (1, 2):
        idx = np.flatnonzero(m.half == h)
        # the period starts where tracking of that half starts: up to 600 s before the kick-off,
        # but never inside the other half or before the first tracked frame
        lo = max(ko[h] - 15000, first_tracked[0], (idx[0] if h == 1 else half.nonzero()[0].min()))
        if h == 2:
            lo = max(lo, int(np.flatnonzero(m.half == 1)[-1]) + 1)
        half[lo:idx[0]] = h
    return dataclasses.replace(m, half=half)


def with_prepend(game, h, pre_s, seed):
    """Match containing only half h with `pre_s` seconds of synthetic walking before its kick-off."""
    m = game.match
    idx = np.flatnonzero(m.half == h)
    a, b = int(idx[0]), int(idx[-1]) + 1
    npre = int(pre_s * m.fps)
    rng = np.random.default_rng(seed)
    L, W = m.pitch
    form = m.players[a]
    t = np.arange(npre)[:, None, None]
    ph = rng.uniform(0, 2 * np.pi, (1,) + form.shape)
    per = rng.uniform(8, 20, (1,) + form.shape) * m.fps          # 8-20 s wander period
    amp = 0.6 / np.array([L, W])                                 # 0.6 m amplitude -> ~0.2-0.5 m/s
    pre = form[None] + amp * np.sin(2 * np.pi * t / per + ph)
    players = np.concatenate([pre, m.players[a:b]])
    ball = np.concatenate([np.full((npre, 2), np.nan), m.ball[a:b]])
    half = np.concatenate([np.full(npre, h, np.int8), m.half[a:b]])
    sm = dataclasses.replace(m, n=len(half), ball=ball, players=players, half=half, tags=[],
                             speed=_speed(players, m.fps, m.pitch))
    return sm, npre


def main():
    for name, p in SETS.items():
        print("== params:", name, p or "(defaults)")
        for g in ("metrica1", "metrica2", "metrica3"):
            G = load_game(g)
            ko = kickoffs(G)
            pm = period_match(G, ko)
            det = detect_kickoffs(pm, **p)
            fps = pm.fps
            real = {h: round((det[h] - ko[h]) / fps, 1) for h in det}
            first = {h: int(np.flatnonzero(pm.half == h)[0]) for h in det}
            synth = {}
            for pre in (30, 60, 120):
                for h in (1, 2):
                    sm, npre = with_prepend(G, h, pre, seed=h * 1000 + pre)
                    d = detect_kickoffs(sm, **p)[h]
                    synth[(pre, h)] = round((d - npre) / fps, 1)
            fr = np.array([ko[1]])
            print(g, "real err s (det-ko):", real, "pre-ko s:", {h: round((ko[h] - first[h]) / fps, 1) for h in det},
                  "| synth err s:", synth)
    print("== own-half fraction at logged kick-off (H, A)")
    for g in ("metrica1", "metrica2", "metrica3"):
        G = load_game(g)
        ko = kickoffs(G)
        for h in (1, 2):
            f = np.array([ko[h]])
            print(g, h, round(own_half_fraction(G.match, H, f)[0], 2), round(own_half_fraction(G.match, A, f)[0], 2))


if __name__ == "__main__":
    main()
