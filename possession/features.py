"""Shared geometric primitives. Small, stable, used by several modules.

Module-specific features belong in that module (deadball.py, emissions.py, ...),
not here -- this file is frozen after Phase 0 except for bug fixes, so parallel branches do not
collide on it.
"""

import numpy as np

from .data import A, H


def to_metres(match, xy):
    """Normalised [..., 2] -> metres, same shape."""
    out = np.asarray(xy, dtype=np.float64).copy()
    out[..., 0] *= match.pitch[0]
    out[..., 1] *= match.pitch[1]
    return out


def ball_player_distances(match):
    """[n, S] distance in metres from the ball to every slot. NaN where either is missing."""
    ball = to_metres(match, match.ball)[:, None, :]
    players = to_metres(match, match.players)
    return np.sqrt(((players - ball) ** 2).sum(-1))


def nearest_slot(match, dist=None):
    """(slot[n] int, distance[n]) of the closest tracked object to the ball. slot -1 / NaN where
    there is no ball or no player. Ties go to the lowest slot index (matches starter/baseline.py).
    """
    if dist is None:
        dist = ball_player_distances(match)
    ok = ~np.all(np.isnan(dist), axis=1)
    slot = np.full(match.n, -1, dtype=np.int64)
    best = np.full(match.n, np.nan)
    if ok.any():
        filled = np.where(np.isnan(dist[ok]), np.inf, dist[ok])
        slot[ok] = np.argmin(filled, axis=1)
        best[ok] = filled[np.arange(ok.sum()), slot[ok]]
    return slot, best


def nearest_per_team(match, dist=None):
    """(home_dist[n], away_dist[n]): distance from the ball to the closest player of each team."""
    if dist is None:
        dist = ball_player_distances(match)
    out = []
    for team in (H, A):
        sub = dist[:, match.slot_team == team]
        with np.errstate(all="ignore"):
            d = np.where(np.all(np.isnan(sub), axis=1), np.nan,
                         np.nanmin(np.where(np.isnan(sub), np.inf, sub), axis=1))
        out.append(d)
    return out[0], out[1]


def ball_velocity(match, max_gap=1):
    """[n, 2] ball velocity in m/s by central difference over observed neighbours; NaN where the
    ball is missing on either side (no interpolation across gaps longer than max_gap)."""
    b = to_metres(match, match.ball)
    v = np.full_like(b, np.nan)
    v[1:-1] = (b[2:] - b[:-2]) * (match.fps / 2.0)
    return v


def attack_sign(match):
    """[n] +1 where the HOME team attacks towards x=1 on that frame, -1 where towards x=0,
    0 outside the halves. Away's sign is the negation."""
    s = np.zeros(match.n, dtype=np.int8)
    for h in (1, 2):
        s[match.half == h] = 1 if match.home_attacks_right[h] else -1
    return s


def tag_mask(match, code=None, team=None, type_=None):
    """[n] bool: frames covered by any Smart Tag matching the filters (case-insensitive)."""
    m = np.zeros(match.n, dtype=bool)
    for t in match.tags:
        if code is not None and t.code != code.upper():
            continue
        if team is not None and t.team != team:
            continue
        if type_ is not None and t.type != type_.lower():
            continue
        m[max(0, t.start_frame):min(match.n, t.end_frame)] = True
    return m
