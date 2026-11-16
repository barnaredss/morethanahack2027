"""Dead-ball detector tests on a hand-built synthetic half (no match data needed)."""

import numpy as np

from possession import deadball, pipelines
from possession.data import Match

FPS = 25
KICKOFF = 1000          # synthetic ground truth, only used to check what the detector finds
OUT_START, OUT_END = 3000, 3400   # ball leaves over the touchline, restart at the throw-in
SHORT_GAP = (4500, 4520)          # 0.8 s dropout in midfield, play continues
LONG_GAP = (5000, 5300)           # 12 s dropout in midfield, play continues at normal speed


def make_half(seed=0, n=6500, pad=100, kickoff=KICKOFF):
    rng = np.random.default_rng(seed)
    S = 20
    slot_team = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    tot = n + 2 * pad
    t = np.arange(tot)
    players = np.empty((tot, S, 2))
    # pre-kickoff: own halves (home attacks right in half 1 => home own half is x < 0.5)
    base_x = np.where(slot_team == 0, rng.uniform(0.1, 0.45, S), rng.uniform(0.55, 0.9, S))
    base_y = rng.uniform(0.1, 0.9, S)
    players[:, :, 0] = base_x[None, :] + rng.normal(0, 0.004, (tot, S)).cumsum(0) * 0.05
    players[:, :, 1] = base_y[None, :] + rng.normal(0, 0.004, (tot, S)).cumsum(0) * 0.05
    # after the kickoff everybody roams
    play = t >= pad + kickoff
    players[play, :, 0] = rng.uniform(0.05, 0.95, (play.sum(), S)) * 0.02 + 0.5 + 0.2 * np.sin(
        t[play, None] / 200.0 + np.arange(S)[None, :])
    speed = np.full((tot, S), 3.0) + rng.normal(0, 0.3, (tot, S))
    speed[:pad + kickoff] = 0.8                                  # walking before the kickoff
    speed[pad + kickoff:pad + kickoff + 250] = 6.0              # sprint after the kickoff
    speed[pad + OUT_START:pad + OUT_END] = 0.8                   # whistle: everybody slows down
    # ball: mid-pitch while alive, over the line during the out-of-play episode
    ball = np.full((tot, 2), np.nan)
    ball[:, 0] = 0.5 + 0.15 * np.sin(t / 150.0)
    ball[:, 1] = 0.5 + 0.2 * np.cos(t / 220.0)
    ball[:pad + kickoff] = np.nan                               # not tracked during warm-up
    run = np.arange(OUT_START - 60, OUT_START)
    ball[pad + run, 1] = np.linspace(0.8, 1.04, len(run))        # drifts over the touchline
    ball[pad + OUT_START + 15:pad + OUT_END] = np.nan
    ball[pad + SHORT_GAP[0]:pad + SHORT_GAP[1]] = np.nan
    ball[pad + LONG_GAP[0]:pad + LONG_GAP[1]] = np.nan
    half = np.zeros(tot, dtype=np.int8)
    half[pad:pad + n] = 1
    return Match(name="synth", n=tot, fps=FPS, pitch=(105.0, 68.0), ball=ball, players=players,
                 speed=speed, slot_team=slot_team, half=half,
                 home_attacks_right={1: True, 2: False}), pad


def test_contract_shape_dtype_and_nan_outside_half():
    match, pad = make_half()
    p = deadball.p_dead_a(match)
    assert p.shape == (match.n,) and p.dtype == np.float32
    assert np.isnan(p[:pad]).all() and np.isnan(p[-pad:]).all()
    inside = p[pad:-pad]
    assert not np.isnan(inside).any() and inside.min() >= 0.0 and inside.max() <= 1.0


def test_deterministic():
    match, _ = make_half()
    assert np.array_equal(deadball.p_dead_a(match), deadball.p_dead_a(match), equal_nan=True)


def test_kickoff_detected_not_hardcoded():
    match, pad = make_half()
    ko = deadball.detect_kickoffs(match)[1]
    assert abs(ko - (pad + KICKOFF)) < 6 * FPS
    later, pad2 = make_half(kickoff=1700)       # the detector must follow, not return a constant
    assert abs(deadball.detect_kickoffs(later)[1] - (pad2 + 1700)) < 6 * FPS


def test_half_that_starts_with_play_has_no_pre_kickoff():
    match, pad = make_half()
    match.speed[:] = 3.0
    match.speed[pad:pad + 250] = 6.0
    assert deadball.detect_kickoffs(match)[1] == pad


def test_pre_kickoff_is_dead():
    match, pad = make_half()
    p = deadball.p_dead_a(match)
    assert p[pad + 100:pad + KICKOFF - 100].min() > 0.9
    assert p[pad + KICKOFF + 200:pad + KICKOFF + 600].max() < 0.5


def test_ball_over_line_then_long_gap_is_dead_with_sharp_edges():
    match, pad = make_half()
    p = deadball.p_dead_a(match)
    assert p[pad + OUT_START + 20:pad + OUT_END - 20].min() > 0.7
    # onset within 1.5 s of the true crossing, never late by more than that
    dead = np.flatnonzero(p[pad + 2500:pad + 3500] > 0.5) + 2500
    assert abs(dead[0] - (OUT_START - 20)) < 1.5 * FPS
    assert abs(dead[-1] + 1 - OUT_END) < 1.5 * FPS


def test_short_dropout_is_not_dead():
    match, pad = make_half()
    p = deadball.p_dead_a(match)
    assert p[pad + SHORT_GAP[0]:pad + SHORT_GAP[1]].max() < 0.5


def test_midfield_dropout_with_normal_player_speed_is_less_dead_than_whistle():
    match, pad = make_half()
    p = deadball.p_dead_a(match, clean=False)
    mid = p[pad + LONG_GAP[0] + 25:pad + LONG_GAP[1] - 25].mean()
    out = p[pad + OUT_START + 25:pad + OUT_END - 25].mean()
    assert out > mid + 0.3


def test_clean_stretches_closes_holes_and_drops_islands():
    match, pad = make_half()
    p = np.full(match.n, np.nan, dtype=np.float32)
    p[match.graded] = 0.1
    p[pad + 100:pad + 300] = 0.9
    p[pad + 300:pad + 310] = 0.1          # 0.4 s hole -> closed
    p[pad + 310:pad + 500] = 0.9
    p[pad + 900:pad + 910] = 0.9          # 0.4 s island -> removed
    out = deadball.clean_stretches(p, match)
    assert out[pad + 305] > 0.5 and out[pad + 905] < 0.5 and np.isnan(out[0])


def test_pipelines_registered_and_valid():
    match, pad = make_half()
    for name in ("deadball_a_v1", "deadball_a_v1_majority"):
        codes = pipelines.get(name)(match)
        assert codes.shape == (match.n,) and codes.dtype == np.int8
        assert (codes[:pad] == 3).all() and (codes[pad:-pad] != 3).all()
