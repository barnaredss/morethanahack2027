"""Tests for extval/. Synthetic games need no data; `raw` tests need extval/raw/."""

import os

import numpy as np
import pytest

from possession.data import A, D, H, X, Match
from extval import datasets, degrade, harness, labels
from extval.datasets import Event, ExtGame

HAVE_RAW = os.path.isdir(datasets.METRICA_DIR)
raw = pytest.mark.skipif(not HAVE_RAW, reason="external raw data not downloaded")


def fake_game(n=7000, seed=1):
    """22 random walkers + a ball that follows the owner; a hand-written event stream."""
    rng = np.random.default_rng(seed)
    S = 22
    pl = np.clip(0.5 + np.cumsum(rng.normal(0, 0.0004, (n, S, 2)), axis=0) * 0.3
                 + rng.uniform(-0.3, 0.3, (1, S, 2)), 0.02, 0.98)
    slot_team = np.array([H] * 11 + [A] * 11, dtype=np.int8)
    ball = pl[:, 0, :].copy()
    half = np.zeros(n, dtype=np.int8)
    half[100:n - 100] = 1
    ev = [Event(H, "SET PIECE", "KICK OFF", 100, 100, 1, "p"),
          Event(H, "PASS", "", 100, 150, 1, "p"), Event(H, "PASS", "", 200, 250, 1, "p"),
          Event(A, "BALL OUT", "", 300, 330, 1, "p"),
          Event(A, "SET PIECE", "THROW IN", 600, 600, 1, "p"), Event(A, "PASS", "", 600, 650, 1, "p"),
          Event(H, "RECOVERY", "", 900, 900, 1, "p"), Event(H, "PASS", "", 905, 960, 1, "p"),
          Event(A, "RECOVERY", "INTERCEPTION", 1000, 1000, 1, "p"),   # 5 frames of control only
          Event(H, "PASS", "", 1005, 1060, 1, "p")]
    m = Match(name="fake", n=n, fps=25, pitch=(105.0, 68.0), ball=ball, players=pl,
              speed=np.zeros((n, S)), slot_team=slot_team, half=half,
              home_attacks_right={1: True, 2: False})
    return ExtGame("fake", "metrica", m, ev)


def test_labels_follow_the_rules():
    gt = labels.derive(fake_game())
    assert (gt[:100] == X).all() and (gt[6900:] == X).all()   # n=7000, half 100..6899
    assert (gt[100:300] == H).all()                       # kick-off owner keeps it
    assert (gt[300:600] == D).all()                       # ball out -> restart is dead
    assert (gt[600:900] == A).all()                       # throw-in team, loose ball stays
    assert (gt[900:1000] == H).all()                      # sustained recovery switches
    assert (gt[1000:1060] == H).all()                     # 5-frame touch is not a change


def test_min_control_changes_the_filter():
    g = fake_game()
    assert (labels.derive(g, min_control=1)[1000:1005] == A).all()


def test_degrade_is_deterministic_and_hits_targets():
    g = fake_game()
    a, gt_a, info = degrade.degrade(g, seed=3, tags=False)
    b, gt_b, _ = degrade.degrade(g, seed=3, tags=False)
    c, _, _ = degrade.degrade(g, seed=4, tags=False)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.players, b.players, equal_nan=True)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)
    assert a.players.shape == (g.match.n, 46, 2) and len(a.slot_team) == 46
    assert (a.slot_team == 0).sum() == 23
    assert a.ball_ok[a.graded].mean() <= 1.0 and np.isfinite(a.speed[~np.isnan(a.speed)]).all()


def test_ball_visibility_matches_target_when_clean_ball_is_always_there():
    g = fake_game(n=120000)           # long enough that the gap process averages out
    m, _, info = degrade.degrade(g, seed=0, tags=False, players=False)
    assert abs(m.ball_ok[m.graded].mean() - 0.593) < 0.03


def test_tags_are_well_formed():
    g = fake_game()
    gt = labels.derive(g)
    tags = degrade.make_tags(g.match, gt, g, np.random.default_rng(0))
    assert tags and all(t.end_frame > t.start_frame >= 0 for t in tags)
    assert {t.code for t in tags} & {"SET PIECES", "BUILD UP", "PROGRESSION", "CREATING CHANCES"}
    assert all(t.team in (H, A, -1) for t in tags)


def test_harness_runs_a_callable_on_a_registered_game(monkeypatch):
    g = fake_game()
    monkeypatch.setitem(datasets._CACHE, "fake", g)
    monkeypatch.setattr(harness, "degraded", lambda name, seed, **kw: degrade.degrade(
        datasets.load_game(name), seed=seed, **kw))

    def always_home(match, **_):
        out = np.full(match.n, X, dtype=np.int8)
        out[match.graded] = H
        return out

    res = harness.evaluate_pipeline(always_home, games=["fake"], seeds=2, bootstrap=20)
    assert set(res["games"]) == {"fake"} and len(res["runs"]) == 2
    assert 0.0 <= res["mean"]["spell_f1"] <= 1.0
    assert res["ci"]["spell_f1"][0] <= res["ci"]["spell_f1"][1]
    assert "MEAN" in harness.format_report("always_home", res)


def test_calibration_json_is_committed_and_sane():
    cal = degrade.calibration()
    assert abs(cal["ball"]["visible_rate"] - 0.593) < 0.002
    assert abs(cal["players"]["objects_mean"] - 20.49) < 0.05
    assert len(cal["ball"]["nearest_dist_q"]) == 6


@raw
def test_metrica_games_load_with_contract():
    for name in datasets.DEFAULT_GAMES:
        m = datasets.load_game(name).match
        assert m.fps == 25 and m.ball.shape == (m.n, 2) and m.players.shape[0] == m.n
        assert {h for h, _, _ in m.half_blocks()} == {1, 2}
        assert set(np.unique(m.slot_team)) == {H, A}
        assert np.nanmax(m.players) < 1.2 and np.nanmin(m.players) > -0.2


@raw
def test_metrica_labels_are_football_plausible():
    for name in datasets.DEFAULT_GAMES:
        g = datasets.load_game(name)
        gt = labels.derive(g)
        n_sp, mean_s, dead, home = labels.spell_stats(gt, g.match)
        assert 150 < n_sp < 450 and 8 < mean_s < 25      # possessions per game and duration
        assert 0.2 < dead < 0.45 and 0.4 < home < 0.6
        assert (gt[~g.match.graded] == X).all()


@raw
def test_real_game_degradation_hits_ballpark_of_atd():
    g = datasets.load_game("metrica3")
    m, gt, info = degrade.degrade(g, seed=0, tags=True)
    assert abs(info["ball"]["visible"] - 0.593) < 0.01
    assert 20.0 < info["players"]["objects_mean"] < 21.5
    assert len(m.tags) > 200


def test_ball_error_is_smooth_frame_to_frame():
    """v1's AR(1) bursts moved ~1 m per frame; the v2 error must be smooth (BAR-ATH balls reverse
    direction in ~0.1% of consecutive step pairs)."""
    fit = degrade.fitted("v2")
    e = degrade.ball_error(np.random.default_rng(0), 60000, (105.0, 68.0), fit) * np.array([105.0, 68.0])
    step = np.diff(e, axis=0)
    assert np.percentile(np.sqrt((step ** 2).sum(1)), 99) < 0.6     # metres per frame (v1: 3.7)
    assert e.std() > 0.5                                    # but it still has real amplitude
    old = degrade.ball_error(np.random.default_rng(0), 60000, (105.0, 68.0), degrade.fitted("v1"))
    assert np.percentile(np.sqrt((np.diff(old * np.array([105.0, 68.0]), axis=0) ** 2).sum(1)), 99) > 2.0


def test_outages_follow_the_calibrated_rate_and_are_deterministic():
    g = fake_game(n=100000)
    a = degrade.make_outages(g.match, np.random.default_rng(5))
    b = degrade.make_outages(g.match, np.random.default_rng(5))
    assert np.array_equal(a, b) and not a[~g.match.graded].any()
    cal = degrade.calibration()["players"]["outage"]
    assert 0.2 * cal["share_of_graded"] < a[g.match.graded].mean() < 4 * cal["share_of_graded"]


def test_duplicates_do_not_jitter():
    g = fake_game(n=20000)
    m, _, _ = degrade.degrade(g, seed=1, tags=False, outages=False)
    pl = m.players * np.array(m.pitch)
    both = ~np.isnan(pl[1:, :, 0]) & ~np.isnan(pl[:-1, :, 0])
    steps = np.sqrt(((pl[1:] - pl[:-1]) ** 2).sum(-1))[both]
    assert (steps > 1.5).mean() < 1e-3                        # v1: duplicates teleported every frame


def test_legacy_flags_restore_v1():
    g = fake_game(n=20000)
    new, _, _ = degrade.degrade(g, seed=1, tags=False)
    old, _, _ = degrade.degrade(g, seed=1, tags=False, **degrade.LEGACY_V1)
    assert not np.array_equal(new.ball, old.ball, equal_nan=True)
