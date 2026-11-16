"""DFL / IDSSE support (extval/dfl.py, labels.derive_dfl_*). Synthetic: needs no downloaded data."""

import numpy as np

from possession.data import A, D, H, X, Match
from extval import labels
from extval.datasets import Event, ExtGame, load_game


def fake_dfl(n=400):
    half = np.zeros(n, dtype=np.int8)
    half[20:380] = 1
    status = np.ones(n, dtype=np.int8)
    status[100:140] = 0                        # dead stretch
    poss = np.ones(n, dtype=np.int8)           # 1 home
    poss[200:300] = 2                          # away
    poss[100:140] = 2                          # the tracking names an owner while the ball is dead
    m = Match(name="dfl0", n=n, fps=25, pitch=(105.0, 68.0), ball=np.full((n, 2), 0.5),
              players=np.full((n, 4, 2), 0.5), speed=np.zeros((n, 4)),
              slot_team=np.array([H, H, A, A], dtype=np.int8), half=half, home_attacks_right={1: True, 2: False})
    g = ExtGame("dfl0", "dfl", m, [])
    g.flags = dict(status=status, poss=poss, ball_z=np.zeros(n))
    return g


def test_flag_labels_follow_status_then_possession():
    g = fake_dfl()
    gt = labels.derive(g)
    assert (gt[:20] == X).all() and (gt[380:] == X).all()
    assert (gt[100:140] == D).all()                      # dead wins over the named owner
    assert (gt[20:100] == H).all() and (gt[140:200] == H).all()
    assert (gt[200:300] == A).all() and (gt[300:380] == H).all()
    n_sp, mean_s, dead, _ = labels.spell_stats(gt, g.match)
    assert n_sp == 4 and abs(dead - 40 / 360) < 1e-9


def test_event_labels_available_for_dfl_games():
    g = fake_dfl()
    g.events = [Event(H, "SET PIECE", "KICK OFF", 20, 20, 1, ""), Event(H, "PASS", "", 60, 60, 1, ""),
                Event(A, "PASS", "", 210, 210, 1, "")]
    gt = labels.derive(g, mode="events")
    assert gt.shape == (g.match.n,) and (gt[:20] == X).all()
    assert set(np.unique(gt[20:380])) <= {H, A, D}


def test_dfl_games_listed_only_when_downloaded(monkeypatch):
    import extval.datasets as ds
    monkeypatch.setattr(ds, "RAW", "/nonexistent")
    assert not any(n.startswith("dfl") for n in ds.list_games())


def test_integrated_v4_pipelines_registered_and_valid():
    import os
    from possession import pipelines
    from possession.pipelines import integrated_v4 as v4
    from possession.synthetic import toy_match
    assert {"integrated_v4", "integrated_v4_a2m"} <= set(pipelines.names())
    if not all(os.path.isfile(p) for p in (v4.A2_V4, v4.EV_V4, v4.TEAM_V4)):
        import pytest
        pytest.skip("v4 model files not present")
    match, gt = toy_match("H" * 300 + "D" * 120 + "A" * 300)
    for name in ("integrated_v4", "integrated_v4_a2m"):
        out = pipelines.get(name)(match)
        assert out.shape == (match.n,) and set(np.unique(out)) <= {0, 1, 2, 3}
        assert (out[~match.graded] == 3).all()
        assert (out == pipelines.get(name)(match)).all()
