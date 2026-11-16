"""Possession-estimator tests: synthetic matches, no data files needed."""

import numpy as np

from possession import emissions as E
from possession.data import A, H, Match, Tag
from possession.synthetic import toy_match


def _match(ball_owner, n_slots=6, fps=25, tags=()):
    """Hand-built match: ball_owner[i] in {0 home, 1 away, -1 missing}. Home slots 0-2 on the left
    third, away slots 3-5 on the right third; the ball sits 0.5 m from the owner's slot 0/3."""
    n = len(ball_owner)
    L, W = 105.0, 68.0
    players = np.zeros((n, n_slots, 2))
    base_x = np.array([0.3, 0.35, 0.4, 0.6, 0.65, 0.7])[:n_slots]
    players[:, :, 0] = base_x[None, :]
    players[:, :, 1] = np.linspace(0.3, 0.7, n_slots)[None, :]
    ball = np.full((n, 2), np.nan)
    for i, o in enumerate(ball_owner):
        if o >= 0:
            slot = 0 if o == 0 else 3
            ball[i] = players[i, slot] + np.array([0.5 / L, 0.0])
    half = np.ones(n, dtype=np.int8)
    return Match(name="t", n=n, fps=fps, pitch=(L, W), ball=ball, players=players,
                 speed=np.full((n, n_slots), 2.0), slot_team=np.array([H] * 3 + [A] * 3, np.int8),
                 half=half, home_attacks_right={1: True, 2: False}, tags=list(tags))


def test_contract_and_range():
    m, _ = toy_match("H" * 120 + "A" * 120)
    p = E.p_home_b(m)
    assert p.shape == (m.n,) and p.dtype == np.float32
    ok = ~np.isnan(p)
    assert ((p[ok] >= 0) & (p[ok] <= 1)).all()
    assert np.isnan(p[~m.graded]).all()
    assert not np.isnan(p[m.graded]).any()


def test_clear_control_gives_right_team():
    m = _match([0] * 100 + [1] * 100)
    p = E.p_home_b(m)
    assert p[10:90].min() > 0.85 and p[110:190].max() < 0.15


def test_same_team_gap_is_bridged():
    m = _match([0] * 100 + [-1] * 75 + [0] * 100)
    p = E.p_home_b(m)
    assert p[100:175].min() > 0.7
    assert len(E.control_events(m)) == 0           # no change of controller


def test_hand_over_inside_gap_and_event():
    m = _match([0] * 100 + [-1] * 50 + [1] * 100)
    p = E.p_home_b(m)
    assert p[50] > 0.8 and p[200] < 0.2
    ev = E.control_events(m)
    assert len(ev) == 1 and int(ev[0, 1]) == A and 100 <= ev[0, 0] <= 150
    assert 0 < ev[0, 2] <= 1


def test_touch_is_not_control():
    # home controls, away player touches the ball for 3 frames only, home continues
    owner = [0] * 100 + [1] * 3 + [0] * 100
    m = _match(owner)
    p = E.p_home_b(m)
    assert p[100:103].min() > 0.5
    assert len(E.control_events(m)) == 0


def test_tag_evidence_signs():
    tags = [Tag("BUILD UP", 0, 50, H), Tag("DEFENDING IN MIDDLE THIRD", 50, 80, H),
            Tag("SET PIECES", 80, 90, A)]
    m = _match([-1] * 100, tags=tags)
    ev = E.tag_evidence(m)
    assert ev[10] == 1 and ev[60] == -1 and ev[85] == 0


def test_shape_features_antisymmetric():
    m = _match([-1] * 200)
    f = E.shape_features(m)
    assert f.shape == (m.n, 5)
    # mirror the pitch and swap the roles: odd features change sign
    m2 = _match([-1] * 200)
    m2.players = m.players.copy()
    m2.players[:, :, 0] = 1 - m.players[:, :, 0]
    m2.slot_team = np.array([A] * 3 + [H] * 3, np.int8)
    f2 = E.shape_features(m2)
    assert np.allclose(f[50:150], -f2[50:150], atol=1e-9, equal_nan=True)


def test_deterministic_and_pipeline_runs():
    from possession import pipelines
    m, _ = toy_match("H" * 100 + "D" * 40 + "A" * 100)
    assert np.array_equal(E.p_home_b(m), E.p_home_b(m), equal_nan=True)
    for name in ("emissions_b_v1", "emissions_b_v1_cp"):
        out = pipelines.get(name)(m)
        assert out.shape == (m.n,) and set(np.unique(out)) <= {0, 1, 2, 3}


# ---- learned team model (possession/learned.py) ----

def _swapped(m):
    from dataclasses import replace
    flip = lambda t: {H: A, A: H}.get(t, t)
    return replace(m, slot_team=np.array([flip(int(t)) for t in m.slot_team], np.int8),
                   home_attacks_right={k: not v for k, v in m.home_attacks_right.items()},
                   tags=[Tag(t.code, t.start_frame, t.end_frame, flip(t.team), t.type, t.half) for t in m.tags])


def test_learned_contract_and_symmetry():
    from possession import learned as L
    m = _match([0] * 100 + [-1] * 40 + [1] * 100,
               tags=[Tag("BUILD UP", 0, 60, H), Tag("PROGRESSION", 150, 240, A)])
    p = L.p_home_e(m)
    assert p.shape == (m.n,) and p.dtype == np.float32 and not np.isnan(p).any()
    assert ((p >= 0) & (p <= 1)).all()
    q = L.p_home_e(_swapped(m))
    assert np.allclose(p + q, 1.0, atol=1e-5)          # no home bias by construction
    assert np.array_equal(p, L.p_home_e(m))             # deterministic


def test_learned_numpy_trees_match_manual_walk():
    from possession import learned as L
    model = L.load_model()
    X = np.random.default_rng(0).normal(size=(50, len(L.EVEN_NAMES) + len(L.ODD_NAMES))) * 3
    fast = L.predict_raw(model, X)
    slow = np.zeros(len(X))
    for i, x in enumerate(X):
        for r in model["root"]:
            n = int(r)
            while model["feat"][n] >= 0:
                n = int(model["left"][n] if x[model["feat"][n]] <= model["thr"][n] else model["right"][n])
            slow[i] += model["value"][n]
    assert np.allclose(fast, slow)
    assert model["root"].shape[0] <= 300 and int(model["depth"]) <= 6


def test_integrated_v2_team_runs_on_toy():
    from possession import pipelines
    m, _ = toy_match("H" * 100 + "D" * 40 + "A" * 100)
    out = pipelines.get("integrated_v2_team")(m)
    assert out.shape == (m.n,) and set(np.unique(out)) <= {0, 1, 2, 3}
