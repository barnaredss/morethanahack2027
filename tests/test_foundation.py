"""Phase 0 contract tests. Fast ones run anywhere; `data` ones need the match bundle."""

import os

import numpy as np
import pytest

from possession import decode, deadball, emissions, pipelines
from possession.data import A, D, H, X, ROOT
from possession.run import sha256, validate, write_submission
from possession.scoring import score
from possession.synthetic import toy_match

from games import DATA_DIR  # noqa: E402
HAVE_DATA = os.path.isfile(os.path.join(DATA_DIR, "tactical", "BAR-ATH_tactical_atd.txt"))
data = pytest.mark.skipif(not HAVE_DATA, reason="match data not present")


def test_toy_pipeline_is_perfect_on_clean_toy():
    match, gt = toy_match("H" * 200 + "D" * 50 + "A" * 200)
    pred = decode.majority_decode(match, deadball.p_dead_ball_missing(match),
                                  emissions.p_home_nearest(match), window=5)
    assert pred.shape == (match.n,)
    assert (pred[~match.graded] == X).all()
    r = score(gt, pred)
    assert r["spell_f1_mean"] == pytest.approx(1.0)


def test_estimator_contracts_on_toy():
    match, _ = toy_match("H" * 50 + "A" * 50)
    for arr in (deadball.p_dead_ball_missing(match), emissions.p_home_nearest(match)):
        assert arr.shape == (match.n,) and arr.dtype == np.float32
        ok = ~np.isnan(arr)
        assert ((arr[ok] >= 0) & (arr[ok] <= 1)).all()


def test_every_registered_pipeline_runs_on_toy():
    match, _ = toy_match("H" * 60 + "D" * 20 + "A" * 60)
    for name in pipelines.names():
        out = np.asarray(pipelines.get(name)(match))
        assert out.shape == (match.n,), name
        assert set(np.unique(out)) <= {H, A, D, X}, name


def test_writer_is_byte_deterministic(tmp_path):
    codes = np.array([X, H, A, D] * 5, dtype=np.int8)
    a = write_submission(str(tmp_path / "a.csv"), codes)
    b = write_submission(str(tmp_path / "b.csv"), codes)
    assert sha256(a) == sha256(b)
    with open(a, "rb") as fh:
        assert b"\r" not in fh.read()
    validate(a, len(codes))


@data
def test_atd_matches_documented_stats():
    from possession.data import load_match
    m = load_match()
    assert m.n == 199319 and m.players.shape == (199319, 46, 2)
    assert abs(m.ball_ok.mean() - 0.569) < 0.001          # DATA.md: 56.9% overall
    assert abs(m.ball_ok[m.graded].mean() - 0.593) < 0.001  # 59.3% graded
    assert m.graded.sum() == 158857
    assert m.half_blocks() == [(1, 11185, 90651), (2, 110808, 190197)]
    assert len(m.tags) == 638


@data
def test_reproduces_published_baseline(tmp_path):
    from possession.data import load_match
    m = load_match(with_tags=False)
    path = write_submission(str(tmp_path / "b50.csv"),
                            pipelines.get("baseline_smooth50_exact")(m))
    with open(path, "rb") as fh:
        ours = fh.read()
    with open(os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"), "rb") as fh:
        theirs = fh.read().replace(b"\r\n", b"\n")
    assert ours == theirs
