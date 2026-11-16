"""Load one match into plain numpy arrays: the single input every estimator in this package reads.

`Match` is deliberately source-agnostic. BAR-ATH is loaded by `load_match()`; external validation
games (extval/) build the same object from their own files. Every pipeline takes a `Match` and
nothing else, so the same code runs on the competition game and on degraded external games.

Conventions (shared contract, do not change without updating every module):
  - index i is VIDEO frame i (= ATD frame_id - 1 = submission row i). Arrays cover ALL frames.
  - coordinates are normalised 0-1 like the ATD; `pitch` gives metres. x along the length.
  - NaN means "not observed". Never fill NaN in this module.
  - slot_team: 0 = home (H), 1 = away (A). Label codes follow evaluation/possession_metrics:
    H=0, A=1, D=2, X=3.
  - half[i] is 1 or 2 inside the published half blocks, 0 elsewhere. graded = half > 0.
  - home_attacks_right[h] is True when the home team attacks towards x=1 in half h.
"""

import glob
import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames  # noqa: E402

CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
CACHE_VERSION = 1

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])


@dataclass(frozen=True)
class Tag:
    """One Smart Tagging instance on the video-frame index. end_frame is exclusive."""
    code: str              # upper case, e.g. "SET PIECES"
    start_frame: int
    end_frame: int
    team: int              # H, A, or -1 for N/A / missing
    type: str = ""         # set-piece type, lower case ("throw-in", "kick off (start)", ...)
    half: int = 0          # 1, 2, or 0 if untagged
    labels: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass
class Match:
    name: str
    n: int
    fps: int
    pitch: tuple                 # (length_m, width_m)
    ball: np.ndarray             # [n, 2] float64, NaN when missing
    players: np.ndarray          # [n, S, 2] float64, NaN when slot empty
    speed: np.ndarray            # [n, S] float64 m/s, NaN when missing
    slot_team: np.ndarray        # [S] int8, H or A
    half: np.ndarray             # [n] int8, 0/1/2
    home_attacks_right: dict     # {1: bool, 2: bool}
    tags: list = field(default_factory=list)   # [Tag]
    homography: object = None    # optional (matrices [n,3,3], reproj_err [n], valid [n])

    @property
    def graded(self):
        return self.half > 0

    @property
    def ball_ok(self):
        return ~np.isnan(self.ball[:, 0])

    def half_blocks(self):
        """[(half, first, last_inclusive)] from the half array."""
        out = []
        for h in (1, 2):
            idx = np.flatnonzero(self.half == h)
            if len(idx):
                out.append((h, int(idx[0]), int(idx[-1])))
        return out


# --------------------------------------------------------------------------------------------
# ATD


def _parse_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams".format(meta_path))
    home_id = teams[0].get("id")
    slot_team = np.array([H if p.get("teamId") == home_id else A for p in root.iter("Player")],
                         dtype=np.int8)
    home_dir = None
    for param in teams[0].iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            home_dir = (param.findtext("Value") or "").strip().lower()
    if home_dir not in ("left_to_right", "right_to_left"):
        raise ValueError("{}: no attack_direction_first_half for home team".format(meta_path))
    right_first = home_dir == "left_to_right"
    names = tuple(t.findtext("Name") for t in teams)
    return slot_team, {1: right_first, 2: not right_first}, names


def _parse_raw(raw_path, n_slots):
    """Raw EPTS text -> (frame_id[n], values[n, 3*S + 3]). Exact float parsing (round_trip)."""
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    blob = blob.replace(b":", b",").replace(b";", b",")
    n_cols = 1 + 3 * n_slots + 3
    table = pd.read_csv(io.BytesIO(blob), header=None, sep=",", dtype=np.float64,
                        float_precision="round_trip", engine="c", na_values=["NaN"])
    if table.shape[1] != n_cols:
        raise ValueError("{}: expected {} columns, got {}".format(raw_path, n_cols, table.shape[1]))
    values = table.to_numpy(dtype=np.float64)
    return values[:, 0].astype(np.int64), values[:, 1:]


def _cache_key(paths):
    parts = []
    for p in paths:
        st = os.stat(p)
        parts.append("{}:{}:{}".format(os.path.basename(p), st.st_size, int(st.st_mtime)))
    return "|".join(parts) + "|v{}".format(CACHE_VERSION)


def load_atd(game=GAME, data_dir=None, source="tactical", use_cache=True):
    """(ball[n,2], players[n,S,2], speed[n,S], slot_team[S], home_attacks_right, team_names)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    slot_team, attack, names = _parse_meta(meta_path)
    n_slots = len(slot_team)
    n = source_frames(source, game)

    cache_path = os.path.join(CACHE_DIR, "atd_{}_{}.npz".format(game, source))
    key = _cache_key([meta_path, raw_path])
    if use_cache and os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["ball"], z["players"], z["speed"], slot_team, attack, names

    frame_id, vals = _parse_raw(raw_path, n_slots)
    idx = frame_id - 1                                   # ATD counts from 1, video from 0
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError("frame ids out of range for {} frames".format(n))
    ball = np.full((n, 2), np.nan)
    players = np.full((n, n_slots, 2), np.nan)
    speed = np.full((n, n_slots), np.nan)
    pl = vals[:, :3 * n_slots].reshape(-1, n_slots, 3)
    players[idx] = pl[:, :, :2]
    speed[idx] = pl[:, :, 2]
    ball[idx] = vals[:, 3 * n_slots:3 * n_slots + 2]

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, key=np.array(key), ball=ball, players=players, speed=speed)
    return ball, players, speed, slot_team, attack, names


# --------------------------------------------------------------------------------------------
# Smart Tagging


def load_tags(game=GAME, data_dir=None, fps=None, team_names=("Barcelona", "Atletic Club")):
    """Smart Tagging as [Tag], video-frame intervals, team mapped to H/A by club name."""
    fps = fps or GAMES[game]["fps"]
    path = require(["smart_tagging"], game, data_dir)[0]
    home, away = (t.strip().lower() for t in team_names)
    out = []
    for inst in ET.parse(path).iter("instance"):
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        team_txt = (labels.get("Team") or "").strip().lower()
        team = H if team_txt == home else A if team_txt == away else -1
        half_txt = (labels.get("Half") or "").strip().lower()
        half = 1 if half_txt.startswith("1st") else 2 if half_txt.startswith("2nd") else 0
        out.append(Tag(code=(inst.findtext("code") or "").strip().upper(),
                       start_frame=int(round(float(inst.findtext("start")) * fps)),
                       end_frame=int(round(float(inst.findtext("end")) * fps)),
                       team=team, type=(labels.get("Type") or "").strip().lower(), half=half,
                       labels=labels))
    out.sort(key=lambda t: (t.start_frame, t.end_frame, t.code))
    return out


# --------------------------------------------------------------------------------------------
# Homography and video


def load_homography(game=GAME, data_dir=None, source="tactical"):
    path = require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    return (table[cols].to_numpy(np.float64).reshape(-1, 3, 3),
            table["reprojection_error"].to_numpy(np.float64),
            table["valid"].to_numpy().astype(bool))


def video_path(game=GAME, data_dir=None, source="tactical"):
    """The video file. The bundle ships e.g. BAR-ATH_tactical-001.mp4 rather than the name
    games.py expects, so fall back to a glob on the stem instead of renaming user files."""
    expected = asset_paths(game, data_dir, source)["video"]
    if os.path.isfile(expected):
        return expected
    stem, ext = os.path.splitext(expected)
    hits = sorted(glob.glob(stem + "*" + ext))
    if not hits:
        raise IOError("no video found for {} ({}*{})".format(source, stem, ext))
    return hits[0]


# --------------------------------------------------------------------------------------------


def half_array(n, game=GAME):
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return half


def load_match(game=GAME, data_dir=None, with_tags=True, with_homography=False, use_cache=True):
    """BAR-ATH (tactical) as a Match."""
    ball, players, speed, slot_team, attack, names = load_atd(game, data_dir, "tactical",
                                                              use_cache)
    n = len(ball)
    return Match(
        name=game, n=n, fps=GAMES[game]["fps"], pitch=tuple(GAMES[game]["pitch"]),
        ball=ball, players=players, speed=speed, slot_team=slot_team,
        half=half_array(n, game), home_attacks_right=attack,
        tags=load_tags(game, data_dir, team_names=names) if with_tags else [],
        homography=load_homography(game, data_dir) if with_homography else None,
    )
