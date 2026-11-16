"""Download and load external games as `possession.data.Match` objects.

Sources (both public GitHub repos, cloned into extval/raw/, which is gitignored):
  - metrica-sports/sample-data   games 1-2 CSV, game 3 FIFA EPTS + JSON events. 25 fps, full-pitch
    optical tracking (same family as the BAR-ATH ATD). Licence: no LICENSE file; the README asks
    users to "be responsible" and to acknowledge the source if used publicly.
  - SkillCorner/opendata         A-League 2024/25 broadcast tracking, 10 fps, MIT licence. Players
    and ball are only present when seen in the broadcast frame (is_detected), so it is naturally
    "noisy"; it ships Phases-of-Play, which give a mechanical H/A/D truth.

Every game becomes an `ExtGame`: a clean `Match` (conventions of possession.data: normalised 0-1
coords, NaN = unobserved, index = frame - 1, slot_team 0/1, half 0/1/2, 25 fps) plus the raw
events (Metrica) or phases (SkillCorner) that labels.py turns into H/A/D ground truth.
The tracking is NOT degraded here; degrade.py does that.
"""

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from possession.data import A, H, Match, CACHE_DIR, ROOT

RAW = os.environ.get("MTAH_EXTVAL_RAW") or os.path.join(ROOT, "extval", "raw")
METRICA_DIR = os.path.join(RAW, "metrica", "data")
SC_DIR = os.path.join(RAW, "skillcorner", "data", "matches")
REPOS = {
    "metrica": "https://github.com/metrica-sports/sample-data.git",
    "skillcorner": "https://github.com/SkillCorner/opendata.git",
}
LICENCES = {
    "metrica": "Metrica Sports open sample data: no licence file; README requires responsible use "
               "and acknowledgement of the source for public use",
    "skillcorner": "MIT (c) SkillCorner",
    "dfl": "CC-BY 4.0, Bassek et al., Scientific Data 2025, doi 10.1038/s41597-025-04505-y (IDSSE)",
}
FPS = 25
PITCH = (105.0, 68.0)

# Event = one entry of the source event stream, on the 0-based video-frame index.
Event = namedtuple("Event", "team type sub start end period player")


@dataclass
class ExtGame:
    name: str
    source: str                   # "metrica" | "skillcorner"
    match: Match                  # clean tracking, tags=[]
    events: list = field(default_factory=list)   # [Event] (Metrica)
    phases: list = field(default_factory=list)   # [(period, start, end, team)] 25 fps (SkillCorner)
    natural_noise: bool = False   # True when the tracking already has broadcast-style dropouts
    flags: dict = field(default_factory=dict)    # DFL: per-frame tracking flags (status, poss, ball_z)
    label_mode: str = "flags"     # DFL: "flags" (tracking ball status / possession) or "events" (event rules)

    @property
    def licence(self):
        return LICENCES[self.source]


# --------------------------------------------------------------------------------------------
# download


def download(which=("metrica", "skillcorner")):
    """git-clone the public repos into extval/raw/ (skipped if present). SkillCorner is ~1.7 GB
    because the tracking files live in git-lfs-style blobs; Metrica is ~150 MB."""
    os.makedirs(RAW, exist_ok=True)
    for name in which:
        dest = os.path.join(RAW, name)
        if os.path.isdir(dest):
            continue
        subprocess.check_call(["git", "clone", "--depth", "1", REPOS[name], dest])


# --------------------------------------------------------------------------------------------
# helpers


def _half_from_ranges(n, ranges):
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in ranges:
        half[first:last + 1] = h
    return half


def _home_attacks_right_from_gk(gk_x_by_half):
    """Home goalkeeper's median x in half h < 0.5 means home defends the left goal."""
    return {h: bool(np.nanmedian(x) < 0.5) for h, x in gk_x_by_half.items()}


def _speed(players, fps, pitch):
    """m/s from +-2 frame central differences of the observed track; NaN where either end is NaN."""
    sp = np.full(players.shape[:2], np.nan)
    d = (players[4:] - players[:-4]) * np.array(pitch)
    sp[2:-2] = np.sqrt((d ** 2).sum(-1)) * fps / 4.0
    return sp


# --------------------------------------------------------------------------------------------
# Metrica CSV (games 1, 2)


def _read_metrica_csv(path):
    raw = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64, na_values=["NaN"])
    names = pd.read_csv(path, header=None, skiprows=2, nrows=1).iloc[0].tolist()
    players = [str(n) for n in names[3:-2:2]]
    xy = raw.to_numpy()
    return (xy[:, 0].astype(np.int64), xy[:, 1].astype(np.int64), players,
            xy[:, 3:-2].reshape(len(xy), len(players), 2), xy[:, -2:])


def _metrica_events_csv(path, n):
    df = pd.read_csv(path)
    out = []
    for r in df.itertuples(index=False):
        team = H if r.Team == "Home" else A if r.Team == "Away" else -1
        sf = int(r[4]) - 1
        ef = int(r[6]) - 1 if not pd.isna(r[6]) else sf
        out.append(Event(team, str(r.Type), str(r.Subtype).upper() if isinstance(r.Subtype, str)
                         else "", sf, max(ef, sf), int(r.Period), str(r[8])))
    out.sort(key=lambda e: (e.start, e.end))
    return out


def load_metrica_csv(k):
    base = os.path.join(METRICA_DIR, "Sample_Game_{}".format(k))
    stem = os.path.join(base, "Sample_Game_{}_".format(k))
    per, fr, hp, hxy, ball = _read_metrica_csv(stem + "RawTrackingData_Home_Team.csv")
    _, _, ap, axy, _ = _read_metrica_csv(stem + "RawTrackingData_Away_Team.csv")
    n = int(fr.max())
    assert (fr == np.arange(1, n + 1)).all()
    players = np.concatenate([hxy, axy], axis=1)
    slot_team = np.array([H] * hxy.shape[1] + [A] * axy.shape[1], dtype=np.int8)
    events = _metrica_events_csv(stem + "RawEventsData.csv", n)
    half = _graded_half(n, per, events)
    gk_x = {h: players[(half == h), 0, 0] for h in (1, 2)}      # home GK is the first column
    attack = _home_attacks_right_from_gk(gk_x)
    match = Match(name="metrica{}".format(k), n=n, fps=FPS, pitch=PITCH, ball=ball.copy(),
                  players=players, speed=_speed(players, FPS, PITCH), slot_team=slot_team,
                  half=half, home_attacks_right=attack)
    return ExtGame(match.name, "metrica", match, events)


def _graded_half(n, period, events, last_frame=None):
    """Half array: from the first kick-off event of the half to the last tracked frame of the period
    (BAR-ATH's graded blocks run kick-off to final whistle)."""
    ranges = []
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        if not len(idx):
            continue
        ko = [e.start for e in events if e.type == "SET PIECE" and "KICK OFF" in e.sub
              and e.period == h]
        first = max(int(idx[0]), min(ko)) if ko else int(idx[0])
        ranges.append((h, first, int(idx[-1])))
    return _half_from_ranges(n, ranges)


# --------------------------------------------------------------------------------------------
# Metrica FIFA EPTS (game 3)


def load_metrica_epts():
    base = os.path.join(METRICA_DIR, "Sample_Game_3")
    meta = ET.parse(os.path.join(base, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in meta.iter("Team")]
    home_id = team_ids[0]
    pteam = {p.get("id"): p.get("teamId") for p in meta.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    gk = {p.get("id"): any(pp.findtext("Name") == "position_type" and
                           pp.findtext("Value") == "Goalkeeper" for pp in p.iter("ProviderParameter"))
          for p in meta.iter("Player")}
    pids = sorted(pteam, key=lambda p: (pteam[p] != home_id, p))
    col = {p: i for i, p in enumerate(pids)}
    slot_team = np.array([H if pteam[p] == home_id else A for p in pids], dtype=np.int8)
    attack_txt = {t.get("id"): t.findtext(".//Value") for t in meta.iter("Team")}

    path = os.path.join(base, "Sample_Game_3_tracking.txt")
    with open(path) as fh:
        lines = fh.read().split("\n")
    lines = [ln for ln in lines if ln]
    n = len(lines)
    players = np.full((n, len(pids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for spec in meta.iter("DataFormatSpecification"):
        s, e = int(spec.get("startFrame")), int(spec.get("endFrame"))
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = chan_player[ref.get("playerChannelId")]
            if ref.get("playerChannelId").endswith("_x"):
                order.append(col[pid])
        block = "\n".join(lines[s - 1:e]).replace(":", ",").replace(";", ",")
        arr = pd.read_csv(pd.io.common.StringIO(block), header=None, dtype=np.float64,
                          na_values=["NaN"]).to_numpy()
        assert arr.shape[1] == 2 * len(order) + 3, (arr.shape, len(order))   # frame id first
        assert int(arr[0, 0]) == s and int(arr[-1, 0]) == e
        for j, c in enumerate(order):
            players[s - 1:e, c] = arr[:, 1 + 2 * j:3 + 2 * j]
        ball[s - 1:e] = arr[:, -2:]

    pm = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")
          if p.findtext("Name") and p.findtext("Name").endswith(("_start", "_end"))}
    ranges = [(1, int(pm["first_half_start"]) - 1, int(pm["first_half_end"]) - 1),
              (2, int(pm["second_half_start"]) - 1, int(pm["second_half_end"]) - 1)]
    events = _metrica_events_json(os.path.join(base, "Sample_Game_3_events.json"), home_id)
    period = _half_from_ranges(n, ranges)
    half = _graded_half(n, period, events)
    right1 = attack_txt[home_id].strip().lower() == "left_to_right"
    attack = {1: right1, 2: not right1}
    match = Match(name="metrica3", n=n, fps=FPS, pitch=PITCH, ball=ball, players=players,
                  speed=_speed(players, FPS, PITCH), slot_team=slot_team, half=half,
                  home_attacks_right=attack)
    return ExtGame("metrica3", "metrica", match, events)


def _metrica_events_json(path, home_id):
    out = []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    for e in data:
        subs = e["subtypes"]
        subs = [] if subs is None else subs if isinstance(subs, list) else [subs]
        team = H if e["team"]["id"] == home_id else A
        out.append(Event(team, e["type"]["name"], "-".join(s["name"] for s in subs).upper(),
                         e["start"]["frame"] - 1, max(e["end"]["frame"] - 1, e["start"]["frame"] - 1),
                         int(e["period"]), (e.get("from") or {}).get("name", "")))
    out.sort(key=lambda e: (e.start, e.end))
    return out


# --------------------------------------------------------------------------------------------
# SkillCorner


def sc_match_ids():
    return sorted(int(d) for d in os.listdir(SC_DIR)) if os.path.isdir(SC_DIR) else []


def _resample_to_25(arr10, t10_valid):
    """arr10 [m, ...] at 10 fps (NaN = unobserved) -> [n25, ...] at 25 fps. Linear between two
    observed neighbours, nearest sample within 50 ms otherwise, NaN beyond."""
    m = len(arr10)
    n25 = int((m - 1) * FPS / 10) + 1
    t = np.arange(n25) * 10.0 / FPS
    i0 = np.minimum(np.floor(t).astype(int), m - 1)
    i1 = np.minimum(i0 + 1, m - 1)
    w = (t - i0).reshape((-1,) + (1,) * (arr10.ndim - 1))
    a, b = arr10[i0], arr10[i1]
    lin = a * (1 - w) + b * w                       # NaN if either end missing
    near = np.where(w < 0.5, a, b)                  # nearest sample (at most 0.5 frame = 50 ms)
    return np.where(np.isnan(lin), near, lin)


def load_skillcorner(match_id, cache=True):
    name = "sc{}".format(match_id)
    folder = os.path.join(SC_DIR, str(match_id))
    cpath = os.path.join(CACHE_DIR, "extval_{}.npz".format(name))
    with open(os.path.join(folder, "{}_match.json".format(match_id)), encoding="utf-8") as fh:
        info = json.load(fh)
    L, W = float(info["pitch_length"]), float(info["pitch_width"])
    home_id = info["home_team"]["id"]
    squad = {p["id"]: (H if p["team_id"] == home_id else A) for p in info["players"]}
    pids = sorted(squad, key=lambda p: (squad[p] != H, p))
    col = {p: i for i, p in enumerate(pids)}
    slot_team = np.array([squad[p] for p in pids], dtype=np.int8)

    if cache and os.path.isfile(cpath):
        z = np.load(cpath)
        ball10, pl10, per10 = z["ball"], z["players"], z["period"]
    else:
        with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)),
                  encoding="utf-8") as fh:
            recs = [json.loads(ln) for ln in fh]
        m = len(recs)
        ball10 = np.full((m, 2), np.nan)
        pl10 = np.full((m, len(pids), 2), np.nan)
        per10 = np.zeros(m, dtype=np.int8)
        for i, r in enumerate(recs):
            per10[i] = r["period"] or 0
            b = r["ball_data"]
            if b["x"] is not None and b["is_detected"]:
                ball10[i] = (b["x"], b["y"])
            for p in r["player_data"]:
                j = col.get(p["player_id"])
                if j is not None and p["is_detected"]:
                    pl10[i, j] = (p["x"], p["y"])
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cpath, ball=ball10, players=pl10, period=per10)

    scale = np.array([L, W])
    ball = _resample_to_25(ball10 / scale + 0.5, None)
    players = _resample_to_25(pl10 / scale + 0.5, None)
    n = len(ball)

    ph = pd.read_csv(os.path.join(folder, "{}_phases_of_play.csv".format(match_id)))
    phases, right = [], {}
    for r in ph.itertuples(index=False):
        team = H if r.team_in_possession_id == home_id else A
        phases.append((int(r.period), int(round(r.frame_start * 2.5)), int(round(r.frame_end * 2.5)),
                       team))
        if team == H:
            right.setdefault(int(r.period), []).append(r.attacking_side == "left_to_right")
    attack = {h: bool(np.mean(v) > 0.5) for h, v in right.items()}
    ranges = []
    for h in (1, 2):
        sel = [p for p in phases if p[0] == h]
        if sel:
            ranges.append((h, min(p[1] for p in sel), min(max(p[2] for p in sel), n - 1)))
    half = _half_from_ranges(n, ranges)
    match = Match(name=name, n=n, fps=FPS, pitch=(L, W), ball=ball, players=players,
                  speed=_speed(players, FPS, (L, W)), slot_team=slot_team, half=half,
                  home_attacks_right=attack)
    return ExtGame(name, "skillcorner", match, phases=phases, natural_noise=True)


# --------------------------------------------------------------------------------------------
# registry

DEFAULT_GAMES = ("metrica1", "metrica2", "metrica3")


def list_games():
    names = ["metrica1", "metrica2", "metrica3"] if os.path.isdir(METRICA_DIR) else []
    return names + ["sc{}".format(i) for i in sc_match_ids()] + [
        "dfl{}".format(i) for i in range(1, 8) if os.path.isdir(os.path.join(RAW, "dfl"))]


_CACHE = {}


def load_game(name):
    if name in _CACHE:
        return _CACHE[name]
    if name in ("metrica1", "metrica2"):
        g = load_metrica_csv(int(name[-1]))
    elif name == "metrica3":
        g = load_metrica_epts()
    elif name.startswith("sc"):
        g = load_skillcorner(int(name[2:]))
    elif name.startswith("dflev"):                 # same tracking as dfl<k>, event-rule labels
        from extval.dfl import load_dfl
        g = load_dfl(int(name[5:]))
        g.label_mode = "events"
    elif name.startswith("dfl"):
        from extval.dfl import load_dfl            # Sportec / DFL IDSSE, see extval/dfl.py
        g = load_dfl(int(name[3:]))
    else:
        raise KeyError("unknown game {!r}; have {}".format(name, list_games()))
    _CACHE[name] = g
    return g


def main():
    download()
    for name in DEFAULT_GAMES:
        g = load_game(name)
        m = g.match
        print("{:9s} n={} halves={} ball_ok={:.3f} slots={} licence: {}".format(
            name, m.n, m.half_blocks(), m.ball_ok[m.graded].mean(), len(m.slot_team), g.licence[:40]))


if __name__ == "__main__":
    main()
