"""Sportec / DFL open Bundesliga data (IDSSE) as external games dfl1..dfl7.

Dataset: Bassek, Rein, Weber, Memmert, "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (2025), doi 10.1038/s41597-025-04505-y; figshare doi
10.6084/m9.figshare.28196177; mirror huggingface.co/datasets/pysport/idsse-data. Licence CC-BY 4.0.
7 Bundesliga 2022/23 matches, TRACAB optical tracking at 25 fps, synchronised event data.

We parse the three raw XML files per match directly (no kloppy, so no new dependency): match
information, positions (players + ball) and events. The position file carries, on every BALL frame,
the tracking provider's own flags `BallStatus` (1 alive / 0 dead) and `BallPossession` (1 home / 2
away). They are the closest public analogue of how the competition GT is built ("D where the
corrected tracking marks the ball dead, otherwise the team the tracking names as owning it"), so
they are the primary labels (labels.derive_dfl_flags); the event stream gives a second, rule-based
labelling (events converted to the Metrica Event schema, then labels.derive_metrica).

Conventions: x/y in metres from the centre spot, normalised to 0-1 as x/105 + 0.5, y/68 + 0.5.
The ball's height z is dropped (BAR-ATH's ATD is ground plane only). Half 1 is frames [0, n1), then
GAP_FRAMES of NaN (half time), then half 2. Each half runs kick-off to final whistle, like the
BAR-ATH graded blocks.
"""

import os
import re
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import A, H, Match, CACHE_DIR
from extval.datasets import ExtGame, Event, FPS, PITCH, RAW, _speed

DFL_DIR = os.path.join(RAW, "dfl")
GAP_FRAMES = 500
# TRACAB holds the last ball position, exactly frozen, while BallStatus is dead (measured: zero
# displacement in 100% of dead frames), then jumps at the restart. That is not an observation, and
# BAR-ATH / Metrica show no such plateau. So inside every dead run the ball is treated as unobserved
# (NaN) except for the first DEAD_KEEP_START_S and the last DEAD_KEEP_END_S seconds, which gives the
# same missing-ball structure as Metrica's native track (P(NaN | dead) ~0.8-0.9, see NOTES).
DEAD_KEEP_START_S = 3.0
DEAD_KEEP_END_S = 1.5
HF_BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
# dfl1..dfl7 in this order (the order of the HF file listing)
MATCH_IDS = ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
             "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"]
COMP = {"DFL-MAT-J03WMX": "DFL-COM-000001", "DFL-MAT-J03WN1": "DFL-COM-000001"}
CITATION = ("Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal "
            "and event data in elite soccer. Scientific Data 12, 195 (2025). "
            "doi:10.1038/s41597-025-04505-y (data: figshare doi:10.6084/m9.figshare.28196177, CC-BY 4.0)")


def files(mid):
    com = COMP.get(mid, "DFL-COM-000002")
    return {k: os.path.join(DFL_DIR, "DFL_{}_{}_{}.xml".format(c, k, com + "_" + mid))
            for k, c in (("matchinformation", "02_01"), ("events_raw", "03_02"),
                         ("positions_raw_observed", "04_03"))}


def download():
    """Fetch the 21 files of the 7 matches into extval/raw/dfl (2.6 GB, skipped when present)."""
    import urllib.request
    os.makedirs(DFL_DIR, exist_ok=True)
    for mid in MATCH_IDS:
        for k, path in files(mid).items():
            if not os.path.isfile(path):
                urllib.request.urlretrieve(HF_BASE + os.path.basename(path), path)


def _utc(s):
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _read_info(path):
    root = ET.parse(path).getroot()
    gen = root.find(".//General")
    teams = {}
    for t in root.iter("Team"):
        role = t.get("Role")
        teams[t.get("TeamId")] = (H if role == "home" else A,
                                  {p.get("PersonId"): p.get("PlayingPosition") for p in t.iter("Player")})
    return gen.get("HomeTeamId"), teams


_FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]+)" PersonId="([^"]+)"')


def _parse_positions(path):
    """-> {(section, team_id, person_id): (N int array, xy float array [k,2], extra)}; ball extra is
    (z, possession, status) arrays. Streams the file line by line (~400 MB)."""
    out, cur, rows = {}, None, []

    def flush():
        if cur is not None and rows:
            out[cur] = rows[:]
        rows.clear()

    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("<Frame "):
                p = ln.split('"')
                if cur[1] == "BALL":
                    rows.append((int(p[1]), p[3], float(p[5]), float(p[7]), float(p[9]), int(p[19]), int(p[21])))
                else:
                    rows.append((int(p[1]), p[3], float(p[5]), float(p[7])))
            elif ln.startswith("<FrameSet"):
                flush()
                g = _FS.match(ln)
                cur = (g.group(1), g.group(2), g.group(3))
        flush()
    return out


def _load_arrays(k):
    """Parse (or load the cache of) game k; returns the raw arrays and metadata in a dict."""
    mid = MATCH_IDS[k - 1]
    cpath = os.path.join(CACHE_DIR, "extval_dfl{}.npz".format(k))
    fp = files(mid)
    home_id, teams = _read_info(fp["matchinformation"])
    pids = []                                         # (team role, person, is_gk)
    for tid, (role, players) in teams.items():
        for pid, pos in players.items():
            pids.append((role, pid, pos == "TW"))
    pids.sort(key=lambda x: (x[0], not x[2], x[1]))   # home first, GK first within a team
    col = {pid: i for i, (_, pid, _) in enumerate(pids)}
    slot_team = np.array([r for r, _, _ in pids], dtype=np.int8)
    is_gk = np.array([g for _, _, g in pids])
    if os.path.isfile(cpath):
        z = np.load(cpath)
        return dict(home_id=home_id, slot_team=slot_team, is_gk=is_gk, **{kk: z[kk] for kk in z.files})
    pos = _parse_positions(fp["positions_raw_observed"])
    sections = ("firstHalf", "secondHalf")
    n_h, t0 = {}, {}
    for s in sections:
        lo = min(r[0] for (sec, _, _), v in pos.items() if sec == s for r in v)
        hi = max(r[0] for (sec, _, _), v in pos.items() if sec == s for r in v)
        n_h[s] = (lo, hi - lo + 1)
        t0[s] = next(v[0][1] for (sec, tid, _), v in pos.items() if sec == s and tid == "BALL")
    n1, n2 = n_h["firstHalf"][1], n_h["secondHalf"][1]
    n = n1 + GAP_FRAMES + n2
    off = {"firstHalf": 0, "secondHalf": n1 + GAP_FRAMES}
    half = np.zeros(n, dtype=np.int8)
    half[:n1] = 1
    half[n1 + GAP_FRAMES:] = 2
    ball = np.full((n, 2), np.nan)
    bz = np.full(n, np.nan)
    status = np.full(n, -1, dtype=np.int8)
    poss = np.full(n, -1, dtype=np.int8)
    players = np.full((n, len(pids), 2), np.nan)
    for (sec, tid, pid), rows in pos.items():
        lo = n_h[sec][0]
        idx = np.array([r[0] for r in rows]) - lo + off[sec]
        xy = np.array([[r[2], r[3]] for r in rows])
        if tid == "BALL":
            ball[idx] = xy
            bz[idx] = [r[4] for r in rows]
            poss[idx] = [r[5] for r in rows]
            status[idx] = [r[6] for r in rows]
        elif tid != "referee" and pid in col:
            players[idx, col[pid]] = xy
    scale = np.array(PITCH)
    ball = ball / scale + 0.5
    players = players / scale + 0.5
    arr = dict(ball=ball, ball_z=bz, players=players, status=status, poss=poss, half=half,
               t0=np.array([t0["firstHalf"], t0["secondHalf"]]), lo=np.array([n_h[s][0] for s in sections]),
               off=np.array([off[s] for s in sections]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cpath, **arr)
    return dict(home_id=home_id, slot_team=slot_team, is_gk=is_gk, **arr)


def read_events(path, home_id, t0, off, n):
    """DFL events -> Metrica-schema Event list (see labels.py; control events are PASS / SHOT /
    CARRY / RECOVERY / SET PIECE). Wrapper elements (KickOff, ThrowIn, ...) become a SET PIECE plus
    a BALL OUT (throw-in, corner, goal kick) or FAULT RECEIVED (free kick, penalty) at the referee's
    DecisionTimestamp; the Play inside becomes a PASS; OtherBallAction a CARRY; BallClaiming a
    RECOVERY; goals are SHOTs with a GOAL subtype; offsides BALL LOST / OFFSIDE."""
    root = ET.parse(path).getroot()
    start = [_utc(str(t)) for t in t0]
    ends = []
    for ev in root.iter("Event"):
        fw = ev.find("FinalWhistle")
        if fw is not None:
            ends.append((fw.get("GameSection"), _utc(ev.get("EventTime"))))
    split = [e for s, e in ends if s == "firstHalf"]
    split = split[0] if split else start[1]

    def loc(tstr):
        t = _utc(tstr)
        h = 1 if t < split + timedelta(seconds=1) else 2
        f = int(round((t - start[h - 1]).total_seconds() * FPS)) + int(off[h - 1])
        return h, f

    team_of = lambda tid: H if tid == home_id else A      # noqa: E731
    wrappers = {"KickOff": "KICK OFF", "ThrowIn": "THROW IN", "FreeKick": "FREE KICK",
                "CornerKick": "CORNER KICK", "GoalKick": "GOAL KICK", "Penalty": "PENALTY"}
    out = []
    for ev in root.iter("Event"):
        kids = [c for c in ev]
        if not kids or ev.get("EventTime") is None:
            continue
        c = kids[0]
        h, f = loc(ev.get("EventTime"))
        if not (0 <= f < n):
            continue
        tag = c.tag
        if tag in wrappers:
            team = team_of(c.get("Team"))
            out.append(Event(team, "SET PIECE", wrappers[tag], f, f, h, c.get("Player", "")))
            if c.get("DecisionTimestamp"):
                dh, df = loc(c.get("DecisionTimestamp"))
                if 0 <= df < f:
                    if tag in ("ThrowIn", "CornerKick", "GoalKick"):
                        out.append(Event(team, "BALL OUT", "", df, df, dh, ""))
                    elif tag in ("FreeKick", "Penalty"):
                        out.append(Event(1 - team, "FAULT RECEIVED", "", df, df, dh, ""))
            for pl in c.iter("Play"):
                out.append(Event(team_of(pl.get("Team")), "PASS", "", f, f, h, pl.get("Player", "")))
        elif tag == "Play":
            out.append(Event(team_of(c.get("Team")), "PASS", "", f, f, h, c.get("Player", "")))
        elif tag == "ShotAtGoal":
            goal = c.find(".//SuccessfulShot") is not None
            out.append(Event(team_of(c.get("Team")), "SHOT", "GOAL" if goal else "", f, f, h,
                             c.get("Player", "")))
        elif tag == "OtherBallAction":
            out.append(Event(team_of(c.get("Team")), "CARRY", "", f, f, h, c.get("Player", "")))
        elif tag == "BallClaiming":
            out.append(Event(team_of(c.get("Team")), "RECOVERY", "", f, f, h, c.get("Player", "")))
        elif tag == "Offside":
            out.append(Event(team_of(c.get("Team")), "BALL LOST", "OFFSIDE", f, f, h, ""))
    out.sort(key=lambda e: (e.start, e.end))
    return out


def load_dfl(k):
    a = _load_arrays(k)
    mid = MATCH_IDS[k - 1]
    n = len(a["ball"])
    half = a["half"].astype(np.int8)
    players = a["players"]
    # home keeper's median x per half: home defends the left goal when it is < 0.5
    gk = int(np.flatnonzero((a["slot_team"] == H) & a["is_gk"])[0])
    attack = {}
    for h in (1, 2):
        x = players[half == h, gk, 0]
        attack[h] = bool(np.nanmedian(x) < 0.5)
    ball = a["ball"].copy()
    dead = (a["status"] == 0) & (half > 0)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
    for s, e in zip(edges[::2], edges[1::2]):
        lo, hi = s + int(DEAD_KEEP_START_S * FPS), e - int(DEAD_KEEP_END_S * FPS)
        if hi > lo:
            ball[lo:hi] = np.nan
    m = Match(name="dfl{}".format(k), n=n, fps=FPS, pitch=PITCH, ball=ball, players=players,
              speed=_speed(players, FPS, PITCH), slot_team=a["slot_team"], half=half,
              home_attacks_right=attack)
    events = read_events(files(mid)["events_raw"], a["home_id"], a["t0"], a["off"], n)
    g = ExtGame(m.name, "dfl", m, events)
    g.flags = dict(status=a["status"], poss=a["poss"], ball_z=a["ball_z"])
    return g
