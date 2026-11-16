"""Mechanical per-frame H/A/D ground truth for external games.

The competition GT (CHALLENGE.md) is: D where the corrected tracking marks the ball dead, otherwise
the team that last had CONTROLLED possession; a loose or contested ball stays with the last team;
no smoothing. We reproduce that from event streams / phase files with the rules below. Frames
outside the graded half blocks (match.half == 0) are X, exactly like BAR-ATH.

Metrica events (games 1-3), rules, in order:
 R1 control events: PASS, SHOT, CARRY, RECOVERY and SET PIECE by a team are "touches with control".
    CHALLENGE, BALL LOST, FAULT RECEIVED, CARD never grant possession.
 R2 a change of possession to team Y happens at the first control event of a Y run (consecutive
    control events of the same team) provided the run is sustained: the next opponent control
    event (or a stoppage) comes at least `min_control` frames later. Shorter runs are touches /
    deflections without control and are absorbed by the previous possessor (CHALLENGE.md: "what
    does not count as a change of possession"). A restart (SET PIECE, goal kick) always sets the
    possessor and is never filtered.
 R3 dead ball starts at: BALL OUT start; a shot with a GOAL subtype (at its end frame); FAULT
    RECEIVED, offside or hand ball (BALL LOST with OFFSIDE / HAND BALL) only when a free kick /
    penalty restart follows within `max_foul_dead` frames (otherwise it was play-on / advantage);
    the end of a half.
 R4 dead ball ends at the next restart: a SET PIECE event (kick off, throw-in, corner, free kick,
    penalty) or a PASS / BALL LOST event with GOAL KICK in its subtype; fallback: the first control
    event after the trigger. The restart frame itself is alive and owned by the restarting team.
 R5 loose ball keeps the last possessor; inside a half there is no other state, so every graded
    frame is H, A or D.

SkillCorner: Phases-of-Play rows are "team in possession" intervals that exist only while the ball
is in play. Frames inside a phase get its team. Between two phases: a gap shorter than `max_loose`
frames is a loose ball handed over / kept (loose ball stays with the previous team); a longer gap
is D. (The gap length histogram is bimodal; see NOTES.md.)
"""

import numpy as np

from possession.data import A, D, H, X

CONTROL = {"PASS", "SHOT", "CARRY", "RECOVERY", "SET PIECE"}
RESTART_SUBS = ("KICK OFF", "THROW IN", "CORNER KICK", "FREE KICK", "PENALTY", "GOAL KICK")
FOUL_TRIGGERS = ("FAULT RECEIVED",)
FOUL_SUBS = ("OFFSIDE", "HAND BALL")

# min_control = 25 frames (1 s): an opponent control event less than 1 s after the previous one is a
# touch / deflection, not control (CHALLENGE.md, "what does not count"; a pass needs ~1 s to arrive).
# CHALLENGE.md also says "a few seconds", so 50 frames is the strict reading; both are scored in NOTES.md.
DEFAULTS = dict(min_control=25, max_foul_dead=250, max_loose=15, max_unlogged=375)


def _is_restart(e):
    if e.type == "SET PIECE":
        return True
    return e.type == "PASS" and "GOAL KICK" in e.sub


def derive_metrica(game, min_control=None, max_foul_dead=None, **_):
    p = {**DEFAULTS, **{k: v for k, v in dict(min_control=min_control,
                                              max_foul_dead=max_foul_dead).items() if v is not None}}
    p.update({k: v for k, v in _.items() if k in DEFAULTS})
    m = game.match
    gt = np.full(m.n, X, dtype=np.int8)
    ev = game.events
    for half, first, last in m.half_blocks():
        he = [e for e in ev if e.period == half and first <= e.start <= last]
        restarts = [e for e in he if _is_restart(e)]
        controls = [e for e in he if e.type in CONTROL]

        # R3 dead triggers (frame, kind)
        triggers = []
        for e in he:
            if e.type == "BALL OUT":
                triggers.append(e.start)
            elif e.type == "SHOT" and "GOAL" in e.sub.split("-") and "OFF" not in e.sub:
                triggers.append(e.end)
            elif e.type in FOUL_TRIGGERS or (e.type == "BALL LOST" and
                                             any(s in e.sub for s in FOUL_SUBS)):
                nxt = [r.start for r in restarts if e.start < r.start <= e.start + p["max_foul_dead"]
                       and any(k in r.sub for k in ("FREE KICK", "PENALTY"))]
                if nxt:
                    triggers.append(e.start)
        # R3b a ball-out restart (throw-in, corner, goal kick, free kick) that no trigger above
        # explains means the ball went out unrecorded: the stoppage starts when the preceding
        # event ends (bounded to `max_unlogged` frames before the restart).
        for r in restarts:
            if "KICK OFF" in r.sub or any(r.start - p["max_unlogged"] <= t <= r.start for t in triggers):
                continue
            prev = [e for e in he if e.end < r.start and e is not r and e.type != "SET PIECE"
                    and e.start < r.start]
            if prev and not any(t > prev[-1].end for t in triggers if t <= r.start):
                triggers.append(max(prev[-1].end, r.start - p["max_unlogged"]))
        triggers.sort()

        # R4 dead intervals [trigger, restart)
        dead = np.zeros(last - first + 1, dtype=bool)
        restart_pts = {}                                   # frame -> team (alive from there)
        for t in triggers:
            if not (first <= t <= last) or dead[t - first]:
                continue                                  # outside the half / already dead
            nxt = [r for r in restarts if r.start >= t]
            if nxt:
                r = nxt[0]
            else:
                c = [c for c in controls if c.start > t]
                r = c[0] if c else None
            end = r.start if r is not None else last + 1
            dead[t - first:min(end, last + 1) - first] = True
            if r is not None and end <= last:
                restart_pts[end] = r.team
        for r in restarts:
            restart_pts.setdefault(r.start, r.team)

        # R1/R2 possession changes. Runs of consecutive same-team control events.
        stops = sorted(triggers)
        runs = []
        for e in controls:
            if runs and runs[-1][0] == e.team and not any(runs[-1][1] < s <= e.start for s in stops):
                continue
            runs.append((e.team, e.start))
        change = []                                        # (frame, team)
        cur = None
        for k, (team, start) in enumerate(runs):
            nxt_start = runs[k + 1][1] if k + 1 < len(runs) else last + 1
            nxt_stop = min([s for s in stops if s > start] + [nxt_start])
            span = min(nxt_start, nxt_stop) - start
            if cur is None or (team != cur and span >= p["min_control"]):
                change.append((start, team))
                cur = team
        pts = sorted(set(change) | set(restart_pts.items()), key=lambda x: x[0])
        team_at = np.full(last - first + 1, -1, dtype=np.int8)
        for (f, team), nxt in zip(pts, pts[1:] + [(last + 1, -1)]):
            team_at[max(f, first) - first:min(nxt[0], last + 1) - first] = team
        # before the first possession change inside the half: owner of the first restart
        if (team_at < 0).any() and pts:
            team_at[team_at < 0] = pts[0][1]
        seg = np.where(dead, D, team_at).astype(np.int8)
        seg[team_at < 0] = D
        gt[first:last + 1] = seg
    return gt


def derive_skillcorner(game, max_loose=None, **_):
    mx = max_loose if max_loose is not None else DEFAULTS["max_loose"]
    m = game.match
    gt = np.full(m.n, X, dtype=np.int8)
    for half, first, last in m.half_blocks():
        seg = np.full(last - first + 1, D, dtype=np.int8)
        ph = sorted((s, e, t) for (p, s, e, t) in game.phases if p == half)
        prev_end, prev_team = None, None
        for s, e, t in ph:
            s, e = max(s, first), min(e, last)
            if e < s:
                continue
            if prev_end is not None and 0 < s - prev_end <= mx:
                seg[prev_end - first + 1:s - first] = prev_team           # short loose gap
            seg[s - first:e - first + 1] = t
            prev_end, prev_team = e, t
        gt[first:last + 1] = seg
    return gt


def derive_dfl_flags(game):
    """DFL/TRACAB: the tracking's own per-frame flags. D where BallStatus is 0 (dead), otherwise the
    team named by BallPossession (1 home, 2 away). This is how the competition GT is described
    (CHALLENGE.md: "D where the corrected tracking marks the ball dead, otherwise the team the
    tracking names as owning it"). Frames outside the two half blocks are X."""
    m, f = game.match, game.flags
    gt = np.full(m.n, X, dtype=np.int8)
    g = m.graded
    team = np.where(f["poss"] == 1, H, A).astype(np.int8)
    gt[g] = np.where(f["status"][g] == 0, D, team[g])
    return gt


def derive_dfl_events(game, **params):
    """DFL event stream (converted to the Metrica schema by extval.dfl.read_events) through the
    Metrica rules R1-R5. A free kick may be taken up to ~30 s after the foul, hence the wider windows."""
    p = dict(max_foul_dead=750, max_unlogged=750)
    p.update(params)
    return derive_metrica(game, **p)


def derive(game, mode=None, **params):
    """int8[n] H/A/D codes with X outside the graded halves. DFL games have two labellings:
    mode "flags" (default, the tracking's own ball status / possession) or "events"."""
    if game.source == "metrica":
        return derive_metrica(game, **params)
    if game.source == "dfl":
        mode = mode or game.label_mode
        return derive_dfl_events(game, **params) if mode == "events" else derive_dfl_flags(game)
    return derive_skillcorner(game, **params)


def spell_stats(gt, match):
    """(#possessions, mean duration s, dead fraction, home share of alive) inside the graded halves."""
    n_sp, lens = 0, []
    for _, first, last in match.half_blocks():
        seg = gt[first:last + 1]
        cuts = np.flatnonzero(np.diff(seg) != 0) + 1
        for a, b in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [len(seg)]))):
            if seg[a] in (H, A):
                n_sp += 1
                lens.append(b - a)
    g = gt[match.graded]
    alive = ((g == H) | (g == A)).sum()
    return n_sp, float(np.mean(lens)) / match.fps if lens else 0.0, float((g == D).mean()), \
        float((g == H).sum() / max(alive, 1))
