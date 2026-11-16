"""Make a clean external game look like the BAR-ATH ATD.

Everything is seeded (np.random.default_rng from game name + seed) and parameterised by
extval/calibration.json, which holds only unlabelled BAR-ATH statistics (see calibrate.py).

What the BAR-ATH numbers told us (and so what is simulated). v2, 2026-11-09 (v1 = LEGACY_V1):
  * BALL. Metrica's native ball track already misses the ball in long stretches with the same
    structure as BAR-ATH (about 150-170 gaps per game, hazards per zone of the same order), so the
    native gaps are kept. On top, extra dropouts are added with the BAR-ATH zone/density hazard and
    gap-length histogram, with the global hazard scale bisected so the graded visibility equals
    BAR-ATH's 59.3% (no extra dropout when the game is already below it). BAR-ATH's ball is farther
    from the nearest tracked object than Metrica's (median 1.9 m vs 0.3-1.4 m; Metrica's ball is
    snapped to the player), so a SMOOTH position error (Gaussian low-passed noise: core + bursts
    blended by a smooth envelope) is added, fitted to that distance profile and to the frame-to-frame
    step statistics (extval/fit_ball.py). v1 used AR(1) bursts with ~1 m per-frame innovations (a
    jittery ball, reversal share 0.25 vs 0.001 in BAR-ATH) and 3 mm white jitter (both removed).
    SkillCorner games (`natural_noise`) already have broadcast dropouts and are not degraded further.
  * PLAYERS. 22 perfectly identified tracks become 46 anonymous slots (23 per team): each player is
    lost with a per-frame hazard that rises with local crowding (BAR-ATH table), re-acquired after a
    log-normal delay under a NEW slot, plus short-lived duplicate objects (fixed offset; v1 redrew
    the offset every frame). Outside outages BAR-ATH has 21.8 of 22 objects: the rest of the
    shortfall is COMMON-CAUSE OUTAGES (< 17 objects visible, ~50 episodes, 7.7% of frames, ball
    missing in 89%), simulated as episodes with the BAR-ATH length histogram in which ~25% of the
    players survive. BAR-ATH player tracks are smoother than Metrica's, so NO player jitter is added.
    Not simulated (not measurable from unlabelled ATD): swaps of identity between teammates and
    cross-team label swaps (team spread / neighbour-team statistics show no sign of them).
  * TAGS. Synthetic Smart-Tagging-like tags from the ground-truth structure with FP/FN/team/timing
    noise. Their noise levels are assumptions (there is nothing unlabelled to calibrate them on);
    `tag_noise` scales all of them for stress tests.
"""

import json
import os
import zlib
from dataclasses import replace

import numpy as np

from possession.data import A, D, H, ROOT, Match, Tag
from extval.calibrate import EDGES, RADIUS, dens_bin, zone_of
from extval.datasets import _speed

CAL_PATH = os.path.join(ROOT, "extval", "calibration.json")
_CAL = None

DEFAULTS = dict(
    ball_target=0.593,        # published share of graded frames with a ball
    ball_jitter=False,        # v1 had True (3 mm white noise): contradicted by the step statistics
    ball_err_sigma=None,      # 0.0 = no ball position error; None = the fitted model
    error_fit="v2",           # "v2" smooth model (degrade_fit.json) or "v1" legacy AR(1)
    players=True,
    outages=True,             # common-cause tracking outages (new in v2)
    dup_fixed=True,           # duplicate objects keep one fixed offset (v1: new random offset per frame)
    n_slots_per_team=23,      # 46 slots in BAR-ATH
    lost_sigma=1.2,           # log-normal shape of the "player lost" duration
    tags=True,
    tag_noise=1.0,
)
# The pre-2026-11-09 simulator ("v1"), for old-vs-new comparisons: degrade(..., **LEGACY_V1)
LEGACY_V1 = dict(error_fit="v1", ball_jitter=True, outages=False, dup_fixed=False)
TAG_DEFAULT_NOISE = dict(miss=0.08, fp=0.05, team_flip=0.08, jitter_frames=25)


def calibration():
    global _CAL
    if _CAL is None:
        with open(CAL_PATH) as fh:
            _CAL = json.load(fh)
    return _CAL


def _rng(game, seed, salt):
    return np.random.default_rng([zlib.crc32(game.name.encode()), int(seed), salt])


# --------------------------------------------------------------------------------------------
# ball


def _draw_len(rng, hist):
    """Length in frames from a log-binned histogram (log-uniform inside the bin)."""
    k = rng.choice(len(hist), p=np.asarray(hist) / np.sum(hist))
    lo, hi = EDGES[k], (EDGES[k + 1] if k + 1 < len(EDGES) - 1 else 3200)
    return int(np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo + 1)))))


FIT_PATH = os.path.join(ROOT, "extval", "degrade_fit.json")           # v2: smooth error model
FIT_PATH_V1 = os.path.join(ROOT, "extval", "degrade_fit_v1.json")     # v1: AR(1), kept for old-vs-new


def ar1_error(rng, n, sigma_m, tau_frames, pitch, out_frac=0.0, out_sigma_m=0.0, out_tau=75):
    """Slowly varying 2-D ball position error, normalised units: a stationary AR(1) core (std
    sigma_m per axis, correlation time tau_frames; a fast-changing error would show in the step-speed
    distribution, which already matches BAR-ATH without it) plus, for a fraction out_frac of the
    time, bursts (mean length out_tau frames) of a much larger AR(1) error of std out_sigma_m: the
    tracker following the wrong object or a ball far from any player."""
    def ar1(tau):
        rho = np.exp(-1.0 / tau)
        w = rng.normal(0.0, 1.0, (n, 2))
        e = np.empty((n, 2))
        e[0] = w[0]
        for t in range(1, n):
            e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * w[t]
        return e
    err = ar1(tau_frames) * sigma_m
    if out_frac > 0:
        p_end = 1.0 / out_tau
        p_start = out_frac * p_end / (1 - out_frac)
        u = rng.random(n)
        on = np.zeros(n, dtype=bool)
        state = False
        for t in range(n):
            state = (u[t] >= p_end) if state else (u[t] < p_start)
            on[t] = state
        err += ar1(out_tau) * out_sigma_m * on[:, None]
    return err / np.array(pitch)


def _smooth(rng, n, scale_frames):
    """Unit-variance Gaussian-smoothed white noise [n, 2]; scale_frames = std of the kernel."""
    from scipy.ndimage import gaussian_filter1d
    x = gaussian_filter1d(rng.normal(0.0, 1.0, (n, 2)), scale_frames, axis=0, mode="reflect")
    return x * np.sqrt(2.0 * np.sqrt(np.pi) * scale_frames)


def smooth_error(rng, n, pitch, core_sigma_m, core_scale, out_frac, out_sigma_m, out_scale,
                 blend=0.35):
    """v2 ball position error, normalised units. Everything is Gaussian low-pass filtered, so the
    error is smooth frame to frame (unlike v1's AR(1), whose innovations were ~1 m per frame):
       core  : std core_sigma_m, kernel std core_scale frames
       bursts: std out_sigma_m, kernel std out_scale frames, switched in by a smooth envelope
               (sigmoid of low-passed noise, `blend` = softness in std units) that is on for a
               fraction out_frac of the time."""
    err = _smooth(rng, n, core_scale) * core_sigma_m if core_sigma_m > 0 else np.zeros((n, 2))
    if out_frac > 0 and out_sigma_m > 0:
        z = _smooth(rng, n, out_scale)[:, 0]
        theta = np.quantile(z, 1.0 - out_frac)
        env = 1.0 / (1.0 + np.exp(-(z - theta) / blend))
        err = err + _smooth(rng, n, out_scale) * out_sigma_m * env[:, None]
    return err / np.array(pitch)


def ball_error(rng, n, pitch, fit):
    """Dispatch on the fit file's model."""
    if fit.get("model") == "smooth":
        return smooth_error(rng, n, pitch, fit["core_sigma_m"], fit["core_scale"],
                            fit["out_frac"], fit["out_sigma_m"], fit["out_scale"])
    return ar1_error(rng, n, fit["ball_err_sigma_m"], fit["ball_err_tau_frames"], pitch,
                     fit.get("out_frac", 0.0), fit.get("out_sigma_m", 0.0))


def fitted(version="v2"):
    path = FIT_PATH if version == "v2" else FIT_PATH_V1
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return {"model": "smooth", "core_sigma_m": 0.0, "core_scale": 30, "out_frac": 0.0,
            "out_sigma_m": 0.0, "out_scale": 60}


def make_outages(match, rng):
    """Common-cause tracking outages (calibration players.outage): bool[n] on graded frames. Starts
    are Poisson at the BAR-ATH rate, lengths come from its length histogram. In BAR-ATH the ball is
    missing in 89% of outage frames (base rate 41%), so a start is drawn from the frames where the
    clean ball is already missing with that probability, else from frames where it is visible; the
    ball itself is not touched (the extra-dropout step fixes the overall visibility)."""
    cal = calibration()["players"]["outage"]
    out = np.zeros(match.n, dtype=bool)
    g = match.graded
    miss, vis = np.flatnonzero(g & ~match.ball_ok), np.flatnonzero(g & match.ball_ok)
    n_ep = rng.poisson(cal["episodes_per_min"] * g.sum() / match.fps / 60)
    # stratified lengths (one draw per equal-probability slice of the BAR-ATH length histogram,
    # shuffled): the distribution is heavy tailed, i.i.d. draws would make the outage share swing
    # between 3% and 20% from game to game
    cum = np.cumsum(cal["length_hist"]) / np.sum(cal["length_hist"])
    for u in (rng.permutation(n_ep) + rng.random(n_ep)) / max(n_ep, 1):
        k = min(int(np.searchsorted(cum, u)), len(cum) - 1)
        lo, hi = EDGES[k], (EDGES[k + 1] if k + 1 < len(EDGES) - 1 else 3200)
        ln = int(np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo + 1)))))
        pool = miss if (rng.random() < 1.0 - cal["ball_visible_share"] and len(miss)) else vis
        t = int(rng.choice(pool))
        out[t:t + max(1, ln)] = True
    return out & g


def degrade_ball(match, rng, target=0.593, jitter=True, err_sigma=None, error_fit="v2",
                 outage=None):
    cal = calibration()["ball"]
    ball = match.ball.copy()
    native = ~np.isnan(ball[:, 0])
    g = match.graded
    out = {"native_visible": float(native[g].mean())}
    hz_tab = np.array(cal["zone_hazard"])
    zone = np.zeros(match.n, dtype=int)
    dens = np.zeros(match.n, dtype=int)
    idx = np.flatnonzero(native)
    zone[idx] = zone_of(ball[idx], match.pitch)
    b = ball[idx] * np.array(match.pitch)
    cnt = np.zeros(len(idx))
    for s in range(0, len(idx), 20000):                       # chunked: keeps memory small
        sl = slice(s, s + 20000)
        d = np.sqrt(((match.players[idx[sl]] * np.array(match.pitch) - b[sl][:, None, :]) ** 2).sum(-1))
        cnt[sl] = np.nansum(d < RADIUS, axis=1)
    dens[idx] = dens_bin(cnt)
    haz = hz_tab[zone, dens]
    haz[~native] = 0.0
    u = rng.random(match.n)
    # per-candidate gap lengths for each zone, drawn up front so the scale search is monotone
    n_cand = int((u < np.minimum(haz * 50, 1.0)).sum()) + 1
    lens = np.array([[_draw_len(rng, cal["zone_gap_hist"][z]) for z in range(4)]
                     for _ in range(n_cand)])
    cand_all = np.flatnonzero(u < np.minimum(haz * 50, 1.0))

    def apply(scale):
        cand = np.flatnonzero(u < np.minimum(haz * scale, 1.0))
        rank = {t: j for j, t in enumerate(cand_all)}
        drop = np.zeros(match.n, dtype=bool)
        resume = 0
        for t in cand:
            if t < resume or not g[t] or not native[t]:
                continue
            ln = int(lens[rank[t], zone[t]])
            drop[t:t + ln] = True
            resume = t + ln
        return drop

    if native[g].mean() > target:
        lo, hi = 0.0, 50.0
        for _ in range(14):
            mid = (lo + hi) / 2
            vis = (native & ~apply(mid))[g].mean()
            lo, hi = (mid, hi) if vis > target else (lo, mid)
        scale = (lo + hi) / 2
        ball[apply(scale)] = np.nan
        out["extra_scale"] = scale
    else:
        out["extra_scale"] = 0.0
    if jitter:
        # BAR-ATH ball second-difference is ~9 mm (median), Metrica's ~1 mm: white noise sigma
        # solves  target^2 = native^2 + 6 sigma^2 * 1.177^2  (median of a 2-D Rayleigh).
        tgt = calibration()["jitter"]["ball_d2_median_m"]
        a, b_, c = ball[:-2], ball[1:-1], ball[2:]
        good = ~(np.isnan(a[:, 0]) | np.isnan(b_[:, 0]) | np.isnan(c[:, 0]))
        nat = np.median(np.sqrt((((a - 2 * b_ + c) * np.array(match.pitch))[good] ** 2).sum(1)))
        if tgt > nat:
            sigma = np.sqrt((tgt ** 2 - nat ** 2) / 6.0) / 1.177          # metres per axis
            noise = rng.normal(0.0, sigma, ball.shape) / np.array(match.pitch)
            ball = ball + noise
            out["ball_sigma_m"] = float(sigma)
    fit = fitted(error_fit)
    if err_sigma == 0.0:
        pass                                          # explicit "no position error" (used by fit())
    else:
        ball = ball + ball_error(rng, match.n, match.pitch, fit)
        out["ball_error_model"] = fit.get("model", "ar1")
    out["visible"] = float((~np.isnan(ball[:, 0]))[g].mean())
    return ball, out


# --------------------------------------------------------------------------------------------
# players


def degrade_players(match, rng, n_slots_per_team=23, lost_sigma=1.2, outage=None, dup_fixed=True):
    cal = calibration()["players"]
    n, S = match.n, match.players.shape[1]
    scale = np.array(match.pitch)
    clean = match.players
    present = ~np.isnan(clean[..., 0])
    # per-frame crowding (neighbours within 5 m) -> loss hazard (table is per 5 frames)
    crowd = np.zeros((n, S), dtype=np.int8)
    for s0 in range(0, n, 5000):
        sl = slice(s0, min(s0 + 5000, n))
        p = clean[sl] * scale
        d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))
        d[:, np.arange(S), np.arange(S)] = np.inf
        crowd[sl] = np.minimum(np.nansum(d < 5.0, axis=2), 2)
    haz = np.array(cal["loss_hazard_5f_by_crowd"]) / 5.0
    u = rng.random((n, S))
    # lost duration: log-normal with mean chosen so mean visible objects = BAR-ATH's
    mean_vis = cal["objects_mean"]
    if outage is not None:      # outages explain part of the shortfall: independent losses only the rest
        o = cal["outage"]
        mean_vis = (cal["objects_mean"] - o["share_of_graded"] * o["mean_objects"]) / (1 - o["share_of_graded"])
    f_lost = max(1.0 - mean_vis / 22.0, 1e-3)
    mean_lost = f_lost / (1 - f_lost) * cal["tracklet_mean"]
    mu = np.log(mean_lost) - lost_sigma ** 2 / 2

    tracked = present.copy()
    for s in range(S):
        cand = np.flatnonzero((u[:, s] < haz[crowd[:, s]]) & present[:, s])
        resume = 0
        for t in cand:
            if t < resume:
                continue
            ln = max(1, int(rng.lognormal(mu, lost_sigma)))
            tracked[t:t + ln, s] = False
            resume = t + ln
    if outage is not None and outage.any():
        keep = calibration()["players"]["outage"]["mean_objects"] / 22.0
        starts = np.flatnonzero(np.diff(np.concatenate(([0], outage.astype(np.int8)))) == 1)
        ends = np.flatnonzero(np.diff(np.concatenate((outage.astype(np.int8), [0]))) == -1) + 1
        for a, b in zip(starts, ends):
            tracked[a:b, rng.random(S) >= keep] = False          # a fixed random subset survives
    # tracklets -> slots, per team, lowest free slot first
    out = np.full((n, 2 * n_slots_per_team, 2), np.nan)
    slot_team = np.array([H] * n_slots_per_team + [A] * n_slots_per_team, dtype=np.int8)
    segs = []
    for s in range(S):
        d = np.diff(np.concatenate(([0], tracked[:, s].astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            segs.append((a, b, s))
    segs.sort()
    free_at = {t: np.zeros(n_slots_per_team, dtype=np.int64) for t in (H, A)}
    n_births = 0
    for a, b, s in segs:
        team = match.slot_team[s]
        fa = free_at[team]
        k = int(np.flatnonzero(fa <= a)[0]) if (fa <= a).any() else int(np.argmin(fa))
        fa[k] = b
        col = k + (0 if team == H else n_slots_per_team)
        out[a:b, col] = clean[a:b, s]
        n_births += 1
    # duplicates: short-lived second objects shadowing a tracked player, hazard tuned so that the
    # fraction of graded frames with more than 22 objects matches the ATD (2.4%).
    out, dup_rate = _add_duplicates(match, rng, out, slot_team, tracked, clean,
                                    cal["frac_over_22"], n_slots_per_team, dup_fixed=dup_fixed)
    g = match.graded
    stats = {"objects_mean": float((~np.isnan(out[..., 0])).sum(1)[g].mean()),
             "frac_over_22": float(((~np.isnan(out[..., 0])).sum(1) > 22)[g].mean()),
             "births": n_births, "dup_rate": dup_rate}
    return out, slot_team, stats


def _add_duplicates(match, rng, out, slot_team, tracked, clean, target, nspt, mean_len=60,
                    dup_fixed=True):
    n, S = tracked.shape
    g = match.graded
    base_n = (~np.isnan(out[..., 0])).sum(1)
    u = rng.random((n, S))
    lens = rng.geometric(1.0 / mean_len, size=(n, S))
    offs = rng.normal(0, 0.6, (n, S, 2)) / np.array(match.pitch)
    best = None
    for rate in (0.0, 2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4):
        cnt = base_n.copy()
        dup = np.zeros(n, dtype=np.int16)
        events = []
        for s in range(S):
            for t in np.flatnonzero((u[:, s] < rate) & tracked[:, s]):
                ln = lens[t, s]
                events.append((t, min(t + ln, n), s))
                dup[t:t + ln] += 1
        frac = float(((base_n + dup) > 22)[g].mean())
        best = (rate, events) if best is None or abs(frac - target) < abs(best[2] - target) else best
        best = (best[0], best[1], frac) if len(best) == 2 else best
        if frac >= target:
            break
    rate, events = best[0], best[1]
    free_at = np.zeros(out.shape[1], dtype=np.int64)
    for a, b, s in sorted(events):
        team = match.slot_team[s]
        cols = np.arange(nspt) + (0 if team == H else nspt)
        free = [c for c in cols if np.isnan(out[a:b, c, 0]).all()]
        if not free:
            continue
        c = free[0]
        out[a:b, c] = clean[a:b, s] + (offs[a, s] if dup_fixed else offs[a:b, s])
    return out, rate


# --------------------------------------------------------------------------------------------
# tags


def _first_ball(match, frame, span=75):
    seg = match.ball[frame:frame + span]
    ok = np.flatnonzero(~np.isnan(seg[:, 0]))
    return seg[ok[0]] if len(ok) else None


def make_tags(match, gt, game, rng, noise=1.0):
    """Synthetic Smart-Tagging-like tags from the ground-truth structure (see module docstring)."""
    nz = {k: v * noise for k, v in TAG_DEFAULT_NOISE.items()}
    tags = []
    fps = match.fps
    sign_right = lambda h: match.home_attacks_right[h]       # noqa: E731

    def team_noise(team):
        return (1 - team) if (team in (H, A) and rng.random() < nz["team_flip"]) else team

    def jit():
        return int(rng.normal(0, nz["jitter_frames"]))

    for half, first, last in match.half_blocks():
        seg = gt[first:last + 1]
        alive = seg < D
        # restarts: dead -> alive transitions (plus the half kick-off)
        starts = [0] if alive[0] else []
        starts += (np.flatnonzero(~alive[:-1] & alive[1:]) + 1).tolist()
        for k, rel in enumerate(starts):
            f = first + rel
            team = int(seg[rel])
            ball = _first_ball(match, f)
            typ = "free kick"
            if k == 0:
                typ = "kick off (start)"
            elif ball is not None:
                bx, by = ball
                dx = min(bx, 1 - bx) * match.pitch[0]
                dy = min(by, 1 - by) * match.pitch[1]
                if dx < 4 and dy < 4:
                    typ = "corner kick"
                elif dy < 2.5:
                    typ = "throw-in"
                elif dx < 9 and abs(by - 0.5) * match.pitch[1] < 13:
                    typ = "goal kick"
                elif abs(bx - 0.5) * match.pitch[0] < 2 and abs(by - 0.5) * match.pitch[1] < 2:
                    typ = "kick off (after goal)"
            if rng.random() < nz["miss"]:
                continue
            tags.append(Tag("SET PIECES", f - 50 + jit(), f + 25 + jit(),
                            -1 if typ == "throw-in" else team_noise(team), typ, half))
        # phases: split each possession by thirds from the possessor's point of view
        sg = np.flatnonzero(np.diff(np.concatenate(([0], (seg < D).astype(np.int8), [0]))))
        for a, b in zip(sg[::2], sg[1::2]):
            team = int(seg[a])
            if team not in (H, A):
                continue
            right = sign_right(half) == (team == H)
            xs = match.ball[first + a:first + b, 0]
            if np.isnan(xs).all():
                continue
            prog = np.where(right, xs, 1 - xs)
            third = np.digitize(prog, (1 / 3, 2 / 3))                      # 0 own, 1 mid, 2 final
            # run-length of the third with ffill over NaN; drop runs shorter than 2 s
            t_ff = np.array(third, dtype=float)
            t_ff[np.isnan(xs)] = np.nan
            idx = np.where(~np.isnan(t_ff), np.arange(len(t_ff)), 0)
            np.maximum.accumulate(idx, out=idx)
            t_ff = t_ff[idx]
            t_ff[np.isnan(t_ff)] = 1
            cuts = np.flatnonzero(np.diff(t_ff) != 0) + 1
            bounds = np.concatenate(([0], cuts, [len(t_ff)]))
            keep = [(s_, e_) for s_, e_ in zip(bounds[:-1], bounds[1:]) if e_ - s_ >= 2 * fps] or \
                   [(0, len(t_ff))]
            for s_, e_ in keep:
                th = int(t_ff[s_])
                for t_team, code in ((team, ("BUILD UP", "PROGRESSION", "CREATING CHANCES")[th]),
                                     (1 - team, ("DEFENDING IN ATTACKING THIRD",
                                                 "DEFENDING IN MIDDLE THIRD",
                                                 "DEFENDING IN DEFENSIVE THIRD")[th])):
                    if rng.random() < nz["miss"]:
                        continue
                    tags.append(Tag(code, first + a + s_ + jit(), first + a + e_ + jit(),
                                    team_noise(t_team), "", half))
    for e in game.events:                                                 # goals (Metrica only)
        if e.type == "SHOT" and "GOAL" in e.sub.split("-") and "OFF" not in e.sub \
                and match.half[e.end] > 0 and rng.random() >= nz["miss"]:
            tags.append(Tag("GOALS", e.end - 25 + jit(), e.end + 25 + jit(), team_noise(e.team), "",
                            int(match.half[e.end])))
    tags = [replace(t, start_frame=max(0, t.start_frame),
                    end_frame=max(t.start_frame + 1, t.end_frame)) for t in tags]
    tags.sort(key=lambda t: (t.start_frame, t.end_frame, t.code))
    return tags


# --------------------------------------------------------------------------------------------


def degrade(game, seed=0, **params):
    """ExtGame -> (degraded Match, ground-truth codes, info dict). Deterministic in (game, seed)."""
    from extval.labels import derive
    p = {**DEFAULTS, **params}
    m = game.match
    gt = derive(game)
    info = {}
    ball, players, slot_team, speed = m.ball, m.players, m.slot_team, m.speed
    if not game.natural_noise:
        outage = make_outages(m, _rng(m, seed, 4)) if p["outages"] else None
        ball, info["ball"] = degrade_ball(m, _rng(m, seed, 1), p["ball_target"], p["ball_jitter"],
                                          p["ball_err_sigma"], p["error_fit"], outage)
        if p["players"]:
            players, slot_team, info["players"] = degrade_players(
                replace(m, ball=ball), _rng(m, seed, 2), p["n_slots_per_team"], p["lost_sigma"],
                outage, p["dup_fixed"])
            speed = _speed(players, m.fps, m.pitch)
    out = replace(m, ball=ball, players=players, speed=speed, slot_team=slot_team, tags=[])
    if p["tags"]:
        out.tags = make_tags(m, gt, game, _rng(m, seed, 3), p["tag_noise"])
    out.name = "{}@s{}".format(m.name, seed)
    return out, gt, info


# --------------------------------------------------------------------------------------------
# fit: ball position error from the unlabelled BAR-ATH nearest-object-distance profile


