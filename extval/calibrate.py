"""Statistics of the UNLABELLED BAR-ATH tracking, saved as extval/calibration.json (committed).

Only the ATD itself is read (ball / player tracks, the published half blocks): no ground truth,
no Smart Tagging. RULES.md allows this ("statistics of the unlabelled tracking are fine for
calibration"). degrade.py consumes the JSON, so the simulator needs no BAR-ATH data at run time.

    $PY -m extval.calibrate            # writes extval/calibration.json

Contents
  ball.visible_rate           fraction of graded frames with a ball (published: 0.593)
  ball.gap_hist / vis_hist    run-length histograms (frames) of missing / visible stretches
  ball.zone_*                 hazard of a visible->missing transition and the missing-run length
                              histogram, by pitch zone and by local player density at the last seen
                              ball position (zones: 0 central, 1 wide, 2 end, 3 corner)
  players.*                   visible objects per frame, tracklet-length histogram, slot births,
                              loss hazard vs neighbour density, teleports (identity jumps)
  jitter.*                    robust std of the second difference of positions, in metres
"""

import json
import os

import numpy as np

from possession.data import ROOT, load_match

OUT = os.path.join(ROOT, "extval", "calibration.json")
EDGES = [1, 2, 3, 5, 8, 13, 25, 50, 100, 200, 400, 800, 1600, 10 ** 9]   # run-length bin edges
ZONE_N, DENS_N = 4, 3
DENS_EDGES = (3, 6)           # players within RADIUS m: <3, 3-5, >=6
RADIUS = 8.0


def runs_of(mask):
    """(start, length) of True runs."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e - s


def hist(lengths):
    h = np.histogram(lengths, bins=EDGES)[0].astype(float)
    return (h / max(h.sum(), 1)).round(5).tolist()


def zone_of(xy, pitch):
    """0 central, 1 wide strip (<9 m from a touchline), 2 end third-ish (<20 m from a goal line),
    3 corner area (both). xy normalised, [..., 2]."""
    dx = np.minimum(xy[..., 0], 1 - xy[..., 0]) * pitch[0]
    dy = np.minimum(xy[..., 1], 1 - xy[..., 1]) * pitch[1]
    return np.where((dx < 20) & (dy < 9), 3, np.where(dx < 20, 2, np.where(dy < 9, 1, 0)))


def density_at(match, idx, ball=None):
    """# visible players within RADIUS m of the ball (or `ball`) at frames idx."""
    b = (match.ball if ball is None else ball)[idx] * np.array(match.pitch)
    p = match.players[idx] * np.array(match.pitch)
    d = np.sqrt(((p - b[:, None, :]) ** 2).sum(-1))
    return np.nansum(d < RADIUS, axis=1)


def dens_bin(count):
    return np.digitize(count, DENS_EDGES)


def ball_stats(match):
    g = np.flatnonzero(match.graded)
    first, last = g[0], g[-1]
    out = {}
    ok = match.ball_ok
    vis_rate = float(ok[match.graded].mean())
    gaps, vis = [], []
    hz_num = np.zeros((ZONE_N, DENS_N))
    hz_den = np.zeros((ZONE_N, DENS_N))
    gap_by_zone = [[] for _ in range(ZONE_N)]
    for _, a, b in match.half_blocks():
        seg = ok[a:b + 1]
        ms, ml = runs_of(~seg)
        vs, vl = runs_of(seg)
        gaps += ml.tolist()
        vis += vl.tolist()
        # per-frame hazard of leaving the visible state, by zone/density at that frame
        idx = a + np.flatnonzero(seg[:-1])
        zone = zone_of(match.ball[idx], match.pitch)
        dens = dens_bin(density_at(match, idx))
        drop = ~seg[idx - a + 1]
        np.add.at(hz_den, (zone, dens), 1)
        np.add.at(hz_num, (zone, dens), drop)
        # gap length by the zone of the last visible ball position before the gap
        for s, l in zip(ms, ml):
            if s > 0:
                gap_by_zone[int(zone_of(match.ball[a + s - 1][None], match.pitch)[0])].append(l)
    out["step"] = ball_step_stats(match)
    out["nearest_dist_q"] = nearest_dist_quantiles(match).round(3).tolist()
    out["nearest_dist_q_levels"] = list(NEAR_Q)
    out["visible_rate"] = round(vis_rate, 5)
    out["n_gaps"] = len(gaps)
    out["graded_minutes"] = round(len(g) / match.fps / 60, 2)
    out["gap_hist"], out["vis_hist"] = hist(gaps), hist(vis)
    out["gap_mean"], out["vis_mean"] = float(np.mean(gaps)), float(np.mean(vis))
    out["zone_hazard"] = (hz_num / np.maximum(hz_den, 1)).round(6).tolist()
    out["zone_visible_frames"] = hz_den.astype(int).tolist()
    out["zone_gap_hist"] = [hist(x) if len(x) >= 5 else hist(gaps) for x in gap_by_zone]
    out["zone_gap_n"] = [len(x) for x in gap_by_zone]
    return out


NEAR_Q = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def nearest_dist_quantiles(match):
    """Quantiles (m) of the distance from the ball to the nearest tracked object, ball-visible
    graded frames. The one statistic that ties ball and player tracks together."""
    from possession.features import nearest_slot
    _, d = nearest_slot(match)
    d = d[match.graded & match.ball_ok]
    return np.nanquantile(d, NEAR_Q)


def ball_step_stats(match):
    """Frame-to-frame ball motion on graded frames where both frames have a ball (metres): step
    quantiles, shares above 0.5 / 1.5 m, and the direction-reversal share (consecutive step pairs,
    all three frames visible, with a negative dot product). A smooth track has almost no reversals;
    white or fast-changing position error shows up here immediately."""
    b = match.ball * np.array(match.pitch)
    ok = match.ball_ok
    steps, rev_n, rev_d = [], 0, 0
    for _, a, e in match.half_blocks():
        x, k = b[a:e + 1], ok[a:e + 1]
        both = k[1:] & k[:-1]
        d = x[1:] - x[:-1]
        steps.append(np.sqrt((d[both] ** 2).sum(1)))
        triple = both[1:] & both[:-1]
        dot = (d[1:] * d[:-1]).sum(1)[triple]
        rev_n += int((dot < 0).sum())
        rev_d += int(triple.sum())
    st = np.concatenate(steps)
    q = np.quantile(st, (0.5, 0.9, 0.99))
    return {"p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4), "p99": round(float(q[2]), 4),
            "share_gt_0.5": round(float((st > 0.5).mean()), 5),
            "share_gt_1.5": round(float((st > 1.5).mean()), 5),
            "reversal_share": round(rev_n / max(rev_d, 1), 5)}


OUTAGE_THR = 17


def outage_stats(match):
    """Whole-tracking outages: stretches of graded frames with fewer than OUTAGE_THR visible
    objects (of 22). In BAR-ATH they are common-cause (the ball is missing in ~90% of them), not
    independent player losses."""
    vis = (~np.isnan(match.players[..., 0])).sum(1)
    g = match.graded
    low = (vis < OUTAGE_THR) & g
    ls, ll = [], []
    for _, a, b in match.half_blocks():
        s_, l_ = runs_of(low[a:b + 1])
        ls += (s_ + a).tolist()
        ll += l_.tolist()
    mins = g.sum() / match.fps / 60
    return {"threshold": OUTAGE_THR, "share_of_graded": round(float(low[g].mean()), 5),
            "episodes_per_min": round(len(ll) / mins, 4), "length_hist": hist(ll),
            "length_median_frames": float(np.median(ll)),
            "mean_objects": round(float(vis[low].mean()), 3),
            "ball_visible_share": round(float(match.ball_ok[low].mean()), 4)}


def player_step_stats(match):
    """Same idea for player slots: per-slot steps between consecutive visible frames (metres)."""
    scale = np.array(match.pitch)
    pl = match.players * scale
    vis = ~np.isnan(pl[..., 0])
    steps, rev_n, rev_d = [], 0, 0
    for _, a, e in match.half_blocks():
        x, k = pl[a:e + 1], vis[a:e + 1]
        both = k[1:] & k[:-1]
        d = x[1:] - x[:-1]
        steps.append(np.sqrt((d[both] ** 2).sum(-1)))
        triple = both[1:] & both[:-1]
        dot = (d[1:] * d[:-1]).sum(-1)[triple]
        rev_n += int((dot < 0).sum())
        rev_d += int(triple.sum())
    st = np.concatenate(steps)
    q = np.quantile(st, (0.5, 0.9, 0.99, 0.999))
    return {"p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4), "p99": round(float(q[2]), 4),
            "p999": round(float(q[3]), 4), "share_gt_0.5": round(float((st > 0.5).mean()), 6),
            "share_gt_1.5": round(float((st > 1.5).mean()), 6),
            "reversal_share": round(rev_n / max(rev_d, 1), 5)}


def player_stats(match):
    out = {}
    pl = match.players
    vis = ~np.isnan(pl[..., 0])
    gm = match.graded
    n_obj = vis[gm].sum(1)
    out["objects_mean"] = round(float(n_obj.mean()), 3)
    out["objects_hist"] = np.bincount(n_obj, minlength=30)[:30].astype(float).__truediv__(len(n_obj)).round(5).tolist()
    out["frac_over_22"] = round(float((n_obj > 22).mean()), 5)
    for t, nm in ((0, "home"), (1, "away")):
        out["objects_" + nm] = round(float(vis[gm][:, match.slot_team == t].sum(1).mean()), 3)
    # tracklets: visible runs per slot, inside halves
    lens, births, jumps, ends_dens, all_dens = [], 0, 0, [], []
    gap_after = []
    scale = np.array(match.pitch)
    for _, a, b in match.half_blocks():
        v = vis[a:b + 1]
        for s in range(v.shape[1]):
            st, ln = runs_of(v[:, s])
            lens += ln.tolist()
            births += len(st)
            # slot re-use gap (frames the slot is empty between two tracklets)
            gap_after += (st[1:] - (st[:-1] + ln[:-1])).tolist()
            # identity jump: re-acquired within 2 s of the end but >6 m from where it ended
            for k in range(len(st) - 1):
                gap = st[k + 1] - (st[k] + ln[k])
                if gap <= 50:
                    p0 = pl[a + st[k] + ln[k] - 1, s] * scale
                    p1 = pl[a + st[k + 1], s] * scale
                    jumps += float(np.hypot(*(p1 - p0)) > 6.0)
    # loss hazard vs neighbour density: per-frame P(slot visible -> not) by local crowding
    hz = np.zeros((3, 2))
    for _, a, b in match.half_blocks():
        sub = a + np.arange(0, b - a - 5, 5)                     # every 5th frame (speed)
        p = pl[sub] * scale
        d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))
        ns = pl.shape[1]; d[:, np.arange(ns), np.arange(ns)] = np.inf
        crowd = np.nansum(d < 5.0, axis=2)
        nxt = ~np.isnan(pl[sub + 5, :, 0])
        cur = vis[sub]
        cb = np.minimum(crowd, 2)
        for c in range(3):
            sel = cur & (cb == c)
            hz[c, 0] += (sel & ~nxt).sum()
            hz[c, 1] += sel.sum()
    gm_frames = int(gm.sum())
    out["tracklet_hist"] = hist(lens)
    out["tracklet_mean"] = float(np.mean(lens))
    out["births_per_player_min"] = round(births / (22 * gm_frames / match.fps / 60), 3)
    out["slot_reuse_gap_hist"] = hist(gap_after)
    out["slot_reuse_gap_mean"] = float(np.mean(gap_after))
    out["identity_jumps_per_min"] = round(jumps / (gm_frames / match.fps / 60), 3)
    out["loss_hazard_5f_by_crowd"] = (hz[:, 0] / np.maximum(hz[:, 1], 1)).round(5).tolist()
    out["n_slots"] = int(pl.shape[1])
    out["step"] = player_step_stats(match)
    out["outage"] = outage_stats(match)
    return out


def jitter_stats(match):
    """Robust std (1.4826 * MAD) of the second difference of observed positions, metres."""
    out = {}
    scale = np.array(match.pitch)

    def d2(xy):
        a, b, c = xy[:-2], xy[1:-1], xy[2:]
        good = ~(np.isnan(a[..., 0]) | np.isnan(b[..., 0]) | np.isnan(c[..., 0]))
        dd = ((a - 2 * b + c) * scale)[good]
        return dd

    pl_d = np.concatenate([d2(match.players[a:b + 1, s]) for _, a, b in match.half_blocks()
                           for s in range(match.players.shape[1])])
    ball_d = np.concatenate([d2(match.ball[a:b + 1]) for _, a, b in match.half_blocks()])
    for name, dd in (("player", pl_d), ("ball", ball_d)):
        mag = np.sqrt((dd ** 2).sum(1))
        out[name + "_d2_mad_std_m"] = round(float(1.4826 * np.median(np.abs(mag - np.median(mag)))), 5)
        out[name + "_d2_median_m"] = round(float(np.median(mag)), 5)
        out[name + "_d2_p99_m"] = round(float(np.percentile(mag, 99)), 4)
    return out


def main():
    match = load_match(with_tags=False)
    cal = {"source": "BAR-ATH tactical ATD, graded half blocks, unlabelled statistics only",
           "bin_edges_frames": EDGES[:-1] + ["inf"], "radius_m": RADIUS, "dens_edges": list(DENS_EDGES),
           "ball": ball_stats(match), "players": player_stats(match), "jitter": jitter_stats(match)}
    with open(OUT, "w") as fh:
        json.dump(cal, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", os.path.relpath(OUT, ROOT))
    b, p, j = cal["ball"], cal["players"], cal["jitter"]
    print("ball visible {:.3f}, {} gaps (mean {:.0f}f), vis runs mean {:.0f}f".format(
        b["visible_rate"], b["n_gaps"], b["gap_mean"], b["vis_mean"]))
    print("objects {:.2f} (>22: {:.3f}), tracklet mean {:.0f}f, births/player/min {}, jumps/min {}".format(
        p["objects_mean"], p["frac_over_22"], p["tracklet_mean"], p["births_per_player_min"],
        p["identity_jumps_per_min"]))
    print("jitter", j)


if __name__ == "__main__":
    main()
