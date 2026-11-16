"""Self-supervised evaluation of the spike detections against the ATD (no ground truth used).

    video/.venv/Scripts/python video/spike_eval.py full_s
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import CACHE_DIR, load_homography, load_match  # noqa: E402

THRESH = (0.05, 0.1, 0.2, 0.3, 0.5)
VMAX = 40.0          # m/s, upper bound on ball speed (hard kicks ~35 m/s)
FPS = 25
PX_NEAR = (25, 50)   # "detection at the ATD ball" radii in px
Y_BIAS = -9.0        # px: detected ball sits above the ground-plane ATD projection (airborne)


def proj(Hm, x, y):
    d = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
    return (Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) / d, (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]) / d


def load(name):
    z = np.load(os.path.join(CACHE_DIR, "video_spike", name + ".npz"))
    return {k: z[k] for k in z.files}


def main(name, restrict=None):
    z = load(name)
    if restrict:
        keep = np.isin(z["frame"], load(restrict)["frame"])
        z = {k: (v[keep] if k in ("frame", "kind", "clip", "det") else v) for k, v in z.items()}
    m = load_match(with_tags=False)
    Hs, _, _ = load_homography()
    L, W = m.pitch
    fr, kind, clip, det = z["frame"], z["kind"], z["clip"], z["det"]
    n = len(fr)
    print("sample {}: {} frames ({} gap, {} has-ball), {:.3f} s/frame detect".format(
        name, n, int((kind == 1).sum()), int((kind == 0).sum()), float(z["sec_per_frame"])))

    # pitch coords of every candidate
    inv = np.linalg.inv(Hs[fr])
    px, py = det[:, :, 0], det[:, :, 1]
    pit_x = np.full_like(px, np.nan); pit_y = np.full_like(px, np.nan)
    for i in range(n):
        ok = ~np.isnan(px[i])
        pit_x[i, ok], pit_y[i, ok] = proj(inv[i], px[i, ok], py[i, ok])
    score = det[:, :, 4]

    # ---------------- A. frames where the ATD HAS the ball
    h = np.flatnonzero(kind == 0)
    bx = np.empty(len(h)); by = np.empty(len(h)); near = np.empty(len(h))
    for j, i in enumerate(h):
        f = fr[i]
        bx[j], by[j] = proj(Hs[f], *m.ball[f])
        pl = m.players[f]
        okp = ~np.isnan(pl[:, 0])
        d = np.hypot((pl[okp, 0] - m.ball[f, 0]) * L, (pl[okp, 1] - m.ball[f, 1]) * W)
        near[j] = d.min() if okp.any() else np.nan
    dist = np.hypot(det[h, :, 0] - bx[:, None], det[h, :, 1] - (by[:, None] + Y_BIAS))   # [nh,5]
    in_view = (bx > 0) & (bx < 1920) & (by > 0) & (by < 1080)
    print("\nA. ATD-has-ball frames: {} (ATD ball inside image: {})".format(len(h), int(in_view.sum())))
    for stratum, sel in (("all", in_view), ("ball <=2 m from a player", in_view & (near <= 2)),
                         ("ball >2 m from all players", in_view & (near > 2))):
        print("  [{}] n={}".format(stratum, int(sel.sum())))
        print("    tau   recall@25px recall@50px | top1 precision@50px | frames with any cand")
        for t in THRESH:
            s = score[h][sel]; dd = dist[sel]
            ok = s >= t
            hit15 = ((dd <= 25) & ok).any(1).mean(); hit30 = ((dd <= 50) & ok).any(1).mean()
            top = np.nanargmax(np.where(ok, s, -1), 1)
            has = ok.any(1)
            top_ok = dd[np.arange(len(dd)), top] <= 50
            prec = (top_ok & has).sum() / max(has.sum(), 1)
            print("    {:.2f}   {:6.3f}      {:6.3f}   | {:6.3f} (n det {:4d}) | {:6.3f}".format(
                t, hit15, hit30, prec, int(has.sum()), has.mean()))
    # false-detection rate per frame on has-ball frames: cands >tau farther than 30px from ATD ball
    print("  extra candidates (>50 px from the ATD ball) per frame:",
          ", ".join("tau {:.2f}: {:.2f}".format(t, (((dist > 50) & (score[h] >= t)).sum(1)).mean())
                    for t in THRESH))

    # ---------------- B. frames inside long ATD gaps
    g = np.flatnonzero(kind == 1)
    print("\nB. gap frames: {} in {} clips".format(len(g), len(np.unique(clip[g]))))
    # nearest ATD ball before/after the gap for each clip
    cx_m = pit_x * L; cy_m = pit_y * W
    # static cells over ALL candidates >=0.1 (gap + has) -- pitch cells of 1.5 m
    def cell(a, b):
        return (np.floor(np.nan_to_num(a, nan=-1e3) / 1.5).astype(int) * 1000
                + np.floor(np.nan_to_num(b, nan=-1e3) / 1.5).astype(int))
    out_rows = []
    for t in THRESH:
        sel = score >= t
        cells = cell(cx_m, cy_m)
        # clips (any kind) per cell
        from collections import defaultdict
        cc = defaultdict(set)
        for i, k in zip(*np.nonzero(sel)):
            cc[cells[i, k]].add(clip[i])
        static = {c for c, s_ in cc.items() if len(s_) >= 3}
        # top-1 per gap frame
        tot = len(g); has = on_pitch = nonstatic = chain = plaus = moving = 0
        n_static_hits = 0
        for i in g:
            ok = sel[i]
            if not ok.any():
                continue
            has += 1
            k = int(np.nanargmax(np.where(ok, score[i], -1)))
            xx, yy = cx_m[i, k], cy_m[i, k]
            onp = (-3 <= xx <= L + 3) and (-3.5 <= yy <= W + 3.5)
            on_pitch += onp
            st = cells[i, k] in static
            n_static_hits += st
            nonstatic += (not st)
            # chain: any candidate >=t in the adjacent sampled frame of the same clip within vmax
            ch = False
            for j in (i - 1, i + 1):
                if 0 <= j < n and clip[j] == clip[i]:
                    d = np.hypot(cx_m[j][sel[j]] - xx, cy_m[j][sel[j]] - yy)
                    dt = abs(fr[j] - fr[i]) / FPS
                    ch |= bool((d <= VMAX * dt).any())
            chain += ch
            plaus += (onp and (not st) and ch)
        print("  tau {:.2f}: det {:5.1%} | on pitch {:5.1%} | static-cell {:5.1%} | chain-consistent {:5.1%}"
              " | plausible(all 3) {:5.1%}   (shares of all gap frames; plausible = on pitch & not static"
              " & chain)".format(t, has / tot, on_pitch / tot, n_static_hits / tot, chain / tot,
                                 plaus / tot))
        out_rows.append((t, has / tot, plaus / tot))

    # per-clip: how many clips have a plausible moving track at tau=0.2, stationary-in-clip
    t = 0.2
    sel = score >= t
    print("\n  per-clip at tau {:.2f} (top-1 per frame): ".format(t))
    mov = stat = nonet = 0
    for c in np.unique(clip[g]):
        idx = np.flatnonzero(clip == c)
        pts = []
        for i in idx:
            ok = sel[i]
            if ok.any():
                k = int(np.nanargmax(np.where(ok, score[i], -1)))
                pts.append((cx_m[i, k], cy_m[i, k]))
        if len(pts) < 4:
            nonet += 1
            continue
        pts = np.array(pts)
        span = np.hypot(*(pts.max(0) - pts.min(0)))
        if span < 1.0:
            stat += 1
        else:
            mov += 1
    print("    clips with <4 detections: {}, stationary (<1 m span): {}, moving: {}".format(nonet, stat, mov))

    # distance of gap detections to nearest ATD player (ball-vs-boot/head confusion)
    d_pl = []
    for i in g:
        ok = score[i] >= 0.2
        if not ok.any():
            continue
        k = int(np.nanargmax(np.where(ok, score[i], -1)))
        pl = m.players[fr[i]]; okp = ~np.isnan(pl[:, 0])
        if okp.any():
            d_pl.append(np.hypot(pl[okp, 0] * L - cx_m[i, k], pl[okp, 1] * W - cy_m[i, k]).min())
    d_pl = np.array(d_pl)
    print("    gap detections (tau .2): distance to nearest tracked player m: p10 {:.1f} p50 {:.1f} p90 {:.1f}"
          "; <1 m: {:.0%}".format(*np.percentile(d_pl, [10, 50, 90]), (d_pl < 1).mean()))
    # same for ATD ball on has-ball frames (reference)
    print("    reference, ATD ball to nearest player on has-ball frames: p10 {:.1f} p50 {:.1f} p90 {:.1f}"
          .format(*np.nanpercentile(near, [10, 50, 90])))

    # ---------------- C. filtered detector: on pitch, not a repeated static cell, <=10 m from a tracked player
    from collections import defaultdict
    print("\nC. filtered top-1 (on pitch & not static pitch cell & <=10 m from a tracked player)")
    print("   tau   has-ball: top1 within 50px of ATD / all frames (yield)  precision | gap: share with filtered det"
          " | chain-consistent share")
    cells = cell(cx_m, cy_m)
    for t in (0.1, 0.2, 0.3, 0.5):
        sel = score >= t
        cc = defaultdict(set)
        for i, k in zip(*np.nonzero(sel)):
            cc[cells[i, k]].add(clip[i])
        static = {c for c, s_ in cc.items() if len(s_) >= 3}
        top = np.full(n, -1)
        for i in range(n):
            pl = m.players[fr[i]]; okp = ~np.isnan(pl[:, 0])
            best, bs = -1, -1
            for k in range(det.shape[1]):
                if not sel[i, k]:
                    continue
                xx, yy = cx_m[i, k], cy_m[i, k]
                if not ((-3 <= xx <= L + 3) and (-3.5 <= yy <= W + 3.5)) or cells[i, k] in static:
                    continue
                if okp.any() and np.hypot(pl[okp, 0] * L - xx, pl[okp, 1] * W - yy).min() > 10:
                    continue
                if score[i, k] > bs:
                    best, bs = k, score[i, k]
            top[i] = best
        hh = np.arange(len(h))
        has_top = top[h] >= 0
        d_top = dist[hh, np.maximum(top[h], 0)]
        hit = has_top & (d_top <= 50)
        gdet = top[g] >= 0
        gch = 0
        for i in g:
            if top[i] < 0:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < n and clip[j] == clip[i] and top[j] >= 0:
                    if np.hypot(cx_m[j, top[j]] - cx_m[i, top[i]], cy_m[j, top[j]] - cy_m[i, top[i]]) <= VMAX * 3 / FPS:
                        gch += 1
                        break
        print("   {:.2f}   {:5.1%} of frames have a det; hit {:5.1%} of frames; precision {:5.1%} |"
              " {:5.1%} | {:5.1%}".format(t, has_top.mean(), hit.mean(), hit.sum() / max(has_top.sum(), 1),
                                          gdet.mean(), gch / len(g)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
