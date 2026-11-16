"""Feasibility spike: sample frames, run the tile detector, save raw candidates to cache/video_spike/.

Sampling is seeded and derived only from the unlabelled ATD ball-missing mask and the half blocks:
  - "gap" clips: CLIP consecutive frames fully inside a graded ATD ball-missing run of >= MIN_GAP
    frames (start drawn uniformly over all such gap frames, so long gaps weigh more);
  - "has" clips: CLIP consecutive frames where the ATD has the ball throughout.
Every STRIDE-th frame of a clip is detected. Outputs stay in cache/ (never committed).

    video/.venv/Scripts/python video/spike_run.py --windows full --out full_s
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ballnet  # noqa: E402
from possession.data import CACHE_DIR, load_homography, load_match, video_path  # noqa: E402

CLIP, STRIDE, MIN_GAP = 30, 3, 75
SEED = 20271117


def runs(mask):
    """(start, end_exclusive) of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def build_sample(match, n_gap=60, n_has=40, seed=SEED):
    """-> list of dicts(kind, start, gap_start, gap_end). Deterministic."""
    rng = np.random.default_rng(seed)
    miss = match.graded & ~match.ball_ok
    s, e = runs(miss)
    keep = (e - s) >= MIN_GAP
    gs, ge = s[keep], e[keep]
    # valid clip starts per gap, weighted by count
    counts = np.maximum(ge - gs - CLIP + 1, 0)
    cum = np.cumsum(counts)
    clips = []
    for r in rng.integers(0, cum[-1], n_gap):
        g = int(np.searchsorted(cum, r, side="right"))
        off = int(r - (cum[g - 1] if g else 0))
        clips.append(dict(kind=1, start=int(gs[g]) + off, gap_start=int(gs[g]), gap_end=int(ge[g])))
    ok = match.graded & match.ball_ok
    s2, e2 = runs(ok)
    keep = (e2 - s2) >= CLIP
    hs, he = s2[keep], e2[keep]
    counts = he - hs - CLIP + 1
    cum = np.cumsum(counts)
    for r in rng.integers(0, cum[-1], n_has):
        g = int(np.searchsorted(cum, r, side="right"))
        off = int(r - (cum[g - 1] if g else 0))
        clips.append(dict(kind=0, start=int(hs[g]) + off, gap_start=-1, gap_end=-1))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolox_s")
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pitch-tiles", action="store_true", help="only tiles touching the pitch")
    ap.add_argument("--gap-clips", type=int, default=60, help="use the first k of the 60 gap clips")
    ap.add_argument("--has-clips", type=int, default=40, help="use the first k of the 40 has clips")
    ap.add_argument("--n-gap", type=int, default=60)
    ap.add_argument("--n-has", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    match = load_match(with_tags=False)
    clips = build_sample(match, args.n_gap, args.n_has)
    clips = clips[:args.gap_clips] + clips[args.n_gap:args.n_gap + args.has_clips]
    Hs = load_homography()[0]
    sess = ballnet.make_session(args.model, args.provider, args.threads)
    all_wins = ballnet.scaled_windows(args.scale)
    cap = cv2.VideoCapture(video_path())

    frames, kinds, clip_id, dets, n_win = [], [], [], [], []
    t_dec = t_det = 0.0
    for ci, c in enumerate(clips):
        t = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["start"])
        t_dec += time.perf_counter() - t
        for k in range(CLIP):
            t = time.perf_counter()
            if k % STRIDE:
                cap.grab()
                t_dec += time.perf_counter() - t
                continue
            ok, im = cap.read()
            t_dec += time.perf_counter() - t
            assert ok
            t = time.perf_counter()
            wins = (ballnet.pitch_windows(all_wins, Hs[c["start"] + k], args.scale)
                    if args.pitch_tiles else all_wins)
            n_win.append(len(wins))
            d = ballnet.detect_scaled(sess, im, args.scale, wins)
            t_det += time.perf_counter() - t
            pad = np.full((5, 5), np.nan, np.float32)
            pad[:len(d)] = d
            frames.append(c["start"] + k); kinds.append(c["kind"]); clip_id.append(ci); dets.append(pad)
        if ci % 10 == 9:
            print("clip", ci + 1, "/", len(clips), "det s/frame {:.3f}".format(t_det / len(frames)),
                  flush=True)
    n = len(frames)
    print("frames {}  decode {:.3f} s/frame (incl. seek)  detect {:.3f} s/frame ({} windows)".format(
        n, t_dec / n, t_det / n, float(np.mean(n_win))))
    outdir = os.path.join(CACHE_DIR, "video_spike")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, args.out + ".npz"), frame=np.array(frames), kind=np.array(kinds),
             clip=np.array(clip_id), det=np.stack(dets),
             clip_gap_start=np.array([c["gap_start"] for c in clips]),
             clip_gap_end=np.array([c["gap_end"] for c in clips]),
             sec_per_frame=t_det / n, windows_per_frame=np.mean(n_win))


if __name__ == "__main__":
    main()
