"""Render a tactical-video clip with the tracking and a submission's possession labels drawn on it.

    video/.venv/Scripts/python video/render_overlay.py --labels submissions/integrated_v2.csv \
        --at 1:10:00 --seconds 60

Draws every tracked player at their feet (home / away colours), the ATD ball (or its last known
position, faded, while it is not tracked), a banner with the label for that frame, and a +/- 30 s
possession strip along the bottom. For looking at the output, not for choosing anything: it never
reads a ground truth.

Needs the video venv (OpenCV), see video/requirements.txt. The clip goes to MTAH_CACHE_DIR/overlays/
(gitignored): it is a clip of the match video and must never be committed or shared.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import CACHE_DIR, load_homography, load_match, video_path  # noqa: E402

# BGR
COLOURS = {"H": (170, 60, 20), "A": (40, 40, 220), "D": (128, 128, 128), "X": (60, 60, 60)}
BALL = (0, 230, 255)
NAMES = {"H": "HOME (Barcelona)", "A": "AWAY (Athletic)", "D": "DEAD BALL", "X": "not graded"}
STRIP_S = 30


def read_labels(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    return np.array([r[1].strip().upper() for r in rows])


def parse_time(text):
    """'h:mm:ss', 'mm:ss' or a plain frame number -> video frame."""
    if ":" not in text:
        return int(text)
    secs = 0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return int(round(secs * 25))


def project(H, xy):
    """Normalised pitch (n, 2) -> pixels (n, 2); NaN in, NaN out."""
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def clock(match, f):
    for h, first, last in match.half_blocks():
        if first <= f <= last:
            s = (f - first) / match.fps
            return "{} half {:02d}:{:02d}".format("1st" if h == 1 else "2nd", int(s // 60), int(s % 60))
    return "not in play"


def draw_frame(img, f, match, Hs, labels, last_ball):
    H = Hs[f]
    team_col = {0: COLOURS["H"], 1: COLOURS["A"]}
    pts = project(H, match.players[f])
    for slot, (x, y) in enumerate(pts):
        if np.isfinite(x) and np.isfinite(y):
            c = team_col[int(match.slot_team[slot])]
            cv2.ellipse(img, (int(x), int(y)), (14, 6), 0, 0, 360, c, 3, cv2.LINE_AA)
    if match.ball_ok[f]:
        bx, by = project(H, match.ball[f:f + 1])[0]
        last_ball[0] = (bx, by)
        cv2.circle(img, (int(bx), int(by)), 9, BALL, 3, cv2.LINE_AA)
        ball_txt = "ball tracked"
    else:
        if last_ball[0] is not None:
            bx, by = last_ball[0]
            cv2.circle(img, (int(bx), int(by)), 9, (90, 90, 90), 1, cv2.LINE_AA)
        ball_txt = "ball NOT tracked"

    lab = labels[f]
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 70), (25, 25, 25), -1)
    cv2.rectangle(img, (0, 0), (24, 70), COLOURS[lab], -1)
    cv2.putText(img, "Possession: " + NAMES[lab], (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                COLOURS[lab] if lab in "HA" else (230, 230, 230), 3, cv2.LINE_AA)
    info = "frame {}   {}   {}".format(f, clock(match, f), ball_txt)
    cv2.putText(img, info, (w - 760, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2,
                cv2.LINE_AA)

    # possession strip: +/- STRIP_S seconds, current frame in the middle
    n = STRIP_S * match.fps
    y0, hgt = img.shape[0] - 40, 26
    lo, hi = f - n, f + n
    xs = np.linspace(0, w, 2 * n + 1).astype(int)
    for i in range(2 * n):
        g = lo + i
        c = COLOURS[labels[g]] if 0 <= g < len(labels) else (0, 0, 0)
        cv2.rectangle(img, (xs[i], y0), (xs[i + 1], y0 + hgt), c, -1)
    cv2.line(img, (w // 2, y0 - 8), (w // 2, y0 + hgt + 8), (255, 255, 255), 3)
    cv2.putText(img, "-{}s".format(STRIP_S), (8, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "+{}s".format(STRIP_S), (w - 70, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=os.path.join(ROOT, "submissions", "integrated_v2.csv"))
    ap.add_argument("--at", required=True, help="video time h:mm:ss / mm:ss, or a frame number")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--scale", type=float, default=0.5, help="output size relative to 1080p")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    labels = read_labels(args.labels)
    match = load_match(with_tags=False)
    if len(labels) != match.n:
        raise SystemExit("labels have {} rows, expected {}".format(len(labels), match.n))
    Hs, _, _ = load_homography()
    start = parse_time(args.at)
    count = int(round(args.seconds * match.fps))
    end = min(match.n, start + count)

    out = args.out or os.path.join(CACHE_DIR, "overlays", "overlay_{}_{}s.mp4".format(
        start, int(args.seconds)))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cap = cv2.VideoCapture(video_path())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    size = (int(1920 * args.scale), int(1080 * args.scale))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), match.fps, size)
    last_ball = [None]
    for f in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        draw_frame(img, f, match, Hs, labels, last_ball)
        writer.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    writer.release()
    cap.release()
    print("wrote {} ({} frames, {}..{})".format(out, end - start, start, end - 1))


if __name__ == "__main__":
    main()
