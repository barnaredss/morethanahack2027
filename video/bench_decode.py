"""Decode-speed benchmark for the tactical video: sequential read vs random seek, and grab-skip.

Run with video/.venv python. Prints numbers only; writes nothing.
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.data import video_path  # noqa: E402


def main():
    path = video_path()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video", path, "frames", n, "fps", cap.get(cv2.CAP_PROP_FPS),
          "size", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = 20000

    # 1. sequential read with decode+retrieve
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    t = time.perf_counter()
    k = 300
    for _ in range(k):
        ok, _ = cap.read()
        assert ok
    dt = time.perf_counter() - t
    print("sequential read: {:.1f} fps".format(k / dt))

    # 2. sequential grab only (decode, no colour convert / copy)
    t = time.perf_counter()
    for _ in range(k):
        cap.grab()
    dt = time.perf_counter() - t
    print("sequential grab: {:.1f} fps".format(k / dt))

    # 3. stride sampling by grab+retrieve every s frames
    for s in (5, 25):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        t = time.perf_counter()
        got = 0
        for i in range(250):
            cap.grab()
            if i % s == 0:
                cap.retrieve()
                got += 1
        dt = time.perf_counter() - t
        print("grab all, retrieve every {}: {:.1f} frames/s decoded, {:.2f} s per retrieved frame"
              .format(s, 250 / dt, dt / got))

    # 4. random seek
    rng = np.random.default_rng(0)
    targets = rng.integers(0, n, 20)
    t = time.perf_counter()
    for f in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, _ = cap.read()
        assert ok
    dt = time.perf_counter() - t
    print("random seek+read: {:.3f} s per frame ({:.1f} fps)".format(dt / 20, 20 / dt))

    # 5. seek accuracy: compare frame at seek vs sequential
    f = 50000
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - 30)
    for _ in range(30):
        cap.read()
    ok, a = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, b = cap.read()
    print("seek exactness: mean abs diff sequential-vs-seek at frame {}: {:.3f}".format(
        f, float(np.abs(a.astype(int) - b.astype(int)).mean())))
    # 6. short-seek: seek, then sequential run of 25
    t = time.perf_counter()
    for f in targets[:5]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        for _ in range(25):
            cap.read()
    dt = time.perf_counter() - t
    print("seek + 25 sequential: {:.2f} s per burst ({:.1f} fps effective)".format(dt / 5, 125 / dt))


if __name__ == "__main__":
    main()
