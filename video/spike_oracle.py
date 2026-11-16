"""Upper bound for crop strategies: crop centred on the ATD ball (oracle, only possible where the
ATD has the ball) and measure how often the detector finds it. Variants: model, crop size/upscale.

    video/.venv/Scripts/python video/spike_oracle.py full_s yolox_s 640   # native 640 crop
    video/.venv/Scripts/python video/spike_oracle.py full_s yolox_s 320   # 320 crop upscaled 2x
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ballnet  # noqa: E402
import spike_eval as E  # noqa: E402
from possession.data import load_homography, load_match, video_path  # noqa: E402

name, model, crop = sys.argv[1], sys.argv[2], int(sys.argv[3])
rad = 50
z = E.load(name)
m = load_match(with_tags=False)
Hs, _, _ = load_homography()
fr = z["frame"][z["kind"] == 0]
sess = ballnet.make_session(model, "cpu", 12)
cap = cv2.VideoCapture(video_path())
res = []
cap_pos = -10
import time
t0 = time.perf_counter()
for f in fr:
    if f != cap_pos + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, im = cap.read()
    cap_pos = int(f)
    bx, by = E.proj(Hs[f], *m.ball[f])
    x0 = int(np.clip(round(bx - crop / 2), 0, 1920 - crop)); y0 = int(np.clip(round(by - crop / 2), 0, 1080 - crop))
    sub = im[y0:y0 + crop, x0:x0 + crop]
    if crop != 640:
        sub = cv2.resize(sub, (640, 640), interpolation=cv2.INTER_LINEAR)
    sc = 640 / crop
    d = ballnet.detect(sess, sub, [(0, 0)], min_score=0.01)
    d = d[d[:, 4] > 0]
    cx = d[:, 0] / sc + x0; cy = d[:, 1] / sc + y0
    dist = np.hypot(cx - bx, cy - by - (-9))      # compensate the ~ -9 px airborne offset
    res.append((dist, d[:, 4]))
print(name, model, "crop", crop, "{:.3f} s/frame".format((time.perf_counter() - t0) / len(fr)))
print(" tau   recall@{r}px  top1-prec@{r}px  (top1 = best-scoring candidate in the crop)".format(r=rad))
for t in (0.05, 0.1, 0.2, 0.3, 0.5):
    rec = np.mean([((d <= rad) & (s >= t)).any() for d, s in res])
    tops = [(d[np.argmax(s)] <= rad) for d, s in res if len(s) and s.max() >= t]
    print(" {:.2f}   {:.3f}        {:.3f}  (n det {})".format(t, rec, np.mean(tops), len(tops)))
