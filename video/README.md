# Video ball-detection spike. Own venv; outputs go to cache/, never committed.

Result and recommendation: see `spike_report.md`.

```bash
py -3.12 -m venv video/.venv && video/.venv/Scripts/pip install -r video/requirements.txt
# weights (not committed, video/weights/ is gitignored): YOLOX COCO ONNX, Apache-2.0
#   https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx  (36 MB)
#   https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx  (101 MB)
#   yolox_s sha256 c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
export MTAH_DATA_DIR=... MTAH_CACHE_DIR=...
PY=video/.venv/Scripts/python
$PY video/bench_decode.py                                   # decode speed
$PY video/spike_run.py --threads 12 --out full_s            # 1x, 6 native tiles, 1000 frames
$PY video/spike_run.py --threads 12 --scale 2 --gap-clips 25 --has-clips 15 --out s2x_all
$PY video/spike_eval.py full_s [restrict_to_npz]            # self-supervised metrics vs the ATD
$PY video/spike_oracle.py full_s yolox_s 320                # crop-around-ATD-ball upper bound
```

## Possession overlay (for looking at results)

`render_overlay.py` draws the tracking (players at their feet, ATD ball) and a submission's labels on a
clip of the tactical video, with a +/- 30 s possession strip. It reads no ground truth. Output goes to
`cache/overlays/` (gitignored): a clip of the match video must never be committed or shared.

```bash
video/.venv/Scripts/python video/render_overlay.py --labels submissions/integrated_v2.csv --at 27:27 --seconds 60
```

`--at` is video time (h:mm:ss or mm:ss) or a frame number. Needs the video venv (`video/requirements.txt`).
