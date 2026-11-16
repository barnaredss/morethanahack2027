"""YOLOX (COCO "sports ball", class 32) ball detector on native-resolution tiles. Video spike.

Weights: Megvii YOLOX-S/M ONNX (Apache-2.0), see video/README.md. No resizing: the ball is ~10 px in
the 1080p tactical frame, so the frame is cut into native 640x640 tiles (or any list of 640x640
windows) and each tile is fed as is (BGR, 0-255 float32, NCHW). Output of the ONNX is the raw
head; the grid/stride decode is done here.

CPU execution with fixed thread count is deterministic for a given onnxruntime build and input.
"""
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
BALL_CLASS = 32
TILE = 640
FRAME_W, FRAME_H = 1920, 1080


def _grids(size=TILE, strides=(8, 16, 32)):
    gs, ss = [], []
    for s in strides:
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        gs.append(np.stack([xv, yv], -1).reshape(-1, 2))
        ss.append(np.full((n * n, 1), s))
    return np.concatenate(gs).astype(np.float32), np.concatenate(ss).astype(np.float32)


GRID, STRIDE = _grids()


def make_session(model="yolox_s", provider="cpu", threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    path = os.path.join(HERE, "weights", model + ".onnx")
    if provider == "cpu":
        prov = ["CPUExecutionProvider"]
    else:
        so.enable_mem_pattern = False       # required by DirectML
        prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=prov)


def full_frame_windows():
    """6 native tiles covering 1920x1080 (small overlap)."""
    return [(x, y) for y in (0, 440) for x in (0, 640, 1280)]


def clamp_window(cx, cy, size=TILE):
    x0 = int(np.clip(round(cx - size / 2), 0, FRAME_W - size))
    y0 = int(np.clip(round(cy - size / 2), 0, FRAME_H - size))
    return x0, y0


def _nms(boxes, scores, iou=0.3):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a + b - inter + 1e-9) < iou]
    return np.array(keep, dtype=np.int64)


def detect(session, frame_bgr, windows, min_score=0.01, max_keep=5):
    """Ball candidates over the given 640x640 windows -> float array [k, 5] (cx, cy, w, h, score)
    in full-frame pixels, sorted by score, NMS across windows."""
    name = session.get_inputs()[0].name
    batch = np.stack([frame_bgr[y:y + TILE, x:x + TILE].transpose(2, 0, 1) for x, y in windows])
    batch = np.ascontiguousarray(batch, dtype=np.float32)
    out = np.concatenate([session.run(None, {name: batch[i:i + 1]})[0] for i in range(len(batch))
                          ]) if len(batch) else np.zeros((0, 8400, 85), np.float32)
    boxes, scores = [], []
    for (x0, y0), o in zip(windows, out):
        sc = o[:, 4] * o[:, 5 + BALL_CLASS]
        sel = sc >= min_score
        if not sel.any():
            continue
        o = o[sel]
        xy = (o[:, :2] + GRID[sel]) * STRIDE[sel]
        wh = np.exp(o[:, 2:4]) * STRIDE[sel]
        c = xy + np.array([x0, y0], np.float32)
        boxes.append(np.concatenate([c - wh / 2, c + wh / 2], 1))
        scores.append(sc[sel])
    if not boxes:
        return np.zeros((0, 5), np.float32)
    boxes = np.concatenate(boxes); scores = np.concatenate(scores)
    keep = _nms(boxes, scores)[:max_keep]
    b = boxes[keep]
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0],
                     b[:, 3] - b[:, 1], scores[keep]], 1).astype(np.float32)


def scaled_windows(scale):
    """640 tiles over the frame upscaled by `scale` (no overlap beyond the last row/col clamp)."""
    W, H = int(FRAME_W * scale), int(FRAME_H * scale)
    xs = sorted(set(list(range(0, W - TILE + 1, TILE)) + [W - TILE]))
    ys = sorted(set(list(range(0, H - TILE + 1, TILE)) + [H - TILE]))
    return [(x, y) for y in ys for x in xs]


def pitch_windows(wins, matrix, scale, margin=60):
    """Keep the windows that intersect the pitch quadrilateral (+ margin px) in scaled pixels."""
    import cv2
    xs = np.array([0.0, 1.0, 1.0, 0.0]); ys = np.array([0.0, 0.0, 1.0, 1.0])
    d = matrix[2, 0] * xs + matrix[2, 1] * ys + matrix[2, 2]
    q = np.stack([(matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]) / d,
                  (matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]) / d], 1) * scale
    c = q.mean(0)
    q = (c + (q - c) * 1.0 + np.sign(q - c) * margin).astype(np.float32)
    hull = cv2.convexHull(q)
    keep = []
    for x, y in wins:
        rect = np.array([[x, y], [x + TILE, y], [x + TILE, y + TILE], [x, y + TILE]], np.float32)
        area, _ = cv2.intersectConvexConvex(hull, rect)
        if area > 0:
            keep.append((x, y))
    return keep


def detect_scaled(session, frame_bgr, scale, windows, **kw):
    """Upscale the frame (INTER_LINEAR), detect on the given windows (scaled coordinates), return
    candidates in ORIGINAL pixel coordinates."""
    import cv2
    im = frame_bgr if scale == 1 else cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_LINEAR)
    d = detect(session, im, windows, **kw)
    d[:, :4] /= scale
    return d
