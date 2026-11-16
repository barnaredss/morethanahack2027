# Video ball-detection feasibility spike

**Recommendation: NO-GO for the 2026-11-17 submission.** Detection does add ball positions in the
ATD gaps, but only after a 2x upscaled, near-full-frame pass (4 h at 5 Hz, 20 h at 25 Hz on this
CPU), at roughly 65 to 70% precision that cannot be verified in the gaps, with no honest way to
measure the effect on spellF1, and with a byte-for-byte reproducibility risk. Details and the
smallest viable design (if the decision is ever reversed) are below.

Everything is self-supervised against the ATD. `evaluation/gt/*` was not used. Scripts:
`video/bench_decode.py`, `spike_run.py`, `spike_eval.py`, `spike_oracle.py`, `ballnet.py`.
Machine: Ryzen AI 5 340 (12 threads), Radeon 840M, shared with other jobs while measuring,
so timings are, if anything, pessimistic.

## What was run

- Detector: YOLOX-S (and -M for one comparison), COCO "sports ball" (class 32), ONNX, Megvii release
  0.1.1rc0, Apache-2.0 (COCO annotations CC BY 4.0). 36 MB / 101 MB. Pretrained, not fine-tuned.
- Stack: onnxruntime-directml 1.24.4 (CPU and DirectML providers), opencv-python-headless 5.0.0.93
  (prebuilt FFmpeg: avcodec 61.19.100), numpy 2.5.4. Pinned in `video/requirements.txt`.
- The ball is about 10 x 10 px in the 1920x1080 frame, so the frame is never downscaled. It is cut
  into native 640x640 tiles (6 per frame) or, for 2x, an upscaled frame cut into 24 tiles.
- Sample (seeded, built only from the unlabelled ATD ball-missing mask and half blocks): 100 clips of
  30 consecutive frames, every 3rd frame detected. 60 clips lie fully **inside graded ATD ball gaps
  of at least 75 frames**, drawn uniformly over gap frames (long gaps weigh more); 40 clips lie where
  the ATD has the ball throughout. 1,000 frames at 1x; a 400-frame subset (25 gap, 15 has-ball clips)
  at 2x.
- "At the ATD ball" = within 50 px of the projected ATD ball. Detections sit a median 9 px above the
  ground-plane projection (ball in the air), compensated. Gap clips are only 25 to 60 clips, so
  shares below carry roughly +/-10 points of sampling error.

## 1. Speed

| item | number |
|---|---|
| decode, sequential `read` | 326 fps (whole match ~10 min); `grab` only 1,095 fps |
| decode, retrieve every 5th frame | ~810 frames/s scanned (whole match ~4 min) |
| decode, random seek | 64 ms/frame, **frame-exact** (seek vs sequential pixels identical). Seek + 25 sequential: 0.12 s |
| detector, YOLOX-S per 640 tile, CPU 1 / 4 / 6 / 12 threads | 0.209 / 0.151 / 0.108 / 0.068 s |
| detector, DirectML (iGPU) per tile | 0.038 s (1.8x faster than 12 CPU threads) |
| YOLOX-M vs S | ~2.9x slower, +3 to 5 points recall: not worth it |
| **1x full frame (6 tiles), measured** | **0.48 s/frame** |
| **2x full frame (24 tiles), measured** | **1.90 s/frame** |
| 2x, tiles near tracked players only (14 of 24 on average; ball inside a selected tile on 99.5% of ATD-has-ball frames) | ~1.1 s/frame (extrapolated, 14 x 0.079) |

Projected runtime, CPU, 12 threads (decode is negligible):

| plan | 64,643 graded ATD-missing frames | whole graded match (158,857) | all 199,319 frames |
|---|---|---|---|
| 1x 6 tiles, every frame | 8.7 h | 21 h | 27 h |
| 2x player-conditioned, every frame (25 Hz) | 20 h | 49 h | 61 h |
| 2x player-conditioned, every 5th frame (5 Hz) | **4.0 h** | 9.7 h | 12 h |
| 2x player-conditioned, every 5th frame, DirectML | ~2 h | ~5 h | ~6 h |

Top-3, top-4, top-6 or top-8 tiles by player count cover the ball on only 32 / 43 / 61 / 73% of
frames, so cropping to the player cluster does not work; the ball is often far from the crowd.
An oracle crop around the ATD ball (impossible in gaps) shows what localisation would buy: native
640 crop 47% recall, 320 crop upscaled 2x 74%, 213 crop 3x 67%. So the gain comes from upscaling,
not from cropping.

## 2. Recall and precision where the ATD has the ball

On the same 150 has-ball frames, ATD ball inside the image on all, tau = detector confidence:

| | recall@50px tau .1 | tau .3 | top-1 precision tau .1 (raw) | candidates >50 px away per frame, tau .3 |
|---|---|---|---|---|
| 1x, 6 tiles | 61% | 48% | 41% | 0.5 |
| 2x, 24 tiles | **74%** | 65% | 42% | 1.5 |
| 2x, ball within 2 m of a player (crowded) | 81% | 69% | 46% | |
| 2x, ball more than 2 m from all players | 67% | 61% | 37% | |

Raw top-1 precision is poor (about 40%): spare balls beyond the touchline, ball boys, boots, line
markings and the penalty spot compete with the real ball. Filtering (on pitch after unprojection,
not in a pitch cell hit in 3 or more different clips, within 10 m of a tracked player; all derived
from ATD players and the homography, nothing from labels) gives, at 2x:

| tau | has-ball frames with a filtered top-1 | top-1 hits the ATD ball (share of all frames) | **precision** | gap frames with a filtered detection | of which continuous with a neighbouring sampled detection (speed under 40 m/s) |
|---|---|---|---|---|---|
| 0.1 | 94% | 59% | 62% | 71% | 47% |
| 0.2 | 89% | 58% | 65% | 56% | 37% |
| 0.3 | 81% | 55% | 68% | 42% | 30% |
| 0.5 | 55% | 46% | 83% | 25% | 18% |

1x filtered is higher precision but far lower yield (tau .2: 56% with a detection, 20% of gap frames,
15% continuous). Caveat that matters: the has-ball frames are the easy ones (the ATD saw the ball
there). Gap frames are by construction the frames the ATD tracker lost, so true recall and precision
in gaps are **probably lower**, and not measurable without labels.

## 3. ATD-missing (gap) frames, 2x, 250 frames in 25 clips

Raw top-1 at tau 0.2: 94% of gap frames get a candidate, but 80% are on the pitch after
unprojection, 50% fall in a **static** pitch cell and only 34% pass all of: on pitch, not static,
continuous with a neighbouring sampled detection. The static cells are the false positives you
would expect: the largest are 1 to 7 m **outside the touchline** (y 69 to 74 m of a 68 m pitch;
spare balls and ball boys; hit in 5 to 10 different clips each) and one on the penalty spot
(about 10 m, 33 m). Distance to the nearest tracked player has median 10.6 m (ATD ball on has-ball frames:
2.0 m, p90 6.2 m), another sign that many raw gap detections are not the match ball.
Per clip at tau 0.2: 19 clips show a moving track (span at least 1 m), 5 are stationary (under 1 m: possibly a dead
ball at a restart, or a static false positive), 1 has fewer than 4 detections. Stationary clips
show that many gaps are real stoppages where a ball is visible but dead; coverage there is not
information for the possession decoder, although a stationary ball on the pitch could in principle
feed `p_dead_a`.

**Expected coverage gain** (ATD currently covers 59.3% of graded frames, 64,643 missing): at 2x with
the filters, tau 0.3, about 30% of missing frames end up with a continuous, filtered detection
(about 19k frames, +12 points of ball coverage to roughly 71%) at an optimistic ~68% precision;
tau 0.5 gives about 18% (+7 points) at ~83%. Excluded from the sample: gaps under 3 s, which the
pipeline already bridges.

## 4. Determinism and reproducibility (rule: same command, same file, byte for byte)

- Decode: OpenCV's bundled FFmpeg returned pixel-identical frames over repeated reads and over
  seek-vs-sequential. Pin opencv 5.0.0.93 (avcodec 61.19.100).
- Detector on CPU: candidate arrays were **bit-identical** for 1 vs 12 threads and across repeats
  (20 real frames, 6 tiles, and random tensors). Thread count does not matter; fix it anyway.
- Not tested: a different CPU microarchitecture (onnxruntime picks different SIMD kernels), which
  can change floating-point results in the last bits and flip a borderline detection. Only one
  machine was available, so bit-identity on the organisers' machine is **unproven**.
- DirectML: repeat runs on this machine are identical, and on 20 frames top-1 positions matched CPU
  to under 2 px, but its output differs numerically from CPU (max 4e-5 on the raw head) and depends on
  driver and GPU. It **would break the byte-for-byte guarantee** across machines; use CPU as the
  reference path and, at most, DirectML as an unverified accelerator.
- Regenerating needs the 5.2 GB video and 4 h (5 Hz) to 20 h (25 Hz) of CPU: organisers would
  re-run `batch_detect` with the pinned venv and compare the sha256 of `cache/video_ball.npz` (and
  of the final CSV) against the hashes in the report. If the hash differs on their CPU there is no
  fallback other than label agreement, which is not the stated rule.

## 5. Validation problem

The Metrica external games have no matching video, so the detector cannot be applied to them and the
benefit to spellF1 cannot be measured there. Without BAR-ATH labels (forbidden) there are only two
honest, weaker tests:
1. **Masking test on BAR-ATH (self-supervised):** hide the ATD ball in stretches where it is present,
   detect on those frames, run the pipeline with and without the video fill, and compare each output
   with the pipeline's own output on the unmasked ATD ball (a pseudo-label, not GT). This measures
   whether video recovers the decisions a full ball would give. Optimistic, because real gaps are
   harder than masked has-ball stretches.
2. **Robustness on external games:** inject detections into Metrica balls with the measured recall,
   precision and static/off-pitch false positives, and check `integrated_v1` does not degrade. This
   tests sensitivity only, not the real benefit.
Neither can show a gain beyond the 0.04 noise band with 25 gap clips of evidence.

## Recommendation: NO-GO

- Yield is modest: about 30% of missing frames (tau 0.3) get a usable detection at about 68%
  precision, measured on frames easier than the gaps; 2x upscaling (4 to 20 h) is required, 1x gives
  half the yield.
- About one detection in three is probably wrong, including balls on the touchline and the
  penalty spot; this injects confident wrong ball positions into `p_home_b` and `p_dead_a`, which
  currently use "ball absent" as evidence (hand-over bridging, dead-ball absence), so the tuned
  logic would need re-tuning that cannot be validated.
- No clean way to show a spellF1 gain, and an unproven byte-for-byte reproduction on another CPU.
- With the deadline 2026-11-17 the time is better spent on the validated pipeline.

Revisit only with labelled-free evidence of benefit (the masking test), and a fine-tuned detector
or a fixed-camera background model, which would raise precision.

## If the decision is reversed: smallest viable design

1. `video/batch_detect.py` (video venv, CPU provider, `intra_op_num_threads` fixed, pinned stack).
   Frame set = graded frames with `ball_ok == False`, every 5th (frame % 5 == 0), derived from the ATD
   at run time (no hard-coded frames). Per frame: decode sequentially in runs (grab-skip between), 2x
   upscale, tiles chosen as those touched by any tracked player's pixel footprint +/-120 px
   (scaled), detect class 32 with `min_score` 0.1, NMS 0.3.
2. Resumable and deterministic: work in shards of 500 frames in ascending frame order, each written
   atomically to `cache/video_ball_shards/<first>.npz` with its input frame list hash; a shard
   whose file validates is skipped on restart; a manifest records versions, thread count and sha256
   of every shard. The final `cache/video_ball.npz` merges shards in frame order after the filters
   (on pitch, static-cell removal from cells hit in at least 3 distinct 30 s windows, within 10 m of
   a player, chain continuity at 40 m/s) and stores `frame`, `x`, `y` (normalised pitch via
   `inv(H[frame])`), `conf`, rounded to fixed precision to reduce last-bit sensitivity.
3. Consumption: a new pipeline file `possession/pipelines/integrated_video.py` registers
   `integrated_video_v1`. It loads `cache/video_ball.npz`, builds a filled ball by interpolating
   between consecutive detections up to a 0.6 s gap, and calls the existing estimators with
   `dataclasses.replace(match, ball=filled)`. Because estimators read only a `Match`, `p_home_b` and
   `p_dead_a` need no code change. Filled frames should keep a low confidence flag; if the
   estimators need to treat them differently, add a `ball_conf` array to `Match` (a shared, additive
   change to the shared contract). `pipelines.DEFAULT` stays `integrated_v1` unless the masking
   test passes.
