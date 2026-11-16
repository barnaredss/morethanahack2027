# Working notes: every setting and where it came from

This is our working log. It is detailed and dense on purpose. For a readable account start with
[TECHNICAL.md](TECHNICAL.md) (what we built and how it did) and [PIPELINE.md](PIPELINE.md) (what a
run does). Come here for the exact value of a setting, the reason for it, or the numbers behind a
decision.

**How to read it.** The sections are in the order the work was done, so later sections sometimes
correct earlier ones. Each setting is listed with its value and its origin: a football or physics
assumption ("prior"), a fit on other matches, or a statistic of the provided tracking measured
without labels.

**Names used below.**

| name | meaning |
|---|---|
| A, A1, A2 | the dead-ball part. A1 is the first, rule-weighted model (`p_dead_a`); A2 is the tree model that replaced it (`p_dead_a2`) |
| B | the hand-written possession rules (`p_home_b`) |
| C | the decoder (`hsmm_decode`) |
| D, extval | the tests on other matches (`extval/`), and the code that makes them look like the provided tracking ("the simulator", "degradation") |
| F | the test of a ball detector on the video |
| `integrated_v1` | rules only: A1 + B + C |
| `integrated_v2` | A2 + a learned team model + C |
| `integrated_v3` | `integrated_v2` plus the control-moment ("event") detector. **This is the submission** |
| `integrated_v4`, `integrated_v4_a2m` | versions retrained with the Bundesliga matches. Not submitted |
| ATD | the provided automatic tracking |
| BAR-ATH | the competition match |
| LOGO | "leave one game out": train on all matches but one, score on that one |
| seeds | different random copies of the same degraded match |
| spellF1, macroF1 | the ranking score and the per-frame tie-break score |

## D: external validation (`extval/`)

Use: `$PY -m extval.harness --pipeline <name> [--games metrica1,metrica2,metrica3] [--seeds 3]
[--param window=75]`; API `from extval.harness import evaluate_pipeline` returns per-game and mean
`spell_f1`, `macro_f1`, `alive_dead_f1`, `n_pred`, `n_gt` and a bootstrap CI. One-time data fetch:
`$PY -m extval.datasets` (git-clones both repos into the gitignored `extval/raw/`). Degraded games
are cached in `cache/extval_deg/` (key = game, seed, params, mtimes of degrade.py, calibration.json,
degrade_fit.json), so sweeps are fast (~10 s/run for 3 games x 3 seeds, no bootstrap).

### Datasets and licences
| set | what | fps | licence |
|---|---|---|---|
| Metrica sample games 1, 2 (CSV) | full-pitch optical tracking + events, 28/26 slots | 25 | no LICENSE file in repo; README: be responsible, acknowledge source when public |
| Metrica sample game 3 (FIFA EPTS + JSON) | same, 35 slots, CARRY events | 25 | same |
| SkillCorner open data (20 A-League matches) | BROADCAST tracking (~9 players visible), phases of play | 10 -> 25 (linear resample) | MIT |

Default harness set = Metrica 1-3 (full-pitch, like the tactical ATD). SkillCorner (`--games all` or
`sc<id>`) is a secondary, harder-to-compare domain: its dropouts are already natural, so degrade.py
leaves them alone (only tags are synthesised). `baseline_v0` scores ~0.37 spellF1 / 0.75 macroF1 on
two SC games, i.e. easier than BAR-ATH; do not select on SC alone.
Metrica ball is NaN for ~40% of frames natively (mostly while dead), the same structure as BAR-ATH.

### Label rules (`extval/labels.py`, all frames outside the graded halves are X)
Graded halves = first kick-off event of the half to the last tracked frame of the period (BAR-ATH
blocks also run kick-off to whistle). Params: `min_control=25` frames (football reasoning:
the opponent's control event must come >= 1 s later, else it is a touch/deflection (CHALLENGE.md "what
does not count"; a pass needs ~1 s to arrive); CHALLENGE.md also says "a few seconds", so 50 frames is the
strict reading, see "Ship: compliance audit"; sensitivity: 0/12/25/50 frames give 285/279/264/225
possessions on game 1),
`max_foul_dead=250` (a foul/offside only stops play if a free kick follows within 10 s),
`max_unlogged=375` (restart without a logged ball-out: stoppage starts when the previous event
ends, at most 15 s earlier), `max_loose=15` (SkillCorner: gap between phases shorter than 0.6 s is a
loose ball, longer is D). Rules R1-R5 are in the module docstring. Sanity vs football norms (Metrica
1/2/3): 285/275/316 possessions, mean 13.5/14.1/12.7 s, dead 33/32/30%, home share 0.52-0.54, 87-97
dead stretches per game, median 13-15 s. No BAR-ATH GT was used to choose these.

### Degradation model (`extval/degrade.py`, calibration in `extval/calibration.json`)
`extval/calibrate.py` reads ONLY the unlabelled BAR-ATH ATD (graded half blocks) and writes the JSON.
| item | value | origin |
|---|---|---|
| ball visibility target | 0.593 | published (DATA.md) |
| native ball gaps | kept as is | Metrica already has 155-168 gaps/game vs BAR-ATH 150, similar zone hazards |
| extra ball dropouts | per-frame hazard by pitch zone x local density (8 m), bisected scale to hit 0.593 (0 when already below), gap lengths from the BAR-ATH zone histogram | ATD stats |
| ball white jitter (v1, REMOVED in v2) | sigma 3 mm/axis (solved from 2nd-difference median 9.2 mm vs ~1 mm) | ATD stats |
| ball position error (v1, REPLACED in v2, see the correction below) | AR(1) core 0.6 m (tau 50 f) + bursts 20% of time, std 8 m (mean 75 f) | fit (`python -m extval.degrade`) of nearest-object-distance quantiles [.1,.25,.5,.75,.9,.99] = [0.44,0.89,1.91,4.06,7.02,16.35] m, log-MSE, `degrade_fit.json` |
| slots | 46 (23/team); tracklet mean 2670 f, 0.52 births/player/min | ATD stats |
| player loss | per-frame hazard by crowding (neighbours < 5 m): [1.4e-4, 3.3e-4, 1.1e-3]; lost for log-normal(sigma 1.2) frames with mean from objects_mean 20.49; re-acquired in a NEW slot | ATD stats |
| duplicates | short objects (mean 60 f, 0.6 m offset), rate grid-tuned so frames with >22 objects ~2.4% | ATD stats |
| player jitter | none (ATD tracks are smoother than Metrica's: 1.4 mm vs 2.3-2.9 mm) | ATD stats |
| tags | misses 8%, spurious none, team flip 8%, timing jitter sigma 25 f, set piece interval [restart-50, restart+25] f, throw-ins team N/A; codes BUILD UP / PROGRESSION / CREATING CHANCES / DEFENDING IN * THIRD by ball third, SET PIECES typed by ball position, GOALS | ASSUMPTIONS, nothing unlabelled to fit them; `tag_noise=` scales all |
Not simulated: identity swaps between teammates and across teams, ball z, set-piece tags that
disagree with the real Smart Tagging timing convention (unknown), referees. Result of degradation on
the 3 games: ball visible 59.2-59.4%, mean objects 20.7-20.9 (BAR-ATH 20.49), frames >22 objects
1.9-2.2% (2.4%), births ~1000/game (~1220).

### Calibration check (the key result; uses only published baseline scores)
First version (without ball position error): baseline_v0 0.361 spellF1 / 0.699 macroF1, far too easy.
The unlabelled statistic that exposed it was the ball-to-nearest-object distance (BAR-ATH median 1.9 m,
Metrica 0.3-1.4 m because Metrica's ball is snapped to the player); it was then fit, NOT the baseline
score. External mean over Metrica 1-3, 3 seeds (spellF1 CI [0.275, 0.291], macro [0.629, 0.711]):

| baseline window | external spell / macro | published spell / macro |
|---|---|---|
| 0 (raw) | 0.0377 / 0.6705 | 0.0486 / 0.6532 |
| 25 | 0.2216 / 0.6786 | 0.1996 / 0.6752 |
| 50 (`baseline_v0`, floor) | **0.2820 / 0.6726** | **0.2433 / 0.6724** |
| 125 | 0.2520 / 0.6214 | 0.2242 / 0.6356 |

The simulator reproduces the published ordering (spellF1 picks 50, macroF1 picks 25), macroF1 within
0.02 everywhere and spellF1 about +0.03 optimistic (inside the 0.04 noise band; predicted possession
counts 630 vs 292 true per game, i.e. the same over-fragmentation). Treat absolute external spellF1 as
~0.03 too high, and compare pipelines only with each other on the same seeds.

### Correction 2026-11-09: the ball position error was too rough (simulator v2)

**What was wrong.** The v1 ball position error (AR(1) core + bursts, `ar1_error`) matched the
nearest-object distance profile but was a jittery signal: per-frame innovation `sigma*sqrt(1-rho^2)`
is ~1.3 m in a burst. Measured ball steps on graded frames (unlabelled statistics only; steps =
frame-to-frame ball displacement in metres, reversal = share of consecutive step pairs with a negative
dot product; `calibrate.ball_step_stats`, saved in `calibration.json`):

| ball track | step p50 / p90 / p99 (m) | share > 0.5 m | share > 1.5 m | reversal |
|---|---|---|---|---|
| BAR-ATH real | 0.203 / 0.572 / 0.984 | 0.150 | 0.0028 | 0.0013 |
| Metrica clean (1, 3) | 0.212 / 0.620 / 0.946, 0.260 / 0.610 / 0.832 | 0.196, 0.195 | 0.0022, 0.0017 | 0.007, 0.008 |
| v1 degraded (1, 3) | 0.350 / 1.551 / 3.598, 0.384 / 1.690 / 3.637 | 0.36, 0.38 | 0.104, 0.117 | 0.247, 0.243 |
| **v2 degraded** (1, 2, 3) | 0.240 / 0.635 / 0.989, 0.265 / 0.623 / 0.991, 0.280 / 0.614 / 0.896 | 0.209, 0.207, 0.203 | 0.0026, 0.0025, 0.0018 | 0.0081, 0.0086, 0.0084 |

v1 also added 3 mm white ball jitter (calibrated to the 2nd-difference median); it alone puts the
reversal share at 0.020 and is contradicted by the step statistics, so it is OFF in v2.

**What changed (all calibrated from unlabelled BAR-ATH statistics, no score used).**
1. `smooth_error`: Gaussian low-passed noise, core (0.6 m std, kernel 90 frames) + bursts (11 m std,
   kernel 40 frames) switched in by a smooth sigmoid envelope that is on 10% of the time. Fitted by grid
   search (`python -m extval.fit_ball`, `degrade_fit.json`) to BOTH the nearest-object distance
   quantiles AND the six step statistics. Result: distance log-MSE 0.0083 (no error: 1.516) with step
   loss 0.356 (no error: 0.344), i.e. **both targets are met at once**: the error costs nothing on the
   step statistics. The error is needed: without it the nearest-object distance is far too small (p10
   0.05 m vs 0.44), and the data cannot tell a position error from a ball genuinely off the player's feet
   or in the air. The remaining step mismatch (p50 0.24-0.28 vs 0.20, share > 0.5 m 0.21 vs 0.15, reversal
   0.008 vs 0.0013) is already in the clean Metrica ball (native, not caused by the error); it is not
   corrected. v1 parameters are kept in `degrade_fit_v1.json`.
2. Player check (step statistics of slots, `calibrate.player_step_stats`): bulk matched (p50 0.055 vs
   0.054 m, p99 0.26 vs 0.26) but v1 duplicates redrew a random 0.6 m offset EVERY frame: p99.9 step 1.2 m
   vs 0.35, share > 1.5 m 6e-4 vs 4e-5, reversal 0.006 vs 0.0004, and "identity jumps" 1.1/min vs 0.
   Fixed (one fixed offset per duplicate): p99.9 0.32, share > 1.5 m 5e-6, reversal 0.004.
3. Slot behaviour: BAR-ATH has 21.8 objects outside outages and ~50 whole-tracking OUTAGE episodes
   (< 17 objects visible: 7.7% of graded frames, median 2.3 s, 90th percentile 28 s, mean 10.8 objects,
   ball missing in 89%). v1 spread the shortfall over independent player losses (7%), so its object-count
   histogram was wrong (22 objects on 33% of frames vs 49.5%). v2 adds the outages (starts at the BAR-ATH
   rate, drawn from the clean-ball-missing frames with probability 0.89, lengths stratified from the
   BAR-ATH histogram) and sizes independent losses to the non-outage shortfall. Result: 22 objects on
   50% of frames, outage share 3-12% per game (BAR 7.7%), mean objects 20.1-21.3 (20.49), ball visibility
   59.0-59.6%.
4. No sign of cross-team label swaps in the unlabelled statistics (mean team std x/y 14.9/11.9 m vs
   14.6-16.2/11.7-12.9 m in Metrica; nearest object is the other team 0.674 vs 0.659-0.743), at low power.
   Identity swaps between teammates remain unmodelled.

**Calibration check after the fix** (`python -m extval.calcheck`, baseline_v0, Metrica 1-3, seeds 0-2),
external vs published spellF1 / macroF1:

| window | v1 (old) | v2 (new) | published |
|---|---|---|---|
| 0 | 0.0377 / 0.6705 | 0.0646 / 0.6678 | 0.0486 / 0.6532 |
| 25 | 0.2216 / 0.6786 | 0.2314 / 0.6737 | 0.1996 / 0.6752 |
| 50 | 0.2820 / 0.6726 | 0.2880 / 0.6670 | 0.2433 / 0.6724 |
| 125 | 0.2520 / 0.6214 | 0.2592 / 0.6160 | 0.2242 / 0.6356 |

Overshoot is about what it was in v1, not worse: spellF1 is +0.016 to +0.045
above the published floors (noise band 0.04, window 50 is at its edge), macroF1 within 0.02, ordering
reproduced (spellF1 picks 50, macroF1 picks 25). The fix hardly moves the score because the nearest-team
baseline reacts to the size of the ball error, not to its roughness. The ~+0.035 spellF1 optimism has no
identified cause among the unlabelled statistics checked (ball and player steps, object counts, outages, slot
births, team shape); it could also come from the label rule (`min_control`) or from identity swaps. Not
tuned away.

**Old vs new simulator, Metrica 1-3 x seeds 0-2** (spellF1 / macroF1 / aliveF1, n_pred of 292 true):

| pipeline | v1 | v2 |
|---|---|---|
| baseline_v0 | 0.2820 / 0.6726 / 0.838, 630 | 0.2880 / 0.6670 / 0.838, 587 |
| emissions_b_v1 | 0.5315 / 0.7534 / 0.856, 220 | 0.5189 / 0.7494 / 0.856, 227 |
| deadball_a_v1 | 0.2925 / 0.7229 / 0.902, 595 | 0.2813 / 0.7199 / 0.899, 602 |
| integrated_v1 | 0.5288 / 0.7870 / 0.910, 214 | 0.5261 / 0.7856 / 0.909, 221 |

All changes are inside the noise band: the shipped pipelines were not tuned to the rough ball in any way
that matters. Stale fitted weights: `scripts/tune_integrated.py fit` rebuilt on v2 gives held-out LOGO 0.525 /
0.781 / 0.903 vs shipped weights 0.526 / 0.786 / 0.909 (in-sample 0.531 / 0.786 / 0.906): the A and B weights shift
(B SHAPE_W all-game fit [0.010, 0.456, 0.017, 0.152, -1.64] vs shipped [0.031, 0.321, 0.057, 0.184, -1.34]) but
the score difference is 0.001-0.005, i.e. not stale in effect. Shipped values unchanged; the BAR-ATH
`integrated_v1` sha256 is still f16a02b6...1bc6. Warning: `scripts/tune_integrated.py` keys its
`tune_items_*.pkl` cache only on the mtimes of deadball.py / emissions.py, NOT on the simulator, so it silently
reuses stale items after any degrade.py change (the old 1.6 GB file was renamed `*.oldsim.pkl`, the canonical
name now holds v2 items).

**Gap measurement** (`scripts/gaps_measure.py`, `gaps_measure2.py`, oracle dead ball, B emissions, 9 runs;
`DEGRADE_PARAMS='<json>'` selects the simulator; v1 = `degrade.LEGACY_V1`):

| | v1 | v2 |
|---|---|---|
| hand-overs inside ball-missing gaps | 11 of 199 (5%) | 11 of 199 (5%) |
| median length of those gaps | 671 f | 671 f |
| baseline spellF1 (oracle dead) | 0.624 | 0.611 |
| upper bound, perfect hand-overs in gaps | 0.648 | 0.633 |
| upper bound, perfect all missing frames | 0.687 | 0.669 |
| upper bound, perfect visible frames | 0.903 | 0.905 |
| upper bound, perfect everywhere | 0.962 | 0.962 |
| error rate alive frames: visible / missing-with-hand-over / missing-without | 0.117 / 0.336 / 0.283 | 0.126 / 0.323 / 0.294 |

Conclusion unchanged: the loss is on VISIBLE frames (perfect visible 0.90 vs perfect missing 0.67-0.69), not
in gaps.

## C: temporal decoder (`possession/decode.py`, pipeline `decode_c_v1`)

**Model.** Per half, explicit-duration HMM (HSMM) over {H, A, D}, exact Viterbi in numba (segment
start/end DP, duration window `dmax`), then a boundary-refinement pass. X outside the halves.
- Emission: per frame, log of the 3-way posterior ((1-pd)·ph, (1-pd)·(1-ph), pd) divided by the prior,
  scaled by `evidence`. NaN p_dead -> pd := pd0, NaN p_home -> 0.5, which gives exactly 0 for every
  state (uninformative, not 0.5 evidence). Probabilities are clipped to [eps, 1-eps].
- Durations: log-normal, one for alive (H and A), one for D, with hard minimums. Same-state
  continuation is allowed at `cont_cost` so segments longer than `dmax` still work.
- Transitions: alive -> other team (`p_turn`, minus `switch_penalty` nats) or -> D; D -> H/A 50/50.
  Which team restarts is decided by the emissions.
- Refinement: each boundary moves within +/- `refine_window_s` to the maximum-likelihood split
  (sharper, unscaled evidence); an exactly flat plateau (NaN gap) takes its midpoint.
- Optional `change_points=(frame, team, conf)` rows (from B's `control_events`): bonus of
  `cp_weight*conf` nats for a segment of that team starting near the frame (Gaussian, `cp_sigma_s`).

**Parameters** (all in `HSMM_DEFAULTS`; override via `hsmm_decode(..., **params)`). "Syn" = chosen
by coordinate search on synthetic emissions (dev seeds 0-5, objective = mean spellF1 over noise
0.5/1/1.5 on both the matched and a mismatched-duration generator); "prior" = football reasoning.
No BAR-ATH labels were used.

| param | value | origin |
|---|---|---|
| evidence | 0.10 | syn: 0.1 beat 0.35 by +0.05; frames are strongly autocorrelated. Re-tune on extval, it depends on estimator calibration |
| eps | 0.02 | syn: flat. Caps one frame's influence |
| pd0 | 0.40 | prior: ball in play ~55-60 % of match time. syn preferred 0.5 (+0.013); kept the prior |
| alive_median_s / alive_sigma | 8 s / 1.0 | prior: mean possession ~13 s (the organisers' summary says 14 s); syn flat |
| dead_median_s / dead_sigma | 13 s / 0.7 | extval label statistics (dead stretch median 13.4-15 s, 5th-95th pct 4.3-62 s, so sigma ~0.7). Was 9 s (prior "5-20 s" taken from the public GT); sweep 6/9/13 identical |
| dmin_alive_s | 1.0 | extval label statistics (shortest alive spell 0.6-1.0 s, 5th pct 1.4 s; equals the 1 s control rule). Was 0.6 (justified by the public GT's shortest spell); sweep 0.6-1.5 identical. syn prefers 1.5 but that would delete real short spells |
| dmin_dead_s | 3.0 | prior: quickest restart ~3 s; syn preferred 4 (generator min is 5 s, so not trusted) |
| dmax_s | 80 | runtime/memory only |
| p_turn | 0.5 | prior |
| switch_penalty | 1.0 | syn: 0-1 best, larger hurts |
| cont_cost | 6.0 | only relevant past dmax |
| refine_window_s / refine_eps | 4.0 / 0.02 | syn: +0.005 over no refinement (windows 1-8 s similar) |
| cp_weight / cp_sigma_s | 3.0 / 0.4 | syn demo only (simulated change points, +0.02 to +0.04) |

**Synthetic study** (`scripts/decode_c_study.py report`, held-out seeds 100-105, 2x20k-frame halves,
~41 % NaN in p_home; `possession/decode_synth.py`: AR(1) logit noise, burst errors, NaN gaps). spellF1:

| noise | HSMM | majority_decode(50) | HSMM, mismatched truth gen | majority, mismatched |
|---|---|---|---|---|
| 0.0 | 0.773 | 0.557 | 0.712 | 0.454 |
| 0.5 | 0.692 | 0.344 | 0.676 | 0.398 |
| 1.0 | 0.625 | 0.238 | 0.553 | 0.215 |
| 1.5 | 0.366 | 0.128 | 0.341 | 0.113 |
| 2.0 | 0.239 | 0.081 | 0.253 | 0.085 |
| 3.0 | 0.153 | 0.056 | 0.164 | 0.073 |

Even with noise 0 the ceiling is ~0.77 because a boundary inside a NaN gap cannot be located. That
is the case for good ball-gap bridging in B. (Numbers were measured before the final
refine-window 2 -> 4 s change, which adds about +0.005.)

**Runtime.** 161k synthetic frames in ~1.1 s; the full BAR-ATH `decode_c_v1` run is ~2.3 s wall
including data load. Byte-identical across runs (checked sha256).

**Caveat.** `decode_c_v1` uses the placeholder estimators (missing ball = dead), so its public-window
sanity score is not meaningful and was not used for any choice. `scripts/sanity_public.py` was run
once to confirm alignment only.

## A: dead-ball detector (`possession/deadball.py`, pipelines `deadball_a_v1`, `deadball_a_v1_majority`)

**Model.** `p_dead_a(match)` -> P(dead) per graded frame (NaN outside the halves). Per frame, a 12-feature
vector (`frame_features`) goes through a logistic model (`DEFAULT_WEIGHTS`), then `clean_stretches`
closes alive holes < 1.5 s and drops dead islands < 1.5 s. Features, all computed from the `Match` only:

| feature | meaning (football / physics reasoning) |
|---|---|
| `missing` | ball untracked this frame |
| `gap_len`, `gap_short` | log(1+gap seconds) and exp(-gap/2 s): a 1 s dropout is tracking noise, a 10 s absence is a stoppage |
| `gap_edge_last`, `gap_edge_next` | exp(-d/6 m), d = distance of the last / next observed ball position from the pitch outline (0 inside = on a line; negative d counts as 0): ball left over a line, restart from a line |
| `gap_tag` | a SET PIECES / GOALS tag overlaps the gap +/- 10 s. Overlap only, not tag timing: the real Smart Tagging window convention (35 s windows starting ~4-7 s before the restart) differs from the extval simulator's (restart-2 s .. +1 s), so edges are never snapped to tags |
| `gap_slow` | 1 - (mean team speed in the gap / local norm), clipped to [-1, 1]. Whistle: everyone slows. Norm = rolling 60 s median of the team median speed (physics: absolute pace varies by match) |
| `oob` | ball observed > 0.5 m outside the outline for >= 5 frames (0.2 s) |
| `slow` | same speed ratio, per frame |
| `still` | observed ball moved < 1.5 m over +/- 1 s: waiting for a free kick / corner / throw-in |
| `pre_kickoff` | frames of a half before its detected kickoff |

Team speed = median over tracked players of `match.speed`, falling back to position differences for the
~10 % empty ATD speeds, smoothed 1 s.

**Kickoff detection** (`detect_kickoffs`, no frame numbers): sustained play = first frame whose next
10 s of team speed average >= 1.25 x the half's median speed; the kickoff is the last frame of the
preceding lull (within 8 s) still < 0.5 m/s above that lull, provided at the lull both teams are >= 65 %
in their own halves (the kicking team has a few over the line). A half that is already "hot" on its first
frame starts at its first frame; a kickoff less than 15 s after the first frame is treated as no
pre-kickoff period (indistinguishable from a lull). The parameters were re-derived on 2026-11 from
external games and priors only (previously `own_half_min` 0.65 and `min_pre_s` 15 s, "adjusted" with
BAR-ATH half 2 in view; that was dropped as non-defensible). Check: `scripts/kickoff_extval.py` (Metrica
1-3: real tracking from the period start, plus seeded synthetic 30-120 s walking spells before the
kickoff). Result with the shipped values: 10 of 12 synthetic cases within 1.3 s of the truth, median error
0.6 s; failures are Metrica 2 half 2 (kickoff followed by a slow 18 s, detected late) and, on real data,
Metrica 1 half 2, whose 35 s pre-kickoff is not detected. Run on BAR-ATH once, not adjusted afterwards:
half 1 = its first frame (no pre-kickoff detected), half 2 = frame 113700 (the half is graded from
110808, so ~2900 frames before it are D); identical with the old and the new values.

**Parameters.**

| param | value | origin |
|---|---|---|
| logistic weights [bias, missing, gap_len, gap_short, edge_last, edge_next, gap_tag, gap_slow, oob, slow, still, pre_kickoff] | [-2.8, 3.9, -1.0, -6.0, 2.6, 0.0, 1.2, 2.7, 1.4, 1.4, 1.4, 5.0] | IRLS, L2=1, on extval Metrica 1-3 x seeds 0-2 (degraded) frame labels D vs not. Leave-one-game-out frame F1 0.75/0.81/0.80 vs 0.71/0.78/0.75 for "missing = dead". `gap_edge_next` fitted ~0 and is kept at 0. `pre_kickoff` is a prior (extval halves start at kickoff, nothing to fit) |
| edge_scale_m | 6 | prior (throw-in / corner flag proximity); sweep 3 and 10: all within noise (0.287/0.287 vs 0.288 spellF1, alive_dead .908/.904 vs .907) |
| short_gap_s | 2 | prior; sweep 1 / 4: within noise |
| oob_margin_m, oob min frames | 0.5 m, 5 | prior (positional noise of the ball ~0.5 m) |
| slow_win_s | 60 | prior; 30 within noise |
| still_win_s, still_thr_m | 1 s, 1.5 m | prior; still_win 2 identical |
| clean close/min dead | 1.5 s / 1.5 s | prior (restart takes >= 3 s); on/off differ by 0.0004 spellF1 |
| play_ratio, play_window_s, backtrack_s, ramp_rise, search_s | 1.25, 10 s, 8 s, 0.5 m/s, 600 s | football priors (sustained play = 10 s ~ one possession, a bit busier than the half's median; the ramp starts within 8 s; tracking starts at most 10 min before kickoff). Flat on the external check (`scripts/kickoff_extval.py`): ratio 1.1-1.5, window 7-15 s, ramp 0.3-0.8 all give the same errors; backtrack 5 s loses one case |
| own_half_min | 0.60 | extval: own-half fraction of a team at the logged kickoff is 0.64-1.0 over the 6 Metrica halves, so 0.65 rejected real kickoffs; 0.7 and above fail 2 of 12 synthetic tests, 0.5-0.6 do not |
| min_pre_s | 10 s | = play_window_s: a kickoff closer than one window to the first frame cannot be told from a lull (was 15 s); metrica3 half 1 has 14 s of real pre-kickoff |

**Extval result** (mean Metrica 1-3, 3 seeds; weights were fitted on these same games, so in-sample):

| pipeline | spellF1 | macroF1 | alive_dead_F1 |
|---|---|---|---|
| baseline_v0 | 0.282 | 0.673 | 0.838 |
| decode_c_v1 (hsmm + "missing = dead") | 0.280 | 0.697 | 0.856 |
| deadball_a_v1 (hsmm + p_dead_a) | **0.288** [0.280, 0.297] | **0.725** | **0.906** |
| deadball_a_v1_majority (majority_decode + p_dead_a) | 0.283 | 0.673 | 0.839 |

**Known failure modes.**
- Alive gaps: ~36 % of missing-ball frames in extval are alive (dropouts); only their length, location
  relative to the lines and the team speed separate them from dead gaps (length-weighted gap accuracy
  0.85 vs 0.80 for "all gaps dead"). Long mid-pitch dropouts with normal player speed are the main false
  positives.
- Onsets are late: median +23 frames (0.9 s) on extval; label alignment there is itself fuzzy (the
  extval truth starts a stoppage when the previous event ends, often before the ball crosses the line,
  and 375-frame "unlogged" stoppages start with no physical trigger). Offsets (restart touch) are
  exact for 85 % of stretches within 8 frames since the ball reappears at the restart.
- Dead frames with the ball observed and moving (ball retrieval, fouls) are largely missed (21 % of dead).
- Kickoff detection can fire a little late on a half that starts with a long jog (Metrica 2 half 2:
  17 s of the opening labelled dead). On BAR-ATH half 1 and 2 look right.
- No restart-geometry features beyond "ball near a line" (no corner / 6-yard box / wall detection):
  ball position is noisy and a corner flag or box check added nothing measurable over line proximity.
- Predicted dead share on BAR-ATH graded frames 22 % (ball untracked 41 %), 90 stretches, median 11 s.

**Alive/dead F1 on the public window (sanity only, not used for any choice):** deadball_a_v1 0.916
(baseline_v0 0.899); its spellF1 there is lower (0.226 vs 0.326), driven by the placeholder possession
emissions over-fragmenting (38 predicted possessions vs 17), not by dead-ball timing.

## B: possession emissions (`possession/emissions.py`, pipelines `emissions_b_v1`, `emissions_b_v1_cp`)

**Estimator** `p_home_b(match, **params)` -> P(home | alive), never NaN inside the halves.
`control_events(match)` -> rows (frame, team, conf) of controller changes (candidate change points).
A "last controller" chain (possession = team that last had CONTROL; touches, pass-bys and contested
balls change nothing), fused in log-odds with weak independent evidence:
1. **Control evidence.** Per frame and team: nearest player within `r_control`, opponent at least
   `margin` farther, ball-player relative speed < `v_rel` (co-movement). Contested frames give none.
2. **Control episodes.** Runs of evidence (blips <= `join_frames` bridged) lasting >= `min_ctrl_s`.
   Shorter runs are touches/deflections and are ignored.
3. **Chain confidence** from a two-state Markov model of flips, mean spell `T_poss_s`:
   `p(d)=0.5+0.5 exp(-2d/T)`. Loose ball visible: previous controller, conf `p(since control)`.
   Ball missing, same team both sides: bridge `p(a)p(b)/p(a+b)`. Different teams: smooth two-sided
   posterior `p(a)(1-p(b))/(1-p(a+b))` (hand-over somewhere in the gap). Gaps <= `interp_max_s`
   are linearly interpolated and the hand-over is placed where the interpolated ball first reaches
   the new team (`_crossing`).
4. **Soft nearest-team** `tanh((d_away-d_home)/near_scale)`, off when the ball is in flight.
5. **Team shape** (ball-missing fallback, helps everywhere): home-attack-frame block height `S`,
   its forward velocity `dS` (+/-4 s), length-spread difference `comp`, width difference, mean speed
   difference, all smoothed. Only antisymmetric features and no intercept, so there is no home bias.
6. **Smart Tagging**: BUILD UP / PROGRESSION / CREATING CHANCES / ATTACKING TRANSITION / BALL IN
   FINAL THIRD / BALL IN THE BOX -> tag team; DEFENDING IN * / DEFENSIVE TRANSITION -> other team.
   Real tags come in mirrored pairs (one per team), averaged to a single +-1.

| param | value | origin |
|---|---|---|
| interp_max_s | 3.0 | prior: ball travels ~5-17 m in 3 s; linear interpolation only locates hand-overs. Median real gap is 10.7 s (unlabelled stats), so most gaps are NOT interpolated |
| r_control / margin / v_rel | 3.0 m / 1.0 m / 4.0 m/s | physics prior (ball error median 1.9 m in ATD). Extval sweep: all within noise (r 2-4, margin 0.5-2, v_rel 2.5-8 spell 0.59-0.63); kept the priors |
| min_ctrl_s / join_frames | 0.4 s / 5 | football prior: 10 frames, below the 1 s extval control rule because control evidence is noisy; extval 0.2/0.8 both within noise (was justified by "the public GT has sub-second spells", dropped) |
| T_poss_s | 13 s | prior: mean possession 13-14 s (organisers' summary, extval labels); 8 and 25 within noise |
| c_in / c_cap | 0.92 / 0.97 | prior; 0.8-0.97 within noise |
| w_near / near_scale / near_vmax | 0.6 / 3 m / 15 m/s | prior; w_near 0.3-1.2 within noise |
| SHAPE_W [S,dS,comp,width,speed] | [0.031, 0.321, 0.057, 0.184, -1.340] per raw unit | EXTVAL FIT: ridge logistic (lambda 1, standardised) on Metrica 1-3 x seeds 0-2 degraded games, offset = chain+near logit, `scripts/emissions_b_fit.py`. Leave-one-game-out NLL improves on all three games (0.417->0.375, 0.380->0.332, 0.420->0.348); accuracy on ball-missing frames +-0.01 to +0.03. Speed carries the most weight (the pressing/chasing team runs more); simulator-vs-BAR-ATH risk, see below |
| w_shape / shape_win_s | 1.0 / 4 s | fit scale / prior |
| w_tag | 1.0 | unlabelled BAR-ATH statistic: tag-implied team agrees with the control chain on 76% of in-control frames (`scripts/emissions_b_tagagree.py`; logit 0.76 = 1.15, shaded down because chain errors are in the 76%). Extval sim tags (8% flips) are cleaner, so extval cannot set this; w 0/1/2 gave spell 0.618/0.624/0.586 |

**Extval results** (harness, Metrica 1-3 x 3 seeds, placeholder dead-ball = missing ball):

| pipeline | spellF1 | macroF1 | aliveF1 | n_pred (true 292) |
|---|---|---|---|---|
| baseline_v0 | 0.282 | 0.673 | 0.838 | 630 |
| decode_c_v1 | 0.280 | 0.697 | 0.856 | 631 |
| **emissions_b_v1** | **0.531** | 0.754 | 0.857 | 221 |
| emissions_b_v1_cp | 0.529 | 0.753 | 0.856 | 226 |

With an ORACLE dead-ball signal (isolates B; `scripts/emissions_b_dev.py`): team accuracy on alive
frames 0.883 ball-visible / 0.695 ball-missing, spellF1 0.624, macroF1 0.899 (nearest-team placeholder
with the same oracle: 0.74 visible, no opinion on missing, spellF1 0.338). Ablations (oracle dead):
no shape 0.561 -> shape 0.618 (+0.057); tags 0 -> 1.0: +0.006; change points: no gain (0.622 vs
0.624) because the chain already encodes the hand-overs, but `control_events` is available.
Decoder `evidence` 0.1/0.2/0.4 and `switch_penalty` 0 -> flat (<0.006), so C's defaults stand.

**BAR-ATH sanity (informational, NOT used for any choice).** Public window: spellF1 0.459, macroF1
0.662, 17 predicted vs 17 true possessions; baseline_v0 0.326, decode_c_v1 0.243. Team accuracy on
alive frames: ball-visible 0.878 (nearest-team 0.778), ball-missing 1.000 on just 391 frames.

**Failure modes / caveats.** (1) Under the placeholder dead-ball (missing = dead) the long ball gaps
become D, so bridging value is capped until A lands; with A's p_dead, re-check `evidence`.
(2) Ball-missing accuracy 0.70: a long gap with a hand-over inside it is a coin flip unless shape
agrees. (3) Shape weights are fit on Metrica-derived degraded games; the sim reproduces BAR-ATH's
dropout/noise statistics but not its real team behaviour or ATD speed convention (ATD speed is NaN
for GK slots); if BAR-ATH disagrees, set `w_shape` lower. (4) Slot ID swaps between teams are not
simulated, they would fake control episodes. (5) Real control may be shorter than extval's 1 s
rule; `min_ctrl_s` is a prior, not tuned on BAR-ATH GT. (6) `control_events` change points did
not help with the HSMM defaults (cp_weight 3 / sigma 0.4 s); C may retune.

## Integration (`possession/pipelines/integrated.py`, pipeline `integrated_v1`; the default until `integrated_v2` replaced it)

`hsmm_decode(match, p_dead_a(match), p_home_b(match), **params)`. Tuning script:
`scripts/tune_integrated.py` (`fit` / `eval` / `sweep`; caches per game/seed features, logs every sweep
evaluation to `cache/tune_sweep_log.jsonl`, resumable). Everything fitted was refit leave-one-game-out
(LOGO: fit on 2 Metrica games x seeds 0-2, score on the third); all selection on extval, no BAR-ATH labels.
Mean over Metrica 1-3 x 3 seeds, decoder at its OLD defaults (pd0 0.4):

| component | in-sample (fit on all 3) | held-out (LOGO) | shipped weights |
|---|---|---|---|
| A weights + B SHAPE_W both refit | 0.528 / 0.787 / 0.910 | **0.523 / 0.779 / 0.906** | 0.518 / 0.780 / 0.909 |
| A LOGO, B shipped | | 0.518 / 0.779 / 0.906 | |
| A shipped, B LOGO | | 0.523 / 0.780 / 0.909 | |
| B shape term off (A LOGO) | | 0.503 / 0.768 / 0.908 | |

(spellF1 / macroF1 / alive-dead F1.) Gap in-sample -> held-out 0.005 spellF1, inside the noise band, so
neither fitted component is overfit to its 3 games; the refit weights (A all-game fit
[-2.77, 3.88, -0.95, -6.04, 2.60, -0.04, 1.25, 2.68, 1.45, 1.41, 1.41, 5]; B [0.0286, 0.411, 0.0797,
0.181, -1.442]) agree with the shipped ones within the fold-to-fold spread (Metrica 3 held out pushes
`missing`/`gap_short` to 5.5/-8.3), so **A `DEFAULT_WEIGHTS` and B `SHAPE_W` are unchanged**. The shape
term is worth +0.02 held-out, mostly via the speed feature.

**Decoder sweep** (held-out emissions, 37 evaluations, coordinate search, a move must
beat the current by > 0.01 spellF1; 2 passes, second pass changed nothing). spellF1 / n_pred, true 292:

| param | values tried (spellF1) | outcome |
|---|---|---|
| evidence | 0.03 .470, 0.06 .510, 0.10 .523, 0.15 .521, 0.25 .523 (n_pred 164..239) | keep 0.10; above it only count rises |
| switch_penalty | 0 .522, 1 .523, 2 .520, 3 .517, 5 .503 | keep 1 |
| alive_median_s | 5 .522, 8 .523, 12 .524 | keep 8 |
| alive_sigma | 0.7 .522, 1 .523, 1.3 .522 | keep 1 |
| dmin_alive_s | 0.6 .523, 1.0 .523, 1.5 .524, 2.5 .526 | keep 0.6 (2.5 would delete real short spells, +0.003 is noise) |
| pd0 | 0.3 **.534**, 0.4 .523, 0.5 .516 | **0.40 -> 0.30**: per game +0.027 / +0.007 / -0.001; matches the extval label dead share (30-33 %) |
| dead_median_s | 6, 9, 13: identical | keep 9 (A's cleaned p_dead is sharp, durations do not bind) |
| dead_weight (new, default 1.0 = old behaviour) | 0.5 .521, 1 .534, 2 .535 | keep 1.0 |

The surface is flat: only `pd0` moved beyond 0.01 (still inside the 0.04 noise band; adopted because it is
prior-justified by the label statistics). Predicted possessions stay ~217 vs 292 true for every setting
with a good score: more `evidence` or smaller `switch_penalty` adds possessions (239 at evidence 0.25)
without improving F1, so the shortfall is missed hand-overs in the emissions, not decoder smoothing.

**Final parameters.** A weights, A feature params, B `PARAMS`, B `SHAPE_W`: unchanged (see sections A, B).
Decoder: `HSMM_DEFAULTS` unchanged except `pd0` 0.40 -> 0.30 (source: this sweep) and the new
`dead_weight` = 1.0. `team_speed` RuntimeWarning is suppressed (results identical).

**Final held-out table** (Metrica 1-3 x 3 seeds):

| pipeline | spellF1 | macroF1 | aliveF1 | n_pred / 292 |
|---|---|---|---|---|
| emissions_b_v1 (B alone, shape fit in-sample) | 0.531 | 0.754 | 0.857 | 221 |
| A + B, untuned, shipped weights | 0.518 | 0.781 | 0.909 | 212 |
| **integrated_v1, LOGO weights + pd0 0.3** | **0.534** | **0.780** | **0.903** | 217 |
| integrated_v1 via harness (shipped weights) | 0.529 [0.514, 0.545] | 0.787 | 0.910 | 214 |

Honest reading: spellF1 gain over B alone is within noise; the real gain of A is macroF1 +0.03 and
alive/dead F1 +0.05, the tuned decoder recovers spellF1 to B's level. SkillCorner secondary check (3 games,
1 seed, not used to select): baseline_v0 0.373, emissions_b_v1 0.543, deadball_a_v1 0.356, integrated_v1
0.570 (pd0 0.4: 0.558); no collapse. Note `extval/datasets.py` opens SkillCorner JSON without an encoding
and failed on Windows cp1252; fixed in the ship phase (`encoding="utf-8"`).

**BAR-ATH run** (`python -m possession.run`, no labels): 259 possessions, H 66227 / A 52485 / D 40145 /
X 40462 frames, two runs sha256 `309f68023a0b5c66adf5aad3fec2584623dae06d6b248e0c973faf39c08325c1`. The
public-window sanity run was done once at the end and not used to choose anything.

**Suspect / overfit risks.** (1) Metrica-trained A weights and SHAPE_W depend on the degradation
simulator (tags assumed 8 % noisy; speed convention differs from the ATD); 3 games only, LOGO folds differ
visibly for Metrica 3. (2) Under-segmentation (217 vs 292) persists; BAR-ATH yields 259 possessions over
two halves, no way to check against truth. (3) `pd0` 0.3 is a small, prior-backed move; the BAR-ATH
detector's dead share is 22 %, below the extval 30-33 %, so BAR-ATH may prefer a lower dead prior still.

## Ship: compliance audit (RULES.md, no use of `evaluation/gt`)

Audit of every parameter whose recorded origin touched the public GT, the public window or BAR-ATH-specific
behaviour. The sweep showed the score is flat on these, so the cost is ~0.

| item | before | after | basis |
|---|---|---|---|
| kickoff `own_half_min`, `min_pre_s` | 0.65, 15 s | 0.60, 10 s | Metrica kickoffs only, `scripts/kickoff_extval.py`; the other five kickoff params are unchanged priors, shown flat there |
| decode `dmin_alive_s` | 0.6 s ("public GT shortest spell 0.76 s") | 1.0 s | extval labels: shortest alive spell 0.6-1.0 s, p5 1.4 s |
| decode `dead_median_s` | 9 s ("public GT 5.5-20 s") | 13 s | extval labels: median dead stretch 13.4-15 s |
| emissions `min_ctrl_s` | 0.4 s ("public GT has sub-second spells") | 0.4 s | value kept, now a prior (< the 1 s control rule, 0.2/0.8 within noise) |
| extval `min_control` | 25 f, "relaxed because public GT has sub-second spells" | 25 f | football reasoning (touch vs control, CHALLENGE.md). 50 f (the "few seconds" reading) changes the labels: integrated_v1 0.584 spellF1 / 0.791 macroF1 on 246 true possessions per game (vs 0.529 / 0.787 on 292), same n_pred. Absolute extval numbers depend on this label rule |

Extval harness (Metrica 1-3 x 3 seeds, before -> after): spellF1 0.529 -> 0.529, macroF1 0.787 -> 0.787,
aliveF1 0.910 -> 0.910, predicted possessions 214.2 -> 214.1 (292 true). BAR-ATH output: 259 -> 258
possessions, 87 of 199319 frames differ; sha256 309f6802... -> `f16a02b670d64a942d709685b258c8336aac9db7f36dc93412bf38aab0fb1bc6`
(byte-identical on two runs).

Other items found that touch the public GT or BAR-ATH (reported, judged acceptable):
1. `extval/calibration.json`, `degrade_fit.json`: statistics of the UNLABELLED BAR-ATH tracking (allowed, RULES item 3).
2. The simulator's calibration check compares baseline scores with the PUBLISHED full-match scoreboard numbers
   (floor 0.2433 etc.) and the first, too-easy version was extended with a ball position error. The added term
   was fitted to an unlabelled statistic (ball-to-nearest-object distance), not to the score, but the decision to
   add it was triggered by the published score. Published scores are organiser data, not the public GT; disclosed
   in TECHNICAL.md.
3. `w_tag` 1.0 from the agreement of Smart Tagging with the control chain on BAR-ATH (unlabelled, allowed).
4. Public-window sanity scores recorded in this file (B, A sections) were not used to choose anything; the
   sanity script was run for alignment only. Not repeated for the before/after comparison above.
5. Looking at the BAR-ATH kickoff output ("half 1 and 2 look right", "within 25 frames of the tagging window")
   used the Smart Tagging tags (input data) and the detector output, not the GT. No threshold was changed
   after seeing it.
6. `extval/harness.py`: degraded-game cache key did not include `labels.py`, so changing a label rule silently
   reused stale truth; now included.

## Ship: reproduction and leak scan

**Reproduction.** Fresh clone of the branch, new Python 3.12.10 venv from `requirements.txt` alone, empty
`MTAH_CACHE_DIR`, `python -m possession.run --pipeline integrated_v1`: 20 s, sha256
`f16a02b670d64a942d709685b258c8336aac9db7f36dc93412bf38aab0fb1bc6`, byte-identical to the working-tree run;
`validate_submission.py` Valid (199319 rows). pytest in the clone: 48 passed, 3 skipped (data/cache dependent).
`requirements.txt` pins every direct dependency; the transitive ones that pytest pulls in (colorama,
iniconfig, packaging, pluggy, Pygments) are unpinned; they do not touch the output. numpy / pandas / scipy /
numba / llvmlite are pinned exactly and are the ones that matter for bytes.

**Leak scan** (whole history, all refs, 21 commits). No path under `data/` other than the organisers' `data/README.md`,
none under `cache/`, `submissions/` or `extval/raw/`; no mp4, npz, pkl, parquet, jsonl, zip. The only
files over 1 MB in any commit are five organiser-supplied 1.68 MB frame-level CSVs from the first commit
(starter kit import): `evaluation/gt/BAR-ATH_possession_gt_public.csv` (the public 5-minute GT, header-only
labels outside it) and `evaluation/results/baseline_{raw,smooth25,smooth50,smooth125}.csv` (the organisers'
four baseline submissions, whole-match labels from a nearest-player rule). Committed JSON: `extval/calibration.json`
3.2 KB and `extval/degrade_fit.json` 0.3 KB (aggregate histograms and quantiles of the unlabelled tracking,
cannot reconstruct positions), `evaluation/gt/..._public_meta.json` 1 KB (organisers'). No external game
data is committed (extval/raw is untracked and ignored).

## F: video ball-detection spike (`video/`, report in `video/spike_report.md`)

Feasibility only; **recommendation NO-GO** for the 2026-11-17 submission. No pipeline uses video.
Nothing here touches `evaluation/gt`; all checks are self-supervised against the ATD ball.

| parameter | value | origin |
|---|---|---|
| detector | YOLOX-S COCO ONNX, class 32 "sports ball", Apache-2.0 (Megvii release 0.1.1rc0), sha256 `c5c2d13e...998063` | public pretrained, no fine-tuning; -M tried, +3 to 5 points recall at 2.9x cost |
| tile | 640x640 native pixels, no downscale | ball is ~10 px in 1080p; 1x full frame = 6 tiles |
| upscale | 2x, `INTER_LINEAR` | measured: ATD-ball recall@50px 61% (1x) -> 74% (2x); oracle 2x crop 74% |
| match radius | 50 px, detector sits 9 px above the ATD ground-plane projection | measured offset (median) on ATD-has-ball frames |
| score threshold | candidates kept at 0.01; analysis at 0.1 to 0.5 (0.3 reported) | spike sweep, not tuned on labels |
| NMS IoU | 0.3 | convention |
| filters | on pitch (+3 m / 3.5 m margin) after `inv(H)`; not a 1.5 m pitch cell hit in 3+ different clips; within 10 m of an ATD player; neighbour continuity under 40 m/s | football/physics priors (ball speed bound) and ATD player positions |
| sample | 100 clips x 30 frames, every 3rd detected; 60 inside graded ATD gaps of 75+ frames (weighted by gap duration), 40 where the ATD has the ball; seed 20271117 | seeded, from the unlabelled ATD mask only |
| threads | 12 (CPU provider; output bit-identical for 1 to 12 threads) | measured |
| runtime | 64,643 missing frames: 4.0 h at 5 Hz / 20 h at 25 Hz (2x, ~14 tiles/frame, CPU) | measured 1.90 s/frame for 24 tiles, 14-tile selection extrapolated |

Env: separate venv `video/.venv` (`video/requirements.txt`: onnxruntime-directml 1.24.4,
opencv-python-headless 5.0.0.93, numpy 2.5.4, pandas 3.1.1); weights in gitignored `video/weights/`.
Outputs live in `cache/video_spike/` and are not committed. Determinism: CPU bit-identical across
thread counts and repeats on this machine (other CPUs untested); DirectML differs numerically from CPU.

## Gaps: can long ball gaps be mined for hand-overs? (result: NO, upper bound too small; nothing built)

Measured before building (`scripts/gaps_measure.py`, `scripts/gaps_measure2.py`; Metrica 1-3 x degradation
seeds 0-2, ORACLE dead-ball, B's `p_home_b` + `hsmm_decode`, no BAR-ATH data or labels touched). A "hand-over"
is a GT frame where an alive team follows the other alive team with no D between (199 per game; 292-316 GT
possessions, the rest are separated by stoppages).

| quantity | value |
|---|---|
| baseline spellF1 (oracle dead, B emissions) | 0.624, n_pred 217-226 vs 316 GT (metrica3) |
| hand-overs whose frame has no ball (inside a gap) | 11 of 199 (**5 %**); the other 95 % happen with the ball tracked |
| length of those gaps (frames, pct 10/25/50/75/90) | 200 / 417 / 671 / 922 / 1419; none is <= 75 f, so interpolation never applies to them |
| alive frames with ball missing | 20 % of alive frames |
| team error rate: visible / missing stretch with a hand-over / missing stretch without | 0.117 / 0.336 / 0.283 |
| **UPPER BOUND, gaps that contain a hand-over made perfect** | **0.648 (+0.024)** |
| upper bound, perfect on missing stretches with no hand-over | 0.652 (+0.028) |
| upper bound, EVERY missing-ball alive frame perfect (oracle) | 0.687 (+0.063) |
| perfect on ball-VISIBLE alive frames only | **0.903 (+0.279)** |
| perfect everywhere | 0.962 |

Conclusion: even an oracle that placed every gap hand-over perfectly is worth +0.024 spellF1, below the 0.04
noise band, and any real tracking-only signal would recover a fraction of it. So no `possession/gaps.py`,
no `integrated_v2` were built and `integrated_v1` is untouched (its sha is unchanged by this task, only
scripts were added). The under-segmentation (217 vs ~300 possessions) is NOT mainly a gap problem:
perfect visible-ball frames lift spellF1 to 0.90 whereas perfect missing-ball frames lift it to 0.69. The
errors live where the ball IS tracked (11.7 % wrong team over 700k frames = 82k frame errors vs 54k on missing
frames). In the simulator that is mostly ball position error (AR(1) 0.6 m plus 8 m bursts 20 % of the time,
fitted to the nearest-object distance quantiles in `extval/degrade_fit.json`), so any further gain should come
from robust control inference on visible-ball frames (bursts of implausible ball position, slot ID swaps,
contested balls), not from gaps.

Caveats: (1) the 5 % share depends on the simulator's gaps (native Metrica gaps plus zone/density injected
ones). Real ATD may lose the ball more often exactly at contested hand-overs; that cannot be checked without GT,
and if the true share were e.g. 20 % the bound would be roughly four times larger (~+0.10), still an oracle
figure. (2) The native-vs-injected feature check was not run since no feature was built.

## Learned team model (`possession/learned.py`, `possession/models/team_gbm.npz`, pipeline `integrated_v2_team`)

`integrated_v2_team = hsmm_decode(match, p_dead_a(match), p_home_e(match), evidence=0.2)`. `integrated_v1` and
`pipelines.DEFAULT` are untouched (BAR-ATH sha still `f16a02b6...1bc6`, checked).

**Model.** LightGBM 4.7.0 binary classifier (250 trees, max_depth 5, 24 leaves, lr 0.05, min 300 rows/leaf,
L2 10, bagging 0.7 / feature fraction 0.8, 63 bins, seeds fixed, 1 thread, `deterministic`), stacked on
`p_home_b`: 35 features from the Match only (`learned.features_eo`): B's logit `z_b`, chain logit, previous/next
controller sign, time since / until control, episode length, in-episode flag, gap length, per-team nearest
distance difference, soft nearest evidence (frame, +/-1 s, +/-3 s), control-evidence balance in 7 windows
(+/-0.5, 2, 5 s, past/lead 2 and 6 s), the 5 shape features and their logit, tag evidence, ball position in the
home attack frame, players within 10 m of the ball (sum/diff), relative-speed difference, ball speed, visible
object count. Trained on every 4th alive frame of Metrica 1-3 x degradation seeds 0-2 (simulator v2), labels =
extval H/A. **No home bias by construction**: features are split into even / odd (sign flips with a
home<->away swap), training sees each row also swapped with the label flipped, inference averages both views
(test: `p(match) + p(swapped match) = 1`). Features are rounded to 1e-6 (float noise at a ~0 split threshold
otherwise broke that symmetry).

**Shipping.** The trees are exported to `team_gbm.npz` (329 KB, 250 trees, depth 5) and evaluated in pure numpy
(`learned.predict_raw`); lightgbm is needed only to retrain (`requirements-train.txt`, `lightgbm==4.7.0`). `python scripts/train_learned.py final` regenerates the file;
`... check` retrains twice and compares: **byte-identical** (sha256 `a388be89...ce8aa4`, also equal to the committed
file). numpy vs lightgbm predictions agree to float32 (max |dP| = 0.0 on a 200 k-frame run). Retraining after
changing any feature, `degrade.py` or `emissions.py` needs the cache key to change (it hashes their mtimes).

**Leave-one-game-out** (Metrica 1-3 x seeds 0-2 = dev seeds; real A dead-ball; spellF1 / macroF1 / aliveF1,
n_pred of 292), `scripts/train_learned.py logo`, `scripts/learned_dev.py`:

| | spellF1 | macroF1 | aliveF1 | n_pred | team acc alive vis / missing |
|---|---|---|---|---|---|
| integrated_v1 (B emissions, evidence 0.1) | 0.526 | 0.786 | 0.909 | 221 | 0.874 / 0.692 |
| learned, held-out game, evidence 0.1 | 0.567 | 0.802 | 0.907 | 198 | 0.910 / 0.717 |
| learned, held-out game, **evidence 0.2** | **0.581** | 0.803 | 0.908 | 218 | |
| learned, in-sample (fit on all 3), evidence 0.1 | 0.597 | 0.825 | 0.908 | 199 | |

Per game held-out (evidence 0.1) v1 -> learned: Metrica 1 0.517 -> 0.534, Metrica 2 0.512 -> 0.583,
Metrica 3 0.549 -> 0.585; all three improve. In-sample vs held-out gap 0.03: some overfit to the 3 games.
Decoder `evidence` on the held-out probabilities (the only decoder knob moved, 6 + 2 evaluations): 0.05 0.533,
0.10 0.567, 0.20 0.581, 0.30 0.578; `switch_penalty` 0 +0.005, `pd0` 0.4 -0.009. Gain source: team accuracy on
ball-visible frames 0.874 -> 0.910 (feature gain: `z_b` 0.49, shape logit 0.09, control balance windows 0.10,
gap length 0.05, near 3 s 0.04, ball position 0.04). Adding rolling `z_b` windows and a 3 m crowding count gave
nothing (0.5775 vs 0.5796) and was dropped. `control_events` change points were not retried: with B alone they
gave no gain and the model's per-frame probability is what the HSMM already integrates.

**Simulator-artefact check** (`scripts/learned_artefact.py`; model trained on degraded Metrica only):

| data | v1 spellF1 | learned (evidence 0.2) | team acc vis / missing v1 -> learned |
|---|---|---|---|
| clean Metrica 1-3, LOGO model, seed-0 tags | 0.615 | **0.639 (+0.024)**, all 3 games up | 0.917/0.649 -> 0.935/0.702 |
| SkillCorner 4 games, seed 0 (broadcast tracking, native noise) | 0.563 | **0.551 (-0.012)** | 0.924/0.767 -> 0.925/0.745 |

Reading: the gain shrinks from +0.055 (degraded, held-out game) to +0.024 on clean Metrica and vanishes on
SkillCorner. Part of what the model learned is specific to the simulator's ball error (and to the Metrica
football/labels); it does not transfer to a different tracking domain. Whether BAR-ATH behaves like the degraded
games (simulator calibrated on its unlabelled statistics, which is why we trust it) or like SkillCorner cannot be
checked without BAR-ATH labels.

**Fresh seeds 3-5** (not used for any choice; same three Metrica games, so NOT held-out in game terms; harness,
bootstrap 200):

| pipeline | spellF1 [95% CI] | macroF1 | aliveF1 | n_pred / 292 |
|---|---|---|---|---|
| integrated_v1 | 0.5075 [0.493, 0.524] | 0.7805 | 0.907 | 220 |
| integrated_v2_team | **0.5967** [0.582, 0.613] | 0.8164 | 0.905 | 215 |

Per game v1 -> v2: 0.484 -> 0.576, 0.523 -> 0.612, 0.515 -> 0.602. The +0.089 exceeds the 0.04 bar, but it is an
optimistic estimate (the model saw these games under other seeds): the honest game-level numbers are LOGO +0.055
(0.526 -> 0.581, dev seeds, the evidence setting 0.2 chosen on those) and clean Metrica +0.024.

**BAR-ATH** (`python -m possession.run --pipeline integrated_v2_team`): 242 possessions (v1: 258), H 63462 / A 55500 /
D 39895 / X 40462, two runs sha256 `389136433bf13081439b1f5fb6bad96d3da682cd583a906328e051bd60aa2a36`. The
public-window sanity script was run once at the end (format check, spellF1 0.586, 11 predicted vs 17 true
possessions); it was not used for any choice. Fewer possessions than v1 on BAR-ATH and 11 vs 17 on the public
window mean the model is, if anything, MORE conservative than needed there: a reason to doubt blind adoption.

**Risks.** (1) Trained on 3 Metrica games through a simulator: SkillCorner shows no transfer. (2) The tags are
the simulator's (8 % flips), real tags agree 76 % with the chain, so the model may over-trust tag features
(`tag` has low importance). (3) `evidence` 0.2 was chosen on the dev seeds. (4) Slower: features + model ~20 s
for BAR-ATH.

## Dead ball v2 (`deadball.p_dead_a2`, pipeline `integrated_v2_dead`)

`p_dead_a`, `integrated_v1` and `pipelines.DEFAULT` are untouched (integrated_v1 BAR-ATH sha still
`f16a02b6...1bc6`). Kickoff parameters are the audit's. `integrated_v2_dead` = `hsmm_decode(match,
p_dead_a2(match), p_home_b(match))`.

**Why.** Oracle ablation (Metrica 1-3 x seeds 0-2, simulator v2, B emissions, HSMM): `integrated_v1` 0.526
spellF1; true dead state everywhere 0.611; true state on VISIBLE-ball frames only 0.567, on MISSING-ball
frames only 0.568. So both visible-but-dead frames and gap classification matter, about +0.04 each.

**Model.** 36 per-frame features (`frame_features2`: the 11 A1 features without the bias, plus 25 new) into
a boosted-tree ensemble (`possession/deadball_a2_model.npz`, 23 kB; pure-numpy inference `gbdt_predict`;
retrain with `scripts/train_deadball_a2.py fit`, deterministic, no randomness). Then A1's `clean_stretches`
and the prior `pre_kickoff` override (p >= 0.98 before the detected kickoff, because extval halves start at
the kickoff and give nothing to learn it on). New features:

| group | features | reasoning |
|---|---|---|
| ball visible but dead | `vis_edge` (visible ball near a line), `lmv_f1/b1/c4` (log ball displacement over the next / previous 1 s and +/- 2 s) with `vis_f/vis_b` flags, `near_player` (ball within ~1.5 m of a player: held) | ball stationary or held, at a line |
| collective stop | `frac_slow2` (share of players < 0.8 m/s, +/- 1 s), `r2`, `r6` (speed collapse vs the 60 s norm at +/- 1 s and +/- 3 s), `r_before4`, `r_after4`, `r_delta` (collapse before vs after each frame) | whistle: everyone decelerates; before/after asymmetry places the onset |
| restart touch | `lmv_b1` small with `lmv_f1` large (ball still, then moving), `prv_*` below | sharp offsets |
| context | `miss_ctx3/10` (share of missing ball within +/- 3 / 10 s), `nobj`, `nobj_ctx3` (tracked objects / 22: tracking outage vs stoppage), `nxt_dt/nxt_len/nxt_edge` and `prv_dt/prv_len/prv_edge` (seconds to the next gap, its length and line proximity; same for the previous gap, capped at 10 s), `gap_in/gap_out` (position inside the gap) | a visible ball just before a dead gap is probably dead too |

| hyper-parameter | value | origin |
|---|---|---|
| trees / depth / lr / L2 / min leaf / bins | 100 / 4 / 0.1 / 10 / 100 rows / 32 | first guess, never tuned. Depth 6 + 80 trees + min leaf 150 (LOGO, stride 6) gave 0.545 spellF1, same as depth 4: capacity is not the limit |
| training rows | every 3rd graded frame of Metrica 1-3 x seeds 0-2, simulator v2, label D | extval |
| feature scales (edge 6 m, ctx 10 s, slow 60 s) | as A1 | priors |

**Leave-one-game-out** (fit on 2 games x 3 seeds, scored on the 3rd, seeds 0-2; spellF1 / macroF1 /
alive-dead F1 / frame-level dead F1; A1 = `integrated_v1` on the same runs 0.526 / 0.786 / 0.909):

| held out | spellF1 | macroF1 | alive/dead F1 | frame dead F1 |
|---|---|---|---|---|
| metrica1 | 0.553 | 0.801 | 0.925 | 0.833 |
| metrica2 | 0.553 | 0.826 | 0.940 | 0.871 |
| metrica3 | 0.526 | 0.788 | 0.918 | 0.797 |
| **mean** | **0.544** | **0.805** | **0.928** | 0.834 |

Edge timing of dead stretches (LOGO predictions vs extval truth, 822 stretches): median |onset error|
49 -> 21 frames (median signed +26 -> +4; within 25 frames 33 % -> 55 %); offsets (restart touch) within 25
frames 82 % -> 89 %. Frame errors (A1 -> A2): missed dead with visible ball 57k -> 49k, missed dead in
gaps 45k -> 33k, false dead in gaps 61k -> 43k, false dead on visible frames 0.7k -> 4.3k.
Post-processing variants on the LOGO predictions (no clean-up, 3 s clean-up, 0.8 s, logit temperature 0.6
and 1.6): all within 0.005 spellF1 (0.539-0.546), so the A1 clean-up is kept. About 38
evaluations were used in development.

**Fresh seeds 3-5** (run once, after the model was frozen; Metrica 1-3 x seeds 3-5, simulator v2):

| pipeline | spellF1 | macroF1 | alive/dead F1 | frame dead F1 | median abs onset err (f) | n_pred (292 true) |
|---|---|---|---|---|---|---|
| integrated_v1 | 0.5075 | 0.7805 | 0.9070 | 0.785 | 50.5 | 220 |
| **integrated_v2_dead** | **0.5424** | **0.8249** | **0.9463** | 0.878 | 13.6 | 224 |

Gain +0.035 spellF1 (below the +0.04 bar; reported as a smaller-than-target result), +0.044 macroF1,
+0.039 alive/dead F1. Positive on all 9 runs (per-run +0.022 .. +0.051). The weights were fitted on seeds 0-2
of the same three games, so the fresh seeds are new degradations of known games, not new games: the LOGO
numbers above are the unbiased across-game estimate (+0.018 spellF1).

**Simulator-artefact check (secondary, not used for any choice).**

| set | integrated_v1 spell / macro / alive | integrated_v2_dead | per-run spell delta |
|---|---|---|---|
| CLEAN Metrica 1-3 (no degradation, native ball dropouts, no tags) | 0.587 / 0.800 / 0.907 | 0.628 / 0.848 / 0.947 | +0.070, +0.038, +0.018 |
| SkillCorner, 3 games, seed 0 (broadcast tracking, 10 fps resampled) | 0.570 / 0.810 / 0.871 | 0.558 / 0.758 / 0.830 | +0.012, -0.030, -0.017 |

The gain survives on clean Metrica (the model was trained on the degraded copies, so this is a mild
check: the native Metrica ball behaviour is the part the simulator keeps). It does NOT transfer to
SkillCorner: macroF1 -0.05, alive/dead -0.04. SkillCorner is a different domain (broadcast camera, ~9
players visible, 10 fps, dropouts when the ball leaves the frame), the learned context features (tracking
outage via `nobj`, gap-to-line structure, 10 s windows) are tuned to the simulated tactical-camera
dropouts. BAR-ATH is tactical full-pitch tracking like Metrica, so SkillCorner is the weaker analogy, but
this is a real warning that A2 leans on simulator conventions (dead stretches that start before the
ball leaves, gap statistics). If simulator dependence is a concern, A2 should be
treated as a +0.02..0.035 spellF1 option with that risk, not a free win.

**BAR-ATH run** (`python -m possession.run --pipeline integrated_v2_dead`, no labels): 256 possessions
(integrated_v1: 258), H 63612 / A 45365 / D 49880 / X 40462, sha256
`267451314eeb272fdfac90b0f92c9648e04faceeb042394ee92a6f6ead3e51ef`, two runs identical.
Detected dead share of graded frames 28.9 % in 110 stretches, median 13.5 s (A1: 22.4 %, 91 stretches,
10.9 s; extval truth 30-33 %, median 13-15 s). `sanity_public.py` run once as a format check:
spellF1 0.525, macroF1 0.680, alive/dead 0.917, 15 possessions vs 17 (not used for any choice).

**Known failure modes.** Mid-pitch dropouts with normal player speed still look dead when the ball
vanished near a line or the team slowed for another reason (false dead in gaps 43k frames on 9 LOGO runs);
dead frames with the ball visible and moving (ball retrieval) remain mostly missed (49k); learned onset
offsets inherit the extval label convention (the truth starts a stoppage when the previous event ends,
often before the ball crosses the line), which a real GT that marks the crossing would not share;
SkillCorner degradation above. Not done: restart-geometry counts (players in the box, wall), kept out
because the 6-yard-box / corner check added nothing in v1.

## Combination LOGO (`integrated_v2` = A2 dead ball + learned team model, `scripts/combo_logo.py`)

`integrated_v2 = hsmm_decode(match, p_dead_a2(match), p_home_e(match), evidence=0.2)`, committed models, new file
`possession/pipelines/integrated_v2.py`. Leave-one-game-out: for each held-out Metrica game both models are
trained on the other two games x seeds 0-2 (A2: `train_deadball_a2.py logo`, team: `train_learned.py`
`train`), scored on the held-out game x seeds 0-2 (simulator v2). Fold models live in memory / cache only; the
committed model files are unchanged. No tuning, no seeds 3-5, no sanity_public.

spellF1 / macroF1 / alive-dead F1, n_pred (true 285 / 275 / 316, mean 292):

| held-out game | integrated_v1 | A2 only (A's LOGO, evidence 0.1) | team only (`p_dead_a`, evidence 0.2) | **combination** |
|---|---|---|---|---|
| Metrica 1 | 0.517 / 0.761 / 0.895, 211 | 0.553 / 0.801 / 0.925 | 0.534* / | **0.589 / 0.816 / 0.924**, 206 |
| Metrica 2 | 0.512 / 0.796 / 0.915, 217 | 0.553 / 0.826 / 0.940 | 0.583* / | **0.627 / 0.843 / 0.940**, 224 |
| Metrica 3 | 0.549 / 0.800 / 0.917, 236 | 0.526 / 0.788 / 0.918 | 0.585* / | **0.593 / 0.814 / 0.918**, 239 |
| **mean** | 0.526 / 0.786 / 0.909, 221 | 0.544 / 0.799 / 0.923 | 0.581 / 0.803 / 0.908, 218 | **0.603 / 0.824 / 0.927**, 223 |

(*per-game team-only numbers are from the evidence 0.1 run in "## Learned team model"; its mean at evidence
0.2 is 0.581.) The combination with the decoder's default `evidence` 0.1 scores 0.591 / 0.825 / 0.927, 201
possessions (evidence 0.2 is the setting fixed in the team-model section, not re-tuned here).

The two gains add up roughly: +0.077 over v1 (A2 alone +0.018, team alone +0.055). All three held-out games
improve over v1 (+0.072, +0.115, +0.044). The caveats of the team model stand: its gain shrinks to +0.024 on
clean Metrica and is absent on SkillCorner (see "## Learned team model"); nothing here tests A2 on other
domains.

**BAR-ATH** (`python -m possession.run --pipeline integrated_v2`): 241 possessions, H 62424 / A 46168 / D 50265 /
X 40462, two runs identical, sha256 `1c9347200a79d2d9f7d25ebfaf773910d7928253b77cb61de0d80bff268e3d93`.
`integrated_v1` is still `f16a02b6...1bc6`. pytest: 62 passed.

**Adopted** as `pipelines.DEFAULT` (after the combination LOGO above). Fresh-clone
reproduction with `requirements.txt` only (no LightGBM) and an empty cache gives the same sha256
`1c934720...3e93`; Windows only, Linux not checked.

## Event detection (`possession/events.py`, `scripts/train_events.py`, pipeline `integrated_v3`)

**Idea.** The external labels are a mechanical rule over Metrica events (`extval/labels.py`): a team's first control
event (PASS, SHOT, CARRY, RECOVERY, SET PIECE) of a run that survives `min_control` = 25 frames starts a spell; ball
out / goal / foul stops it. We learn to detect those spell onsets from tracking and give the detector's output to the
team model. `integrated_v2`, `integrated_v1`, `pipelines.DEFAULT` and the committed v2 models are untouched
(BAR-ATH `integrated_v2` sha re-checked: `1c934720...3e93`). No BAR-ATH label was read; all selection is on external
games.

**Event counts** (`scripts/event_counts.py`; Metrica events as loaded, then onsets in the derived H/A labels):

| game | events | PASS | CARRY | RECOVERY | CHALLENGE | BALL LOST | BALL OUT | SET PIECE | SHOT | FAULT | alive onsets (turnover / after dead ball) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Metrica 1 (CSV) | 1745 | 799 | 0 | 278 | 233 | 257 | 51 | 77 | 24 | 22 | 283 (197 / 86) |
| Metrica 2 (CSV) | 1935 | 964 | 0 | 248 | 311 | 233 | 49 | 80 | 24 | 20 | 273 (184 / 89) |
| Metrica 3 (JSON) | 3620 | 1121 | 1395 | 294 | 335 | 288 | 64 | 90 | 20 | 12 | 314 (217 / 97) |

CHALLENGE, BALL LOST, FAULT RECEIVED and CARD never grant possession in the rule; BALL OUT, goals and fouls only
start dead stretches. The pass-release target was not built: the label rule uses the control onset, not the release.

**Target.** Per frame and team: "a spell of that team starts within +/- 8 frames (0.32 s)" in the derived labels,
including restarts (after a dead ball) and turnovers (previous state the other team). Two views per row (own /
opponent columns, odd columns negated) so home and away share one model and there is no home bias.

**Detector.** LightGBM (150 trees, 16 leaves, depth 5, lr 0.05, 200 rows/leaf, L2 10, 63 bins, bagging 0.7, seeds fixed,
1 thread, every 2nd graded frame of Metrica 1-3 x seeds 0-2; first-guess hyper-parameters, never tuned). Features
(`events.feature_names()`): the 35 team-model features, 10 ball-kinematics features (mean speed over [-8,0], [0,8],
[-25,-8], [8,25] frames, velocity-change magnitude, cosine / angle of the direction change, share of missing ball
frames in +/-0.34 s), 14 per-team nearest-player features for own and for the opponent (distance now and at -12 / -6 /
+6 / +12 / +25 frames, min distance over [-12,0] and [0,12], relative speed of the nearest player now and over +/-10
frames, players within 5 m and 3 m, approach), and 8 odd distance-difference features. Top gain: nearest-player
distance difference min over the next 0.5 s, missing-ball share, past ball speed, velocity change. Peak-picking:
score smoothed over +/-3 frames, maximum within +/-0.5 s, threshold 0.3 (0.2 for the decoder change-point variant).

**Detection quality** (held-out game, +/-12 frames = 0.5 s, one-to-one team-matched, 3 degradation seeds per game):

| set | threshold | precision | recall | recall on turnover onsets | picked / true |
|---|---|---|---|---|---|
| dev seeds 0-2 | 0.2 | 0.529 | 0.745 | 0.668 | 3677 / 2610 |
| dev seeds 0-2 | 0.3 | 0.637 | 0.674 | 0.574 | 2761 / 2610 |
| dev seeds 0-2 | 0.4 | 0.708 | 0.607 | 0.483 | 2238 / 2610 |
| seeds 6-8 | 0.2 | 0.545 | 0.755 | 0.679 | 3619 / 2610 |
| seeds 6-8 | 0.3 | 0.645 | 0.676 | 0.575 | 2734 / 2610 |

Per game at 0.2 (dev): Metrica 1 0.54 / 0.74, Metrica 2 0.51 / 0.79, Metrica 3 0.54 / 0.71 (precision / recall).

**Integration, two ways tried** (everything refitted per fold):
(a) `hsmm_decode(..., change_points=events)` next to the unchanged `p_home_e` (cp_weight 3 or 6, thresholds 0.2 / 0.3):
+0.004 .. +0.010 spellF1. (b) event-derived columns added to the team model and the model retrained on
OUT-OF-FOLD event scores: **+0.068**. (b) wins and is `integrated_v3`; (a) on top of (b) adds +0.004 (not used).
`integrated_v3 = hsmm_decode(p_dead_a2, p_home_e3, evidence=0.2)`. The extra columns (`events.team_extra`): time since /
until the last / next picked event, its confidence, last / next event team sign, difference of the two teams' time
since / until, confidence-weighted proximity of the nearest event per team, and +/-0.5 s / 2 s windowed
score differences (9 odd + 5 even). Stacking: the team model is trained on event scores each training game gets
from an event model fitted on the OTHER training games, never on in-sample scores.

**Leave-one-game-out** (every fitted piece, A2 dead ball, event model, team model, refitted on the two other games x
seeds 0-2; scored on the held-out game; `scripts/train_events.py logo`; spellF1 / macroF1 / alive-dead F1, n_pred of 292 true):

| held-out game (dev seeds 0-2) | integrated_v2 | (a) cp thr 0.2 w6 | (b) integrated_v3 |
|---|---|---|---|
| Metrica 1 | 0.589 | 0.594 | **0.651** |
| Metrica 2 | 0.627 | 0.647 | **0.709** |
| Metrica 3 | 0.593 | 0.597 | **0.652** |
| mean spell / macro / alive, n_pred | 0.603 / 0.824 / 0.927, 223 | 0.613 / 0.825 / 0.927, 230 | **0.671 / 0.829 / 0.928, 256** |

(v2 reproduces the 0.603 of "## Combination LOGO".) Gain +0.068 (+0.062, +0.082, +0.059 per game). The gain is spellF1,
not macroF1 (+0.004): more of the true short spells are found (256 predicted possessions vs 223, truth 292).

**Fresh seeds 6-8** (never used before; same fold models, trained on seeds 0-2 of the other two games, so held-out in
game and in degradation seed; run once after the pipeline was fixed):

| held-out game | integrated_v2 | (a) cp thr 0.2 w6 | (b) integrated_v3 |
|---|---|---|---|
| Metrica 1 | 0.574 | 0.581 | **0.663** |
| Metrica 2 | 0.620 | 0.636 | **0.703** |
| Metrica 3 | 0.582 | 0.590 | **0.664** |
| mean spell / macro / alive, n_pred | 0.592 / 0.826 / 0.930, 220 | 0.603 / 0.827 / 0.930, 229 | **0.677 / 0.834 / 0.929, 253** |

+0.084 spellF1, above the +0.04 bar on all three held-out games and on both seed sets. Number of evaluations used for
the decision: one LOGO with 7 variants (v2, four cp settings, b, b+a); no detector or team hyper-parameter was tuned.

**Secondary checks (not used for selection).**

| set | integrated_v2 | integrated_v3 |
|---|---|---|
| clean Metrica 1-3 (LOGO fold models, native ball dropouts, seed-0 tags) | 0.665 | **0.756** |
| SkillCorner, 4 games, seed 0, models fit on all 3 Metrica games | **0.550** (macro 0.759) | 0.522 (macro 0.754) |
| degraded Metrica, tags removed everywhere (all pieces refitted without tags; A2 as fitted) | 0.572 | **0.641** (+0.069) |

Reading. (1) The gain is not an artefact of the simulator's tags: the simulator builds tags from the truth, so the model
was refitted and tested with no tags and still gains +0.069. (2) It survives clean Metrica (+0.09). (3) It does NOT
transfer to SkillCorner (-0.028): the detector learns the Metrica event convention (control onsets as annotated in
Metrica's event stream, ~1 s control rule) and Metrica tracking, whereas SkillCorner's labels are Phases of Play at 10 fps
from broadcast tracking. The same domain warning as for the v2 team model (+0.024 clean / -0.012 SkillCorner) applies and
is stronger here: v3's gain on Metrica is large precisely because the labels are built from the events the detector
imitates. BAR-ATH's real GT is built from corrected tracking and a human possession definition (CHALLENGE.md), tactical
full-pitch tracking like Metrica; whether it follows Metrica's convention cannot be checked without its labels. Treat
+0.07 as an upper bound for BAR-ATH.

**Shipping.** `possession/models/event_gbm.npz` (138 KB, 150 trees) and `team_gbm_ev.npz` (324 KB, 250 trees); pure-numpy
inference (`learned.predict_raw`). `python scripts/train_events.py final` regenerates both; `... check` retrains twice:
**byte-identical** (event model sha256 `4cfcd417...3dbec`, team model `1b8872e2...b62b`). Other commands: `det`
(detector P/R), `logo`, `test678`, `secondary`, `notags`.

**BAR-ATH** (`python -m possession.run --pipeline integrated_v3`; no labels read, `sanity_public` not run): 251 possessions
(integrated_v2: 241), H 61864 / A 46668 / D 50325 / X 40462, two runs identical, sha256
`17c7c6b6c0e21de3e41c96ae5b46ec47424ad9cf858d1a17a645726f8c39560a`. More possessions than v2, the same direction as on the
external games (256 vs 223 predicted, 292 true).

**Risks.** (1) SkillCorner transfer failed; (2) the third-ranked detector feature is the share of missing ball frames
around the frame (the simulator's dropout pattern), which may not match BAR-ATH's real dropouts; (3) trained on 3 games
(9 runs); (4) A2 in the `notags` run was fitted with tags; (5) feature extraction is ~40 s on BAR-ATH.

**Adopted** as `pipelines.DEFAULT` (after the LOGO and seeds 6-8 results above), replacing
`integrated_v2`. Fresh-clone reproduction with `requirements.txt` only (no LightGBM) and an empty cache gives
the same sha256 `17c7c6b6...560a`; Windows only, Linux not checked.

## DFL / IDSSE data (`extval/dfl.py`, games `dfl1..dfl7`, `scripts/dfl_*.py`, pipelines `integrated_v4`, `integrated_v4_a2m`)

**Data and licence.** Sportec / DFL open Bundesliga data (IDSSE): 7 matches, 2022/23, TRACAB optical tracking at 25 fps,
synchronised events, CC-BY 4.0. Cite as: Bassek, M., Rein, R., Weber, H., Memmert, D., *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y; data
doi:10.6084/m9.figshare.28196177 (mirror huggingface.co/datasets/pysport/idsse-data). **Download**: 21 XML files, 2.63 GB
(per match: ~0.4 GB positions, ~0.7 MB events, 12 KB match information) into the gitignored `extval/raw/dfl/`, about 2 minutes
(`extval.dfl.download()`). **Load**: the first load of a match parses the position XML in ~9 s and caches the arrays in
`cache/extval_dfl<k>.npz`. No new dependency: the loader reads the XML directly, kloppy is NOT needed (`requirements*.txt`
unchanged). Pitch 105 x 68 m, ground plane only (ball z dropped); half 1, 500 NaN frames (half time), half 2; the halves run
kick-off to final whistle.

**Label flags EXIST.** Every BALL frame of the position file carries `BallStatus` (1 alive / 0 dead) and `BallPossession`
(1 home / 2 away; an owner is also named while the ball is dead, exactly the situation CHALLENGE.md describes for the
competition tracking). `labels.derive_dfl_flags` = D where the status is 0, else the owning team. That is the closest public
analogue of how the BAR-ATH GT is built, so it is the PRIMARY label (games `dfl<k>`). Second label (games `dflev<k>`, same
tracking): the event stream converted to the Metrica schema (`dfl.read_events`: Play -> PASS, ShotAtGoal -> SHOT, SuccessfulShot ->
GOAL, OtherBallAction -> CARRY, BallClaiming -> RECOVERY, wrappers KickOff / ThrowIn / FreeKick / CornerKick / GoalKick / Penalty ->
SET PIECE plus BALL OUT or FAULT RECEIVED at the referee's DecisionTimestamp, Offside) through the unchanged Metrica rules R1-R5
(`max_foul_dead`, `max_unlogged` widened to 750 frames). Event wall-clock times are mapped to frames through the first tracking
timestamp; the DFL events carry no terminations, so ball-out and fouls rely on the DecisionTimestamp.

**Label statistics** (graded halves; possessions / mean duration / dead share):

| game | flags | events | frame agreement | alive-vs-dead agreement | team agreement (both alive) | event labels scored vs flags (spell / macro) |
|---|---|---|---|---|---|---|
| dfl1 | 511 / 6.7 s / 41.5% | 376 / 10.8 s / 30.5% | 0.768 | 0.851 | 0.853 | 0.419 / 0.767 |
| dfl2 | 500 / 5.5 s / 51.7% | 395 / 8.7 s / 39.6% | 0.694 | 0.828 | 0.706 | 0.253 / 0.667 |
| dfl3 | 402 / 7.2 s / 47.6% | 332 / 10.9 s / 33.8% | 0.743 | 0.818 | 0.850 | 0.470 / 0.739 |
| dfl4 | 456 / 6.7 s / 46.5% | 397 / 8.9 s / 37.7% | 0.792 | 0.878 | 0.835 | 0.472 / 0.783 |
| dfl5 | 421 / 7.7 s / 44.3% | 348 / 11.6 s / 31.2% | 0.757 | 0.841 | 0.845 | 0.457 / 0.746 |
| dfl6 | 401 / 8.4 s / 40.7% | 371 / 10.8 s / 29.4% | 0.796 | 0.869 | 0.875 | 0.530 / 0.794 |
| dfl7 | 370 / 9.2 s / 41.9% | 316 / 12.8 s / 31.3% | 0.787 | 0.881 | 0.836 | 0.444 / 0.773 |
| **mean** | **437 / 7.3 s / 44.9%** | **362 / 10.6 s / 33.4%** | 0.762 | 0.852 | 0.829 | 0.435 / 0.753 |

Reference: Metrica event-rule labels 285 / 275 / 316 possessions, mean 13.5 / 14.1 / 12.7 s, dead 33 / 32 / 30%. BAR-ATH public
summary (DATA.md, first 5 minutes of play only): 17 possessions, mean 14.2 s, dead 1469 of 7500 frames = 19.6% (about 360 possessions
if that rate held for the whole match). The DFL flags are MORE fragmented (7.3 s; 9-17% of spells shorter than 1 s; median 3-5 s vs 7-8 s
in Metrica) and have MORE dead time than Metrica or BAR-ATH; the event-rule DFL labels sit between Metrica and the flags. The dead share
matches DFL's own playing time (about 58% effective). The two DFL conventions disagree on a quarter of the frames, and event-rule
labels scored against the flags reach only 0.44 spellF1: the label convention alone is worth ~0.3 spellF1 of headroom.

**TRACAB artefact, and what we did about it.** While `BallStatus` is 0 the ball position is held EXACTLY frozen (zero displacement in
100% of dead frames) and jumps at the restart. That is not an observation and BAR-ATH / Metrica show no such plateau (Metrica's tracker
simply loses the ball: P(NaN | dead) 0.79-0.89, P(NaN | alive) 0.08-0.19). So `load_dfl` sets the ball to NaN inside each dead run except
the first 3.0 s and the last 1.5 s (`DEAD_KEEP_*`): P(NaN | dead) 0.84 and, after degradation, P(NaN | alive) 0.06 (0.00-0.11 per game;
dfl2 has 52% dead time so its visibility stays below the 59.3% target). A first version (keep 2.0 s / 1.0 s) had P(NaN | alive) = 0.03:
models then reached an unrealistic alive/dead F1 of 0.98-0.99 on held-out DFL games (missing ball means dead); it was discarded and
everything below was rebuilt. Even now, the 0.97-0.99 alive/dead F1 of DFL-trained dead-ball models is very likely still partly this
artefact (the kept-visible dead frames are exactly stationary): read DFL alive/dead numbers with that in mind.

**Degradation realism** (`scripts/dfl_realism.py`, simulator v2 unchanged, seeds 0-2, mean over the 7 games vs the unlabelled BAR-ATH
statistics in `calibration.json`):

| statistic | BAR-ATH | DFL degraded |
|---|---|---|
| ball visible (graded) | 0.593 | 0.589 (0.565 dfl2 .. 0.595) |
| ball gaps per match | 150 | 105 |
| ball-to-nearest-object distance q10 / q50 / q90 / q99 (m) | 0.44 / 1.91 / 7.02 / 16.35 | 0.70 / 2.49 / 7.83 / 17.77 |
| ball step p50 / p90 (m), share > 0.5 m | 0.203 / 0.572, 0.150 | 0.234 / 0.633, 0.190 |
| objects per frame, share > 22, outage share | 20.49, 0.024, 0.077 | 20.37, 0.022, 0.077 |
| births per player-minute | 0.52 | 0.82 |
| P(NaN \| dead), P(NaN \| alive) | not measurable | 0.84, 0.06 |

Object counts and outages match; the ball is a little farther from the nearest object and rougher (the same excess Metrica shows after
degradation); slots are born more often (more substitutes). Calibration check, `baseline_v0` on the 7 degraded games vs the PUBLISHED
floors (spell / macro): window 0 0.084 / 0.716 (published 0.049 / 0.653), 25 0.237 / 0.715 (0.200 / 0.675), **50 0.258 / 0.703 (0.243 / 0.672)**,
125 0.201 / 0.643 (0.224 / 0.636). Window 50 is the spellF1 optimum as published (+0.015); macroF1 runs +0.03 high and its optimum moves to
window 0 (published: 25): the larger dead share makes dead/alive easier.

**Step 4: transfer of the COMMITTED (Metrica-trained) models** (`scripts/dfl_transfer.py`; nothing fitted; flags labels, degraded seeds
0-2; spellF1 / macroF1 / alive-dead F1, n_pred of n_true):

| game | v1 | v2 | v3 |
|---|---|---|---|
| dfl1 | 0.284 / 0.762 / 0.890, 232 of 511 | 0.326 / 0.749 / 0.876, 234 | 0.365 / 0.756 / 0.875, 260 |
| dfl2 | 0.280 / 0.722 / 0.860, 247 of 500 | 0.327 / 0.716 / 0.836, 261 | 0.381 / 0.724 / 0.835, 287 |
| dfl3 | 0.346 / 0.786 / 0.899, 220 of 402 | 0.385 / 0.743 / 0.853, 224 | 0.443 / 0.751 / 0.852, 236 |
| dfl4 | 0.354 / 0.761 / 0.881, 253 of 456 | 0.374 / 0.739 / 0.859, 266 | 0.413 / 0.749 / 0.860, 272 |
| dfl5 | 0.362 / 0.765 / 0.883, 228 of 421 | 0.396 / 0.767 / 0.879, 223 | 0.442 / 0.773 / 0.879, 246 |
| dfl6 | 0.376 / 0.730 / 0.859, 242 of 401 | 0.429 / 0.766 / 0.879, 237 | 0.485 / 0.772 / 0.878, 278 |
| dfl7 | 0.362 / 0.794 / 0.904, 220 of 370 | 0.408 / 0.768 / 0.877, 235 | 0.446 / 0.774 / 0.875, 242 |
| **mean, degraded, flags** | 0.338 / 0.760 / 0.882, 234 of 437 | 0.378 / 0.750 / 0.865, 240 | **0.425 / 0.757 / 0.865, 260** |
| mean, degraded, event-rule labels | 0.354 / 0.729 / 0.897, 234 of 362 | 0.387 / 0.721 / 0.894, 240 | 0.403 / 0.721 / 0.894, 260 |
| mean, CLEAN tracking, flags | 0.403 / 0.798 / 0.899, 244 of 437 | 0.427 / 0.780 / 0.881, 241 | 0.497 / 0.790 / 0.879, 274 |
| mean, CLEAN tracking, event-rule labels | 0.405 / 0.757 / 0.909, 244 of 362 | 0.418 / 0.736 / 0.900, 241 | 0.441 / 0.736 / 0.900, 274 |

(Clean = undegraded tracking with the seed-0 synthetic tags.) **The learned layers DO transfer to a new provider**: v3 beats v1 by
+0.087 spellF1 on the flags labels (v2 +0.040), on every one of the 7 games, and by +0.049 on the event-rule labels. That is the
opposite of the SkillCorner result (v3 0.522 vs v1 0.563), which therefore looks like a domain problem (broadcast tracking, 10 fps Phases
of Play) rather than overfitting to Metrica's labels. The gain is about half of the Metrica LOGO gain (+0.145); on DFL the learned layers
cost 0.003-0.010 macroF1 and 0.017 alive/dead F1 (the A2 dead-ball model is worse than the rules there: DFL's dead-ball definition
differs); all pipelines under-segment (260 of 437 possessions).

**Step 5: retraining on Metrica 1-3 + DFL 1-7** (`scripts/dfl_retrain.py`). The v3 recipe is unchanged (features, hyper-parameters,
stacking on out-of-fold event scores, 1 thread, fixed seeds); only the training games differ. Strides scale with the number of training
runs (committed: 9 runs, A2 3 / event 2 / team 4; 27 runs: 9 / 6 / 12), keeping the row count of the committed fit; out-of-fold event
scores come from 3 game groups instead of one game each. EVERY fitted piece (A2, event model, team model) is refitted per fold; each
held-out game = its seeds 0-2. spellF1 / macroF1 / alive-dead F1, n_pred of n_true; DFL games trained AND scored with the tracking-flag
labels, Metrica with the Metrica event rules:

| held-out game | v1 (rules) | v3 recipe refitted on the other 9 games (= `integrated_v4` recipe) | same, A2 fitted on Metrica only (`integrated_v4_a2m`) |
|---|---|---|---|
| metrica1 | 0.517 / 0.761 / 0.895, 211 of 285 | 0.640 / 0.751 / 0.840, 242 | 0.673 / 0.821 / 0.924, 252 |
| metrica2 | 0.512 / 0.796 / 0.915, 217 of 275 | 0.668 / 0.794 / 0.858, 236 | 0.711 / 0.853 / 0.936, 260 |
| metrica3 | 0.549 / 0.800 / 0.917, 236 of 316 | 0.637 / 0.817 / 0.889, 261 | 0.656 / 0.817 / 0.919, 282 |
| dfl1 | 0.284 / 0.762 / 0.890, 232 of 511 | 0.475 / 0.885 / 0.968, 257 | 0.406 / 0.785 / 0.888, 277 |
| dfl2 | 0.280 / 0.722 / 0.860, 247 of 500 | 0.550 / 0.898 / 0.992, 298 | 0.450 / 0.756 / 0.851, 328 |
| dfl3 | 0.346 / 0.786 / 0.899, 220 of 402 | 0.578 / 0.904 / 0.981, 242 | 0.486 / 0.782 / 0.871, 262 |
| dfl4 | 0.354 / 0.761 / 0.881, 253 of 456 | 0.547 / 0.890 / 0.974, 276 | 0.466 / 0.767 / 0.866, 294 |
| dfl5 | 0.362 / 0.765 / 0.883, 228 of 421 | 0.575 / 0.909 / 0.982, 257 | 0.499 / 0.795 / 0.885, 272 |
| dfl6 | 0.376 / 0.730 / 0.859, 242 of 401 | 0.606 / 0.899 / 0.968, 275 | 0.540 / 0.783 / 0.878, 300 |
| dfl7 | 0.362 / 0.794 / 0.904, 220 of 370 | 0.583 / 0.904 / 0.974, 250 | 0.503 / 0.807 / 0.895, 273 |
| **mean Metrica (3)** | 0.526 / 0.786 / 0.909, 221 of 292 | 0.648 / 0.787 / 0.862, 247 | **0.680 / 0.830 / 0.926**, 265 |
| **mean DFL (7)** | 0.338 / 0.760 / 0.882, 234 of 437 | **0.559 / 0.898 / 0.977**, 265 | 0.479 / 0.782 / 0.876, 287 |
| **mean all 10** | 0.394 / 0.768 / 0.890, 230 of 394 | **0.586 / 0.865 / 0.942**, 259 | 0.539 / 0.797 / 0.891, 280 |

Reference (Metrica-only LOGO, "## Event detection"): v3 0.671 / 0.829 / 0.928, 256 of 292. The DFL games scored against the event-rule
labels instead: v1 0.354, full refit 0.423, A2-Metrica-only 0.431 spellF1.

**Leave-one-provider-out** (all pieces refitted; the Metrica-only fit is the committed recipe and reproduces the step-4 v3 row):

| train -> test | v1 | retrained | other label type |
|---|---|---|---|
| Metrica 1-3 -> DFL 1-7 (flags) | 0.338 / 0.760 / 0.882, 234 of 437 | 0.425 / 0.770 / 0.874, 261 | event-rule labels: v1 0.354, retrained 0.402 |
| DFL 1-7 (flags) -> Metrica 1-3 | 0.526 / 0.786 / 0.909, 221 of 292 | **0.614 / 0.770 / 0.847**, 244 | |

Both directions beat the rules: the learned team / event layers are not Metrica-specific (a model that never saw a Metrica frame gains
+0.088 spellF1 on Metrica, 0.614 vs 0.671 for the Metrica-trained fit).

**Which convention does the model follow?** The tracking-flag one. Evidence: the DFL event-rule twin games `dflev<k>` (same tracking,
labels from the converted events, otherwise the identical recipe; `fold ... --ev`) give clearly WORSE models: DFL held-out 0.409 / 0.770 /
0.911 (0.559 with flags on flags), Metrica held-out 0.598 (0.648 with flags, 0.680 with flags + Metrica-only A2), DFL -> Metrica 0.502,
below the rules (0.526). Even scored against the event-rule labels the flag-trained models are better (0.423 vs 0.409). So (a) the
event-rule DFL labels are noisier than the flags (events lack terminations, ball-out and fouls are inferred, a conversion step), and (b)
the flags carry a convention (7 s spells, 45% dead) unlike Metrica's event rules (14 s, 31%). Mixed training makes the dead-ball model
compromise: trained with the flags it learns DFL's dead-ball definition, loses 0.064 alive/dead F1 on Metrica (0.862 vs 0.926) and
gains 0.10 on DFL (0.977 vs 0.876, partly the artefact above). The team and event models are not hurt: retraining THEM on 10 games and
keeping the Metrica dead-ball model gives 0.680 on Metrica vs 0.671 for the Metrica-only fit (+0.009; the three games +0.00 .. +0.02).

**Registered pipelines** (new files; committed v1-v3 files and `pipelines.DEFAULT` untouched; the recipe is `integrated_v3`):
`integrated_v4` = A2 + event model + team model, all fitted on Metrica 1-3 + DFL 1-7 with flag labels
(`possession/deadball_a2_model_v4.npz` 23 KB, `possession/models/event_gbm_v4.npz` 138 KB, `team_gbm_ev_v4.npz` 328 KB;
`scripts/dfl_retrain.py final`; **retrained twice: byte-identical**, sha256 `ae194039...`, `36a3c25e...`, `74df8af0...`).
`integrated_v4_a2m` = the same event / team models with the committed Metrica-only dead-ball model (the third column above).
In-sample sanity through the numpy pipeline (metrica1 / dfl1, seed 0): v4 0.679 / 0.469, v4_a2m 0.721 / 0.406.

**BAR-ATH** (no labels read, `sanity_public` not run), `python -m possession.run`:

| pipeline | possessions | H / A / D / X | sha256 |
|---|---|---|---|
| integrated_v3 (re-checked, unchanged) | 251 | 61864 / 46668 / 50325 / 40462 | `17c7c6b6c0e21de3e41c96ae5b46ec47424ad9cf858d1a17a645726f8c39560a` |
| integrated_v4 (two runs identical) | 255 | 54070 / 39617 / **65170** / 40462 | `f17a0ae81879d9c95393cc630e2f808227e7f2d689eb70a27ad6e546f6bbab2d` |
| integrated_v4_a2m (two runs identical) | 280 | 61493 / 47642 / 49722 / 40462 | `a715cba03af39e961dbb88d95ce6112ee921bb2b6b2c98a8e2a5fbc503808e3d` |

**Reading and risks.** (1) The learned layers transfer across providers, in both directions, so v3's Metrica result was not mainly
label-convention overfitting; the SkillCorner loss is a domain effect. (2) The full 10-game refit (`integrated_v4`) predicts 41% dead
time on BAR-ATH against 32% (v3) and 31% (v4_a2m) (a statistic of OUR OUTPUT, no label involved): it has absorbed the DFL flag
convention's heavy dead time (41-52% of DFL frames) and TRACAB's frozen-ball artefact, while the BAR-ATH public summary (19.6% dead in the
first 5 minutes) points the other way. Its DFL advantage comes from the dead-ball model on DFL's own convention, and it loses 0.03
spellF1 / 0.06 alive-dead F1 on Metrica. We do NOT recommend `integrated_v4` for BAR-ATH on this evidence. (3) `integrated_v4_a2m` is the
variant the numbers support: Metrica LOGO 0.680 / 0.830 / 0.926 (v3 0.671 / 0.829 / 0.928), better than v3 on DFL (0.479 vs 0.425) with
v3's dead ball unchanged; but the Metrica gain over v3 is +0.009 (noise band 0.04) and the DFL gain mixes train-on-DFL with the flags
convention, so v3 vs v4_a2m is a judgement call, an argument for more training providers rather than a significant score
difference. (4) Training labels from other games are free; the remaining uncertainty is BAR-ATH's convention (flag-like 7 s spells vs
Metrica-like 14 s): the public 5 minutes (mean 14.2 s) say Metrica-like. (5) The DFL ball is NaN in dead runs by our rule, a realism
choice with its own leak risk (P(NaN | alive) 0.06 vs Metrica 0.08-0.19). (6) The event-rule DFL labels are a best-effort conversion,
used only as the second label type.

Parameters and origin: `DEAD_KEEP_START_S` 3.0 / `DEAD_KEEP_END_S` 1.5 (chosen so that P(NaN | dead) ~0.84 like Metrica's native track; no
BAR-ATH label involved), `GAP_FRAMES` 500, event-label windows `max_foul_dead` = `max_unlogged` = 750 (a free kick can come ~30 s after the
foul; Metrica uses 250 / 375), OOF groups 3, strides as above; everything else is the committed v3 recipe. Reproduce:
`$PY scripts/dfl_retrain.py build <game>`, `fold <game|metrica|dfl> [--ev] [--a2m]`, `report [--ev] [--a2m]`, `final [--check]`;
`scripts/dfl_transfer.py run|report`; `scripts/dfl_realism.py --calcheck`.
