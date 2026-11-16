# How this entry was built

A short account of the order the work was done in. The method itself is described in
[TECHNICAL.md](TECHNICAL.md), and every parameter with its origin in [NOTES.md](NOTES.md).

## Starting decisions

| decision | choice |
|---|---|
| inputs | tactical tracking (ATD), Smart Tagging and homography. The broadcast feed was dropped (not synchronised, a third of its homography unusable); tactical-video ball detection was tried later and dropped |
| tuning and validation | external games with tracking and labels, degraded to look like BAR-ATH's tracking. The public five minutes are a format and alignment check only |
| runtime | Python 3.12, pinned dependencies, deterministic output |

## Order of work

1. **Foundation.** A fast loader for the tracking, tags and homography; fixed interfaces between
   the dead-ball estimator, the possession estimator and the decoder; a pipeline registry. Checked
   by reproducing the organisers' `baseline_smooth50.csv` byte for byte.
2. **External validation** (`extval/`). Metrica's three open sample games and SkillCorner's open
   data, labelled mechanically from their event and phase data, degraded with noise calibrated on
   BAR-ATH's unlabelled tracking (ball visibility, gaps, ball error, outages, track fragmentation).
   Checked against the organisers' published baseline scores.
3. **Rules-based pipeline** (`integrated_v1`). A dead-ball model, a "last team to control the
   ball" chain with team-shape fallback, and an explicit-duration HMM decoder.
4. **Learned layers** (`integrated_v2`, `integrated_v3`). A learned dead-ball model, a team model
   stacked on the chain, and a control-onset event detector, each kept only after it improved
   spellF1 on games it was not trained on.
5. **Audit and packaging.** Parameters that cited the public window or BAR-ATH-specific behaviour
   were re-derived from external games; technical document, fresh-clone reproduction check, scan
   for match data.
6. **Further checks.** The simulator's ball error was corrected to match the real ball's
   smoothness; video ball detection, gap mining and ball-drift cleaning were measured and dropped;
   seven open Bundesliga matches (Sportec / DFL) were added as a transfer test.
