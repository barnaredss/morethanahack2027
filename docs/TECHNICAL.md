# Technical document: who has the ball, frame by frame

Team: motion

This document explains what we built, what data it uses, the possession rules it follows, how well
it did in our own tests and where it can fail. Two companion files go deeper:
[PIPELINE.md](PIPELINE.md) walks through one run step by step, and [NOTES.md](NOTES.md) is our
working log, with every setting and where its value came from.

The submission is produced by one command: `python -m possession.run`.

## 1. The approach in short

For every frame of the match we have to say: home has the ball (H), away has it (A), the ball is
out of play (D), or no answer (X). We only answer X outside the two halves.

We split the problem into three questions and answer them in order:

1. **Is the ball in play?** A small model looks at each frame and estimates the chance that play is
   stopped.
2. **If it is in play, which team has it?** A set of football rules gives a first answer, and a
   second small model refines it.
3. **What is the most sensible sequence of possessions for the whole half?** A decoder takes the
   two per-frame answers and turns them into whole possessions and stoppages.

The third step matters most for the ranking score. The score counts how many real possessions we
recover as whole possessions, so a method that flickers between teams from frame to frame scores
badly even when most of its frames are right.

We could not measure anything on the competition match itself, because its labels are private and
the rules forbid tuning on the public five minutes. So every part is built to run on any match, and
we tested all of them on other matches where the answer is known (section 3).

### Step 1: is the ball in play?

The strongest hint that play has stopped is that the tracking loses the ball. But the ball is
missing on 41 % of the frames, and about a third of those are simply the tracker losing the ball
while play goes on. To tell the two apart, the model (`possession/deadball.py`) looks at:

- how long the ball has been missing, and the gaps before and after;
- whether the ball was last seen, or reappears, near a touchline or goal line;
- whether all the players slow down together, as they do after a whistle;
- whether a visible ball is outside the lines, standing still, or held next to a player;
- how many players are tracked, because sometimes the whole tracking drops out for a moment, and
  that is not a stoppage;
- whether the tactical tagging marks a set piece or a goal nearby.

A separate rule finds each kickoff from how the players move and where they stand, without using
any frame numbers, and marks the time before it as dead.

### Step 2: which team has it?

First, **football rules written by hand** (`possession/emissions.py`). They follow the challenge's
definition: a team keeps the ball until the other team gains *controlled* possession. Section 4
lists the rules. When the ball is not tracked, the rules fall back on the shape of the two teams:
which block is pushing forward, which is more compact, which is moving faster. They also use the
team attached to the tactical tags.

Second, **a model that learns the moment a team gains control** (`possession/events.py`). For every
frame and each team, it estimates the chance that the team gains control within a third of a
second: a reception, a recovery or a restart. It looks at how the ball's speed and direction change
and at how quickly the nearest player of each team closes in on the ball over the next
half-second.

Third, **a model that makes the final call** (`possession/learned.py`). It reads the answer from
the hand-written rules, the detected control moments and about thirty other measurements, and
outputs the chance that home has the ball. It cannot favour the home team by construction: every
frame is scored twice, once as it is and once with the two teams swapped, and the two results are
averaged.

### Step 3: the decoder

The decoder (`possession/decode.py`) looks at a whole half at once. It knows roughly how long
possessions and stoppages last, that a possession cannot last less than a second and a stoppage
less than three, and that a direct change from one team to the other needs solid evidence. From
that it finds the single most likely sequence of possessions and stoppages. A last pass moves each
boundary to the frame where the evidence changes most.

### Why this design

We started with rules only, because rules can be checked by reading them and there is no labelled
possession data for this match. We added the three learned models on top, one at a time, and kept
each one only because it improved the score on a match it had not been trained on (section 6).

The three models are small sets of decision trees stored in the repository (23 KB, 138 KB and
324 KB). Producing the submission needs only `numpy`, not the training data or a machine-learning
library.

## 2. What we used from the provided data, and what we dropped

| provided data | what we did with it |
|---|---|
| Tactical tracking (players, ball, speeds) | **Used.** It is the only source of positions. |
| Tactical tagging (Smart Tagging) | **Used** as a hint. Set-piece and goal tags help decide whether a ball gap is a stoppage, and the team on the phase tags is weak evidence of who has the ball. We never use tag times to place a boundary. |
| Tracking metadata (pitch, teams, attack direction) | **Used.** |
| Tactical video | **Tried and dropped.** We tested a public ball detector on a sample of frames (`video/spike_report.md`). It would fill about 30 % of the frames where the ball is missing, but roughly one detection in three looked wrong, we had no way to measure the benefit, and we could not guarantee the same output on another computer. |
| Broadcast video and tracking | **Dropped early.** It is not synchronised with the tactical video, and a third of its camera calibration is unusable. |
| Homography | **Not used** by the submitted pipeline. We only used it to draw the tracking on the video when checking results by eye. |
| Public 5-minute ground truth | **Used only as a check** that the file is well formed and lined up with the video. See section 7. |

## 3. Data from other matches

Nothing here comes from the provided match. Nobody labelled anything by hand for this entry: the
labels below are computed by code from data the providers published.

| dataset | what it is | where the labels come from | how we used it |
|---|---|---|---|
| Metrica Sports sample data, 3 matches | Full-pitch tracking at 25 frames per second, with event logs (passes, recoveries, shots, ball out) | Metrica's event logs | Training the three models, setting the decoder, and choosing between versions |
| Sportec / DFL open data, 7 Bundesliga matches (2022/23) | Full-pitch tracking at 25 frames per second. Each frame carries the provider's own "ball in play" and "team in possession" flags, plus event logs | The provider's flags (main labels) and its event logs (second labels) | A test of whether models trained on Metrica work on another provider. Not used to train the submission |
| SkillCorner open data, up to 4 of 20 matches | Tracking from the TV broadcast at 10 frames per second, about 9 players visible | SkillCorner's "phases of play" | A secondary check only |
| Statistics of the provided tracking, without labels | How often the ball is missing, how long the gaps are, how noisy the ball position is, how often players are lost | None needed | Making the other matches look like the provided tracking (below) |

**How we turn event logs into labels.** The code in `extval/labels.py` follows the challenge's
definition. The ball is dead from a ball-out, foul or goal until the restart. Otherwise the ball
belongs to the team that last had controlled possession, and touches and deflections are ignored.
One choice in it is a judgement: an opponent's action that comes less than 1 second after the
previous one counts as a touch, not as control. With 2 seconds instead, each match has about 246
possessions instead of 292, so the exact scores depend on this choice.

**Making clean tracking look like ours.** The other providers' tracking is much cleaner than the
tracking we were given. Testing on clean data would flatter any method, so we degrade it first
(`extval/degrade.py`). We hide the ball until it is visible on 59 % of frames, as in the provided
match, with gaps of realistic length. We also add errors to the ball position, drop and re-create
player tracks, add duplicate players, and add moments when the whole tracking drops out. Every
amount is matched to statistics measured on the provided tracking, which needs no labels. The
tactical tags cannot be matched that way, so the simulated tags (8 % missing, 8 % with the wrong
team, shifted in time) are our assumption.

**Licences and credit.**
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data. It has no licence
  file; its README asks users to act responsibly and to credit the source.
- SkillCorner open data: https://github.com/SkillCorner/opendata, MIT licence.
- Sportec / DFL data: CC-BY 4.0. Bassek, M., Rein, R., Weber, H., Memmert, D., *An integrated
  dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025),
  doi:10.1038/s41597-025-04505-y; data doi:10.6084/m9.figshare.28196177. The data belongs to the
  Deutsche Fußball Liga.

None of this data is in our repository. The scripts download it into a folder that git ignores, and
the submission does not need it. The trained models can be rebuilt with
`scripts/train_deadball_a2.py` and `scripts/train_events.py final`; training the same model twice
gives an identical file.

## 4. The possession rules we implemented

- **Control.** A team controls the ball on a frame when its nearest player is within 3 m of the
  ball, the nearest opponent is at least 1 m farther away, and the ball moves together with that
  player. When both teams are close, nobody is given control. The 3 m is generous on purpose: in
  the provided tracking the ball sits about 1.9 m from the nearest player on a typical frame.
- **A touch is not control.** Control has to last at least 0.4 seconds. Anything shorter is treated
  as a touch or a deflection and does not change possession.
- **Loose balls and duels.** A loose or contested ball stays with the last team that controlled
  it. So a tackle, a header won in a duel or a deflection changes nothing unless that team then
  controls the ball. The longer since the last control, the less sure the rules are.
- **When the ball is not tracked.** If the same team controlled the ball before and after the gap,
  it keeps it. If the teams differ, possession changed somewhere in the gap. For gaps up to 3
  seconds we estimate where from the ball's path. For longer gaps we rely on the shape of the two
  teams and on the tags.
- **Restarts.** After a stoppage either team may restart. The team is decided by what happens
  once the ball is back in play, usually next to the players of the team taking the restart.
- **Durations.** A possession lasts at least 1 second and a stoppage at least 3. Typical lengths,
  about 8 seconds for a possession and 13 for a stoppage, come from the other matches.
- **The learned models add no rules of their own.** They learn from the labels of section 3, so
  where they overrule the hand-written rules, they follow those labels.
- **No answer (X)** is written only outside the two halves. Inside them we always answer.

## 5. Missing and noisy tracking

- **Missing ball.** The ball is missing on 41 % of the graded frames, and a typical gap lasts about
  11 seconds. Step 1 decides whether a gap is a stoppage, step 2 decides who has the ball after
  it, and the decoder prevents short flickers. A frame with no information counts as "no
  evidence", not as a vote for any answer.
- **Noisy ball position.** We do not smooth the ball. The control rule is tolerant instead (3 m),
  and the models were trained on matches with ball errors added.
- **Players.** The tracking has no player identities: its 46 track numbers are reused when a
  player is lost. We only use where the players of each team are on each frame, never who they
  are. About 10 % of the speeds are empty, and we compute those from the positions.
- **Not handled.** A track that jumps from one player to another, or from one team to the other.

## 6. Results in our own tests

We cannot score ourselves on the competition match. The numbers below come from the three Metrica
matches, degraded as described in section 3. To keep them honest, each match is scored by models
trained **only on the other two**.

The ranking score is `spellF1`: the share of real possessions recovered as whole possessions.
`macroF1` is the per-frame score used as tie-break.

| match held out | simplest baseline (nearest player) | rules only | + learned dead ball and team model | **+ control-moment detector (submitted)** |
|---|---|---|---|---|
| Metrica 1 | | 0.517 | 0.589 | **0.651** |
| Metrica 2 | | 0.512 | 0.627 | **0.709** |
| Metrica 3 | | 0.549 | 0.593 | **0.652** |
| **average spellF1** | 0.288 | 0.526 | 0.603 | **0.671** |
| **average macroF1** | 0.667 | 0.786 | 0.824 | **0.829** |

What the table shows:
- Each added part improved the score on all three matches.
- The last part, the control-moment detector, added 0.068. On a second set of degraded copies that
  we had not used for any decision, it added 0.084 (0.592 to 0.677).
- That gain comes from finding more of the short possessions: the submitted version finds 256
  possessions per match where 292 exist, against 223 without the detector.
- The detector does not depend on the simulated tags: with all tags removed it still adds 0.069.
- On the competition match, the submitted pipeline outputs 251 possessions and 32 % dead time.

**Does it work on other providers?** We ran the same models, trained on Metrica only, on matches
from two other providers.

| data | rules only | submitted |
|---|---|---|
| 7 Bundesliga matches, degraded, labels from the provider's own tracking flags | 0.338 | **0.425** |
| the same matches, labels from the event logs | 0.354 | **0.403** |
| SkillCorner, 4 matches (TV-broadcast tracking) | **0.563** | 0.522 |

The Bundesliga test is the closest we have to the competition. It is another provider and league
with full-pitch tracking, and the labels come from the tracking's own "ball dead" and "owning
team" flags, which is how the challenge says its ground truth is built. The submitted pipeline
beat the rules on all 7 matches. On SkillCorner it did worse than the rules. That data is very
different (TV broadcast, 10 frames per second, about 9 players visible), and we think that is the
reason, but we cannot prove it.

**A version we did not submit.** We also retrained the models on all 10 matches. That version
learned the Bundesliga convention, in which possessions are short (about 7 seconds) and 45 % of the
time is dead, and it predicted 41 % dead time on the competition match. The organisers' published
summary of the first five minutes shows long possessions (14 seconds on average), much closer to
Metrica's. On held-out Metrica matches the retrained version was no better (0.680 against 0.671,
which is within noise). So we kept the Metrica-trained version.

## 7. Limitations

**Expect a lower score on the real match than the 0.67 above.** The main reasons:

- **We do not know the labelling convention of the real ground truth.** Where exactly a touch
  becomes control, and how much time counts as dead, changes the labels a lot. On the Bundesliga
  matches, labels built from the tracking flags and labels built from the event logs disagree on a
  quarter of the frames. Our models follow Metrica's convention. If the real ground truth draws
  the line elsewhere, our score drops, and we cannot check this without labels.
- **The models learned from only three matches.** They did transfer to the Bundesliga matches, but
  with about half the gain (0.087 against 0.145).
- **Our tests run on imitation data.** The degraded matches imitate the provided tracking, but not
  perfectly. On them, the simplest baseline scores 0.288, while the organisers report 0.2433 for it
  on the real match, so our test is somewhat too easy. One known gap: in the provided tracking the
  ball goes missing during live play about twice as often as in our degraded matches.
- **Possessions are still missed.** We find about 88 % as many possessions as exist. In our tests
  most of the missed changes of possession happen while the ball is tracked, not while it is
  missing.
- **Some stoppages are missed.** Mainly those where the ball stays visible and moving, such as a
  player fetching the ball. Some tracking dropouts in midfield are still mistaken for stoppages.
- **Kickoff detection is a rule of thumb.** It can fire late when a half starts slowly.
- **Reproduction was checked on Windows only** (section 8).

**Two corrections we made before submitting.** While building the pipeline, the kickoff rule's
settings had been adjusted while looking at this match's second-half kickoff, and four other
settings had been justified by pointing at the public five minutes. We did not think either was
acceptable under the rules, so we set all of them again using only the other matches. This changed
87 of the 199,319 frames. We read the public five minutes to understand what the labels mean, as
the challenge allows, and we used them to check that our file is well formed and lined up. No
version or setting was chosen by its score on them.

**With more time** we would train on many more matches and providers, for example PFF's open World
Cup 2022 data with 64 matches. We would also train a ball detector on the tactical camera view to
fill the gaps where the ball is missing.

## 8. How to reproduce

Python 3.12. From a fresh copy of the repository:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
export MTAH_DATA_DIR=/path/to/data                # the organisers' data; default is ./data
.venv/Scripts/python -m possession.run --out submissions/motion.csv
.venv/Scripts/python evaluation/validate_submission.py submissions/motion.csv
```

- The run takes under a minute on a laptop.
- It gives the same file every time. Nothing in it is random.
- It prints the file's SHA-256. For this version:
  `17c7c6b6c0e21de3e41c96ae5b46ec47424ad9cf858d1a17a645726f8c39560a`
- We reproduced that file from a fresh copy of the repository, with a new environment and an empty
  cache, on Windows. We have not checked Linux or macOS.
- `python -m pytest -q` runs the 71 tests.
- To repeat our tests on other matches: `python -m extval.datasets` downloads them (several GB),
  then `python -m extval.harness --pipeline integrated_v3` scores the pipeline.
- To retrain the models: `pip install -r requirements-train.txt`, then the two training scripts
  named in section 3.
