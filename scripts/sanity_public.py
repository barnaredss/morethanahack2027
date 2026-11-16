"""SANITY CHECK ONLY -- score pipelines on the public 5-minute window of BAR-ATH.

    python scripts/sanity_public.py                       # every registered pipeline
    python scripts/sanity_public.py baseline_v0 mine_v2

RULES.md: the public window is for checking that a pipeline is *correct* (aligned, well-formed,
labels meaning what we think). It is NOT a tuning set and must never be used to choose between
two versions or to set a parameter. Selection happens on external validation (extval/) only.
17 possessions cannot rank anything; one miss moves spellF1 by ~6 points.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import pipelines  # noqa: E402
from possession.data import load_match  # noqa: E402
from possession.scoring import headline, read_codes, score  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def main(names):
    gt = read_codes(GT)
    match = load_match()
    names = names or pipelines.names()
    print("SANITY ONLY - public 5 min, 17 possessions. Do not select or tune on this.\n")
    print("{:<28} {:>8} {:>8} {:>8} {:>6} {:>6}".format(
        "pipeline", "spellF1", "macroF1", "aliveF1", "poss", "gt"))
    for name in names:
        r = headline(score(gt, pipelines.get(name)(match)))
        print("{:<28} {:>8.4f} {:>8.4f} {:>8.4f} {:>6} {:>6}".format(
            name, r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["n_pred_spells"],
            r["n_gt_spells"]))


if __name__ == "__main__":
    main(sys.argv[1:])
