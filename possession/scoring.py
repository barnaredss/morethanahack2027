"""Thin wrapper over the official metric library, for code that holds label arrays, not files.

Used by scripts/sanity_public.py (BAR-ATH public window, sanity ONLY) and by the external
validation harness (extval/), which is where every model-selection decision is made.
"""

import os
import sys

import numpy as np

from .data import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

HEADLINE = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive",
            "n_gt_spells", "n_pred_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s")


def score(gt_codes, pred_codes, bootstrap_n=0):
    """Full pm.evaluate() result dict. Both arrays are int codes H=0 A=1 D=2 X=3, same length."""
    gt = np.asarray(gt_codes, dtype=np.int8)
    pred = np.asarray(pred_codes, dtype=np.int8)
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def headline(result):
    return {k: result[k] for k in HEADLINE}


def read_codes(path):
    return pm.label_codes(pm.read_label_csv(path))
