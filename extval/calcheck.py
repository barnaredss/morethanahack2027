"""Calibration check: baseline_v0 at windows 0/25/50/125 on the degraded externals vs the PUBLISHED
full-match scores (spellF1 / macroF1). Uses only published numbers, never BAR-ATH labels.

    $PY -m extval.calcheck ['{"error_fit": "v1", ...}']      # optional degrade params as JSON
"""
import sys, json
from extval.harness import evaluate_pipeline
dp = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
pub={0:(0.0486,0.6532),25:(0.1996,0.6752),50:(0.2433,0.6724),125:(0.2242,0.6356)}
print("window | ext spell/macro | published | n_pred")
for w,(ps,pm_) in pub.items():
    r=evaluate_pipeline("baseline_v0",{"window":w},seeds=3,bootstrap=0,degrade_params=dp)["mean"]
    print("%4d | %.4f / %.4f | %.4f / %.4f | %.0f"%(w,r["spell_f1"],r["macro_f1"],ps,pm_,r["n_pred"]),flush=True)
