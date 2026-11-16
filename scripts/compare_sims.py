import json
from extval.harness import evaluate_pipeline
from extval.degrade import LEGACY_V1
out = {}
for name in ("baseline_v0", "emissions_b_v1", "deadball_a_v1", "integrated_v1"):
    for label, dp in (("old", LEGACY_V1), ("new", None)):
        r = evaluate_pipeline(name, seeds=3, bootstrap=0, degrade_params=dp)
        out["%s/%s" % (name, label)] = r["mean"]
        m = r["mean"]
        print("%-16s %s spell %.4f macro %.4f alive %.4f n_pred %.0f n_gt %.0f" % (
            name, label, m["spell_f1"], m["macro_f1"], m["alive_dead_f1"], m["n_pred"], m["n_gt"]), flush=True)
json.dump(out, open("compare_sims.json", "w"), indent=1)
