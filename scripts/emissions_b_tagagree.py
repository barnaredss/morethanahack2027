"""UNLABELLED BAR-ATH statistic: agreement of each Smart Tagging phase code with the control chain
(frames inside control episodes only). Used to set the tag weight; no GT involved."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import emissions as E
from possession.data import H, A, load_match
m = load_match()
p = dict(E.PARAMS)
prep = E._prepare(m, p)
owner, conf, eps, _ = E._chain(m, p, prep)
inside = np.zeros(m.n, bool)
for t, s, e in eps: inside[s:e] = True
inside &= m.graded
print("episodes", len(eps), "inside frac of graded", round(inside.sum() / m.graded.sum(), 3))
tot = {}
for t in m.tags:
    s = E.TAG_SIGN.get(t.code)
    if s is None or t.team not in (H, A): continue
    team = t.team if s > 0 else 1 - t.team
    a, b = max(0, t.start_frame), min(m.n, t.end_frame)
    sel = inside[a:b]
    if sel.sum() == 0: continue
    agree = (owner[a:b][sel] == team).sum(); n = sel.sum()
    g = tot.setdefault(t.code, [0, 0, 0]); g[0] += agree; g[1] += n; g[2] += 1
allag = sum(v[0] for v in tot.values()); alln = sum(v[1] for v in tot.values())
for k, v in sorted(tot.items()): print("%-30s tags %3d frames %6d agree %.3f" % (k, v[2], v[1], v[0] / v[1]))
print("ALL agree %.3f over %d frames" % (allag / alln, alln))
# coverage of tags on graded frames
print("tag coverage of graded frames", round((E.tag_evidence(m)[m.graded] != 0).mean(), 3))
