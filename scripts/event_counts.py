import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extval.datasets import load_game
from extval import labels
import numpy as np
from possession.data import H, A, D
for g in ("metrica1","metrica2","metrica3"):
    gm = load_game(g)
    c = collections.Counter(e.type for e in gm.events)
    print(g, "n_events", len(gm.events), dict(c))
    sub = collections.Counter((e.type, e.sub) for e in gm.events if e.type in ("SET PIECE","CARRY","RECOVERY"))
    print("  ", dict(sub))
    gt = labels.derive(gm)
    m = gm.match
    n_on = 0; on_by_prev_dead = 0
    for _, a, b in m.half_blocks():
        s = gt[a:b+1]
        for t in range(1, len(s)):
            if s[t] in (H, A) and s[t] != s[t-1]:
                n_on += 1
                if s[t-1] == D: on_by_prev_dead += 1
    print("  alive onsets", n_on, "after dead", on_by_prev_dead, "turnovers", n_on-on_by_prev_dead)
