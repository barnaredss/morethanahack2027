"""integrated_v4: the integrated_v3 recipe refitted on Metrica 1-3 + DFL/IDSSE 1-7.

integrated_v4 = hsmm_decode(match, p_dead_a2(match, A2 v4), p_home_e3(match, event v4, team v4), evidence=0.2)

Same features, same hyper-parameters, same stacking on out-of-fold event scores as integrated_v3
(scripts/train_events.py); only the training games differ (scripts/dfl_retrain.py final). New model
files, the committed v1-v3 files are untouched: possession/deadball_a2_model_v4.npz,
possession/models/event_gbm_v4.npz, possession/models/team_gbm_ev_v4.npz.
See docs/NOTES.md "## DFL / IDSSE data". pipelines.DEFAULT is unchanged.
"""

import os

from possession import deadball, decode, events, learned
from possession.pipelines import register

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A2_V4 = os.path.join(_HERE, "deadball_a2_model_v4.npz")
EV_V4 = os.path.join(_HERE, "models", "event_gbm_v4.npz")
TEAM_V4 = os.path.join(_HERE, "models", "team_gbm_ev_v4.npz")


@register("integrated_v4")
def integrated_v4(match, **decoder_params):
    decoder_params.setdefault("evidence", 0.2)
    base = learned.features_eo(match)
    p_dead = deadball.p_dead_a2(match, model=deadball.load_a2_model(A2_V4))
    p_home = events.p_home_e3(match, ev_model=events.load_model(EV_V4),
                              team_model=learned.load_model(TEAM_V4), base=base)
    return decode.hsmm_decode(match, p_dead, p_home, **decoder_params)


@register("integrated_v4_a2m")
def integrated_v4_a2m(match, **decoder_params):
    """integrated_v4 with the COMMITTED dead-ball model (A2 fitted on Metrica only). Evaluated as `--a2m`
    in scripts/dfl_retrain.py: the DFL flag labels teach the dead-ball model TRACAB's frozen-ball artefact
    and a different dead-ball convention, which costs Metrica alive/dead F1 (see NOTES)."""
    decoder_params.setdefault("evidence", 0.2)
    base = learned.features_eo(match)
    p_home = events.p_home_e3(match, ev_model=events.load_model(EV_V4),
                              team_model=learned.load_model(TEAM_V4), base=base)
    return decode.hsmm_decode(match, deadball.p_dead_a2(match), p_home, **decoder_params)
