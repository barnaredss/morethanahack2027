"""integrated_v3: A2 dead ball + event-aware team model + HSMM decoder.

integrated_v3 = hsmm_decode(match, p_dead_a2(match), events.p_home_e3(match), evidence=0.2)

p_home_e3 is the learned team model with extra columns derived from a learned detector of control
onsets (possession/events.py). Committed models: possession/models/event_gbm.npz and team_gbm_ev.npz.
integrated_v2 and pipelines.DEFAULT are untouched. See docs/NOTES.md "## Event detection".
"""

from possession import deadball, decode, events, learned
from possession.pipelines import register


@register("integrated_v3")
def integrated_v3(match, **decoder_params):
    decoder_params.setdefault("evidence", 0.2)
    base = learned.features_eo(match)          # shared by the detector and the team model
    return decode.hsmm_decode(match, deadball.p_dead_a2(match), events.p_home_e3(match, base=base),
                              **decoder_params)
