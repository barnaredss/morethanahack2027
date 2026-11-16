"""integrated_v2_team: A's dead-ball detector + the LEARNED team model + C's HSMM decoder.

integrated_v2_team = hsmm_decode(match, p_dead_a(match), p_home_e(match), evidence=0.2)

`evidence` 0.10 -> 0.20: the stacked GBM is better calibrated than the hand-built emissions, so the
decoder trusts a frame more (held-out sweep on seeds 0-2: 0.10 -> 0.569, 0.20 -> 0.580, 0.30 -> 0.578;
see docs/NOTES.md "## Learned team model"). `integrated_v1` and `pipelines.DEFAULT` are untouched.
"""

from possession import deadball, decode, learned
from possession.pipelines import register


@register("integrated_v2_team")
def integrated_v2_team(match, deadball_params=None, **decoder_params):
    decoder_params.setdefault("evidence", 0.2)
    p_dead = deadball.p_dead_a(match, **(deadball_params or {}))
    p_home = learned.p_home_e(match)
    return decode.hsmm_decode(match, p_dead, p_home, **decoder_params)
