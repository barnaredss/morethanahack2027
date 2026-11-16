"""Integration pipeline: the dead-ball detector + the possession emissions + the decoder.

integrated_v1   hsmm_decode(match, p_dead_a(match), p_home_b(match), **decoder_params)

Defaults of all three components (A's logistic weights, B's SHAPE_W, the decoder's HSMM_DEFAULTS)
were fixed jointly on external validation with leave-one-game-out checks, see docs/NOTES.md
"## Integration" and scripts/tune_integrated.py.
"""

from possession import deadball, decode, emissions
from possession.pipelines import register


@register("integrated_v1")
def integrated_v1(match, emission_params=None, deadball_params=None, **decoder_params):
    p_dead = deadball.p_dead_a(match, **(deadball_params or {}))
    p_home = emissions.p_home_b(match, **(emission_params or {}))
    return decode.hsmm_decode(match, p_dead, p_home, **decoder_params)
