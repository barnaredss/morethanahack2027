"""integrated_v2: the learned dead-ball model (A2) + the learned team model + the HSMM decoder.

integrated_v2 = hsmm_decode(match, p_dead_a2(match), p_home_e(match), evidence=0.2)

Both models are the committed ones (possession/models/). `integrated_v1`, `integrated_v2_team`,
`integrated_v2_dead` and `pipelines.DEFAULT` are untouched. See docs/NOTES.md "## Combination LOGO".
"""

from possession import deadball, decode, learned
from possession.pipelines import register


@register("integrated_v2")
def integrated_v2(match, **decoder_params):
    decoder_params.setdefault("evidence", 0.2)
    return decode.hsmm_decode(match, deadball.p_dead_a2(match), learned.p_home_e(match), **decoder_params)
