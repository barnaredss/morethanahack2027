"""Dead-ball pipelines: the dead-ball detector plugged into the shared emission and decoders."""

from possession import deadball, decode, emissions
from possession.pipelines import register


@register("deadball_a_v1")
def deadball_a_v1(match, **params):
    p_dead = deadball.p_dead_a(match)
    p_home = emissions.p_home_nearest(match)
    return decode.hsmm_decode(match, p_dead, p_home, **params)


@register("deadball_a_v1_majority")
def deadball_a_v1_majority(match, window=50):
    p_dead = deadball.p_dead_a(match)
    p_home = emissions.p_home_nearest(match)
    return decode.majority_decode(match, p_dead, p_home, window=window)


@register("integrated_v2_dead")
def integrated_v2_dead(match, emission_params=None, deadball_params=None, **decoder_params):
    """integrated_v1 with the second dead-ball model (docs/NOTES.md "## Dead ball v2")."""
    p_dead = deadball.p_dead_a2(match, **(deadball_params or {}))
    p_home = emissions.p_home_b(match, **(emission_params or {}))
    return decode.hsmm_decode(match, p_dead, p_home, **decoder_params)
