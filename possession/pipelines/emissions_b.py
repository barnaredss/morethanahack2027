"""Possession pipelines: control-chain emissions + the HSMM decoder.

emissions_b_v1       p_home_b + deadball.p_dead_ball_missing + hsmm_decode (no change points)
emissions_b_v1_cp    same, with control_events fed to the decoder as change points
The dead-ball estimator is still the Phase-0 placeholder (missing ball = dead); integration swaps
in the dead-ball module's.
"""

from possession import deadball, decode, emissions
from possession.pipelines import register


def _run(match, use_cp, p_dead_fn, emission_params, **decoder_params):
    p_dead = p_dead_fn(match)
    p_home = emissions.p_home_b(match, **emission_params)
    cps = emissions.control_events(match, **emission_params) if use_cp else None
    return decode.hsmm_decode(match, p_dead, p_home, change_points=cps, **decoder_params)


@register("emissions_b_v1")
def emissions_b_v1(match, emission_params=None, **decoder_params):
    return _run(match, False, deadball.p_dead_ball_missing, emission_params or {}, **decoder_params)


@register("emissions_b_v1_cp")
def emissions_b_v1_cp(match, emission_params=None, **decoder_params):
    return _run(match, True, deadball.p_dead_ball_missing, emission_params or {}, **decoder_params)
