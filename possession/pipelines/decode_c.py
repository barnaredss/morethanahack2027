"""Decoder pipelines: placeholder estimators + the HSMM decoder.

decode_c_v1 uses the Phase-0 placeholder estimators (`p_dead_ball_missing` treats every missing ball
as dead, which is wrong; the real ones come from the dead-ball and possession modules). It exists to exercise the
decoder end to end; the decoder takes whatever estimators the integration step plugs in.
"""

from possession import deadball, decode, emissions
from possession.pipelines import register


@register("decode_c_v1")
def decode_c_v1(match, **params):
    p_dead = deadball.p_dead_ball_missing(match)
    p_home = emissions.p_home_nearest(match)
    return decode.hsmm_decode(match, p_dead, p_home, **params)
