"""Pipeline registry.

A pipeline is `fn(match: Match, **params) -> np.ndarray[int8]` of label codes (H=0, A=1, D=2, X=3)
over ALL match.n frames. It must be deterministic and must read nothing but the Match (so the
external-validation harness can run it on other games).

Each component adds ITS OWN file in this package and registers there:

    from possession.pipelines import register

    @register("deadball_v1")
    def run(match, **params): ...

Every module in this package is imported automatically, so no shared file needs editing and
parallel branches do not conflict. The name "default" is what `python -m possession.run` uses
when no --pipeline is given; only the integration step changes which pipeline that points to.
"""

import importlib
import pkgutil

_REGISTRY = {}
DEFAULT = "integrated_v3"


def register(name):
    def deco(fn):
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise KeyError("pipeline {!r} registered twice".format(name))
        _REGISTRY[name] = fn
        return fn
    return deco


def _discover():
    for info in pkgutil.iter_modules(__path__):
        importlib.import_module("{}.{}".format(__name__, info.name))


def get(name=None):
    _discover()
    name = name or DEFAULT
    if name not in _REGISTRY:
        raise KeyError("no pipeline {!r}; have {}".format(name, sorted(_REGISTRY)))
    return _REGISTRY[name]


def names():
    _discover()
    return sorted(_REGISTRY)
