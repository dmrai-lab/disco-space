"""The replay sources a Space serves, chosen by the configuration's ``source``: ``disco`` (the DiSCo phantom, a
:class:`~space.sources.disco.Layout` in demo mode or :class:`~space.sources.disco.Columns` in full mode) or ``brain``
(:class:`~space.sources.brain.Brain`, a brain composed from an FOD field, tissue fractions and replay packs)."""
from .. import pipeline as P

SOURCES = ("disco", "brain")


def source_class(cfg):
    """The :class:`space.pipeline.Source` subclass the configuration names; its class methods are what the page is
    built from before any data loads."""
    name = cfg["source"]
    if name not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {name!r}")
    if name == "brain":
        from .brain import Brain
        return Brain
    from .disco import Columns, Layout
    return Columns if P.mode(cfg) == "full" else Layout


def source(cfg, *, local=None):
    """The source the configuration names, loaded (``local``: a local copy of its data instead of the Hub's)."""
    return source_class(cfg).load(cfg, local=local)
