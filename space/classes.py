"""The timing classes as waveforms: each stored class of the time-segment layout is ONE waveform shape (a square-pulse
PGSE, a stimulated echo, a cosine OGSE with its edges ramped), and a shell on it plays that shape at the amplitude
its b-value needs, along each of its directions. The shape is built once from the configuration's ``[shapes]`` entry
by dmipy-sim's own sequence builders (:func:`template`), on the layout's grid (TE 53.5 ms, 1000 samples), and checked
against the layout's stored partition when the layout loads; :func:`play` scales it to the rows. A class has a fixed
shape: an OGSE built at b = 1000 s/mm^2 with 200 T/m/s edges plays the same edge durations at every other b (the
layout stores that shape, nothing else)."""
from __future__ import annotations

import functools

import numpy as np

KINDS = ("pgse", "pgste", "ogse")
N_T = 1000                                           # the layout's grid: 1000 samples over the echo time


def kind(sh):
    """The class's pulse kind: ``pgse`` (the default), ``pgste`` or ``ogse``."""
    k = sh.get("kind", "pgse")
    if k not in KINDS:
        raise ValueError(f"a timing class's kind is one of {KINDS}, got {k!r}")
    return k


def _key(sh):
    return tuple(sorted((k, v) for k, v in sh.items() if k != "label"))


@functools.lru_cache(maxsize=None)
def _template(key):
    import dmipy_sim as d
    sh = dict(key)
    k = kind(sh)
    if k == "pgse":
        return d.pgse([[0.0, 0.0, 1.0]], float(sh["delta"]), float(sh["Delta"]), gradient_strengths=0.1, TE=float(sh["TE"]),
                      n_t=N_T, slew_rate=np.inf)
    if k == "pgste":                                  # TM = Delta - delta between the 90s; TE = 2 delta + TM
        return d.pgste([[0.0, 0.0, 1.0]], float(sh["delta"]), float(sh["Delta"]) - float(sh["delta"]), gradient_strengths=0.1,
                       n_t=N_T, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    return d.ogse([[0.0, 0.0, 1.0]], float(sh["frequency"]), float(sh["duration"]), shape="cosine",
                  bvalues=[float(sh["b_built"]) * 1e6], TE=float(sh["TE"]), n_t=N_T, slew_rate=float(sh["slew_rate"]))


def template(sh):
    """The class's waveform along z, one row (a ScannerSequence): ``pgse`` square lobes ``delta`` wide ``Delta`` apart
    at ``TE``; ``pgste`` square lobes storing for ``TM = Delta - delta``; ``ogse`` a cosine of ``frequency`` (Hz) over
    ``duration`` (s) on each side of the 180, built at ``b_built`` (s/mm^2) with ``slew_rate`` (T/m/s) edges."""
    return _template(_key(sh))


def unit_b(sh):
    """``(profile (n_t,), b per (T/m)^2)``: the class's physical gradient along its axis normalised to a peak of 1, and
    the b-value (s/m^2) of that shape at a peak amplitude of 1 T/m."""
    from dmipy_sim.acquisition.waveforms import calc_b
    t = template(sh)
    G = np.asarray(t.G, np.float64)[0, :, 2]
    peak = float(np.abs(G).max())
    return G / peak, float(calc_b(t)[0]) / peak ** 2


def amplitude(sh, b):
    """The peak gradient (T/m) the class plays to reach ``b`` (s/mm^2)."""
    return float(np.sqrt(max(float(b), 0.0) * 1e6 / unit_b(sh)[1]))


def play(sh, bvals_SI, dirs):
    """The ScannerSequence of rows on the class: its shape at each row's amplitude along each row's unit direction
    (``bvals_SI`` in s/m^2, ``dirs`` ``(n, 3)``); a b = 0 row plays nothing."""
    from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
    t = template(sh)
    prof, unit = unit_b(sh)
    b = np.asarray(bvals_SI, np.float64).ravel()
    u = np.asarray(dirs, np.float64).reshape(-1, 3)
    n = np.linalg.norm(u, axis=1)
    u = np.where(n[:, None] > 0, u / np.where(n > 0, n, 1.0)[:, None], 0.0)
    amp = np.sqrt(np.maximum(b, 0.0) / unit)
    G = prof[None, :, None] * (amp[:, None] * u)[:, None, :]
    return ScannerSequence(G=G.astype(np.float32), dt=t.dt, rf=t.rf, readout=t.readout)


def describe(sh):
    """The class's timing in a few words, for the gradient table."""
    k = kind(sh)
    if k == "ogse":
        return f"cosine {float(sh['frequency']):g} Hz, {float(sh['duration']) * 1e3:g} ms per side"
    if k == "pgste":
        return f"STE δ {float(sh['delta']) * 1e3:g} / Δ {float(sh['Delta']) * 1e3:g} ms"
    return f"δ {float(sh['delta']) * 1e3:g} / Δ {float(sh['Delta']) * 1e3:g} ms"
