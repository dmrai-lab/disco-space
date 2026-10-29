"""The b-tensor encoding class: three gradient shapes psi_1..3 on the spin echo's two encoding windows, stored as one
class of the layout (six terms: each shape before and after the 180), and the rows that play any b-tensor of them.

A row plays ``G(t) = sum_k psi_k(t) a_k``, the 3x3 level matrix ``A`` (row k the vector shape k is played along), so
its b-tensor is ``B = A^T M A`` with ``M`` the shapes' Gram matrix (``M_kl = int k_k k_l dt``, the moments to the
readout). Every encoding is a choice of ``A``: spherical ``A = sqrt(b/3) M^-1/2 R^T`` (``B = b/3 I`` at any rotation
``R``), planar ``A = sqrt(b/2) M^-1/2 [u; v; 0]`` (``B = b/2 (u u^T + v v^T)``), linear ``A = sqrt(b) c d^T`` with ``c``
the shape mix of the most b per peak gradient (``c^T M c = 1``; ``B = b d d^T``). The layout reads every row from
the same stored vectors, exactly as a pulse pair's rows are read: the encoding is exact in the substrate (the mean of
the walkers' phase factors), not a Gaussian approximation.

The waveform is ``[tensor] waveform`` in ``data/``: the physical gradient ``(1000, 3)`` (T/m) on the layout's
spin-echo grid (TE 53.5 ms), axis k the shape psi_k; the page checks it against the stored partition at load.

The maps: MD the first cumulant of the spherical powder average, and the microscopic fractional anisotropy from the
linear against the spherical powder average (Lasic et al. 2014, Front Phys 2:11; Szczepankiewicz et al. 2016,
NeuroImage 125:653): ``ln(S_LTE/S_STE) = (2/5) b^2 (D_iso D_Delta)^2`` to second order, fitted over the b-values with a
cubic term, and muFA is the FA of the axisymmetric microscopic tensor with that ``D_iso`` and ``D_Delta``."""
from __future__ import annotations

import functools
import os

import numpy as np

from . import pipeline as P

ENCODINGS = ("LTE", "PTE", "STE")
B_DELTA = {"LTE": 1.0, "PTE": -0.5, "STE": 0.0}
N_T = 1000
TE = 0.0535


def config(cfg):
    """The configuration's ``[tensor]`` table (None without one)."""
    return cfg.get("tensor")


@functools.lru_cache(maxsize=None)
def _waveform(path):
    G = np.load(path)
    if G.shape != (N_T, 3) or not np.isfinite(G).all():
        raise ValueError(f"the b-tensor waveform {path} is not a finite (1000, 3) gradient")
    return np.asarray(G, np.float64)


def waveform(tc):
    """``(1000, 3)``: the class's physical gradient (T/m), axis k the shape psi_k at its designed amplitude."""
    return _waveform(os.path.join(P.DATA_DIR, tc["waveform"]))


@functools.lru_cache(maxsize=None)
def _grid():
    import dmipy_sim as d
    return d.pgse([[0.0, 0.0, 1.0]], 0.0102, 0.0167, gradient_strengths=0.1, TE=TE, n_t=N_T, slew_rate=np.inf)


def _sequence(G):
    """``G (n, 1000, 3)`` on the spin-echo grid, RF schedule and readout of the layout's group g0."""
    from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
    g = _grid()
    return ScannerSequence(G=np.asarray(G, np.float32), dt=g.dt, rf=g.rf, readout=g.readout)


def template(tc):
    """The class as the layout stores it: three rows, row k shape psi_k along axis k."""
    P_ = waveform(tc)
    G = np.zeros((3, N_T, 3))
    for k in range(3):
        G[k, :, k] = P_[:, k]
    return _sequence(G)


def gram(tc):
    """``M (3, 3)`` (s/m^2 per (T/m)^2 of level): the b-tensor of ``sum_k psi_k a_k`` is ``A^T M A``."""
    from dmipy_sim.acquisition.waveforms import btensor_from_gradient
    g = _grid()
    G = waveform(tc)[None] * np.asarray(g.effective_gate, np.float64)[None, :, None]
    return btensor_from_gradient(G, g.dt)[0]


def _inv_sqrt(M):
    w, V = np.linalg.eigh(M)
    return V @ np.diag(w ** -0.5) @ V.T


def lte_mix(tc, n=4000):
    """``c (3,)`` with ``c^T M c = 1``: the shape mix whose single waveform ``sum_k c_k psi_k`` gives the most b per
    squared peak gradient (searched over ``n`` Fibonacci directions of the sphere of mixes)."""
    M = gram(tc); W = waveform(tc)
    c = fibonacci(n, hemisphere=True)
    peak = np.abs(W @ c.T).max(axis=0)
    gain = np.einsum("nk,kl,nl->n", c, M, c) / peak ** 2
    best = c[int(np.argmax(gain))]
    return best / np.sqrt(best @ M @ best)


def fibonacci(n, *, hemisphere=True):
    """``n`` unit vectors spread on the (upper half-) sphere, deterministic."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n if hemisphere else 1.0 - 2.0 * i / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z ** 2, 0.0, None))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def frame(d):
    """``R (3, 3)`` orthonormal with first column ``d`` (the encoding's axis), the rest completing it."""
    d = np.asarray(d, np.float64); d = d / np.linalg.norm(d)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - (ref @ d) * d; u /= np.linalg.norm(u)
    return np.stack([d, u, np.cross(d, u)], axis=1)


def levels(tc, encoding, b_SI, d):
    """``A (3, 3)``: the level matrix of one row of ``encoding`` at ``b_SI`` (s/m^2) about the axis ``d`` (the
    direction of a linear encoding, the normal of a planar one, the first axis of a spherical one's rotation)."""
    M = gram(tc); Mi = _inv_sqrt(M); R = frame(d)
    if encoding == "LTE":
        return np.sqrt(b_SI) * np.outer(lte_mix(tc), R[:, 0])
    if encoding == "PTE":
        return np.sqrt(b_SI / 2.0) * Mi @ np.stack([R[:, 1], R[:, 2], np.zeros(3)])
    if encoding == "STE":
        return np.sqrt(b_SI / 3.0) * Mi @ R.T
    raise ValueError(f"an encoding is one of {ENCODINGS}, got {encoding!r}")


def rows(tc, bvals, n_dirs):
    """The acquisition: one b = 0 row, then for each b (s/mm^2) ``n_dirs`` rows of each encoding (their axes spread on
    the hemisphere). Returns ``(sequence, encoding (n,), b (n,) s/mm^2)``."""
    W = waveform(tc)
    dirs = fibonacci(int(n_dirs))
    G = [np.zeros((N_T, 3))]; enc = ["b0"]; bs = [0.0]
    for b in bvals:
        for e in ENCODINGS:
            for d in dirs:
                G.append(W @ levels(tc, e, float(b) * 1e6, d)); enc.append(e); bs.append(float(b))
    return _sequence(np.stack(G)), np.array(enc), np.array(bs)


def check(tc, seq, enc, bs):
    """The rows' b-tensors as built: ``{encoding: (max |b - b_target| / b, max |b_delta - target|)}`` from
    dmipy-sim's own b-tensor of the sequence, and the peak gradient per axis (T/m) of every row."""
    from dmipy_sim.acquisition.waveforms import btensor_invariants, calc_btensor
    B = calc_btensor(seq)
    b, b_delta, _ = btensor_invariants(B)
    out = {}
    for e in ENCODINGS:
        m = enc == e
        if m.any():
            out[e] = (float(np.max(np.abs(b[m] / 1e6 - bs[m]) / bs[m])), float(np.max(np.abs(b_delta[m] - B_DELTA[e]))))
    return out, np.abs(np.asarray(seq.G, np.float64)).max(axis=(1, 2))


def powder(S, enc, bs, bvals):
    """``{(encoding, b): S/S0 averaged over the encoding's rows at b}`` per voxel, ``S`` ``(..., n)`` normalised to its
    b = 0 row."""
    return {(e, float(b)): np.nanmean(S[..., (enc == e) & (bs == float(b))], axis=-1) for e in ENCODINGS for b in bvals}


def md_ufa(pw, bvals):
    """``(MD (um^2/ms), muFA)`` per voxel from the powder averages: MD the first cumulant of the spherical signal,
    ``ln S_STE = -b MD + b^2 V / 2`` fitted over the b-values (``-ln S_STE / b`` at one b); muFA from
    ``ln(S_LTE/S_STE) = (2/5) b^2 (D_iso D_Delta)^2 + c b^3`` fitted over the b-values (the b^2 term alone at one or
    two), the FA of the axisymmetric microscopic tensor with that ``D_iso = MD`` and ``D_Delta``; NaN where a signal is
    not positive."""
    b = np.asarray(sorted(float(x) for x in bvals)) * 1e-3                 # s/mm^2 -> ms/um^2
    with np.errstate(divide="ignore", invalid="ignore"):
        ste = np.stack([np.log(pw[("STE", float(x) * 1e3)]) for x in b], axis=-1)
        dif = np.stack([np.log(pw[("LTE", float(x) * 1e3)]) for x in b], axis=-1) - ste
    A = np.stack([-b, 0.5 * b ** 2], axis=1) if len(b) >= 2 else -b[:, None]
    md = np.einsum("k,...k->...", np.linalg.pinv(A)[0], ste)
    A = np.stack([b ** 2, b ** 3], axis=1) if len(b) >= 3 else b[:, None] ** 2
    x = np.clip(2.5 * np.einsum("k,...k->...", np.linalg.pinv(A)[0], dif), 0.0, None)   # (D_iso D_Delta)^2, (um^2/ms)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        ufa = np.sqrt(1.5) * np.sqrt(6.0 * x / (3.0 * md ** 2 + 6.0 * x))
    return md, ufa
