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
NeuroImage 125:653): the gamma model fitted to each powder signal, ``V_LTE - V_STE = (4/5) (D_iso D_Delta)^2``, and
muFA is the FA of the axisymmetric microscopic tensor with that ``D_iso`` and ``D_Delta``."""
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


def gamma_fit(b, y, iters=60):
    """``(MD, V)`` per voxel of the gamma model ``ln S/S0 = -(MD^2 / V) ln(1 + b V / MD)`` (Lasic et al. 2014;
    Szczepankiewicz et al. 2016) fitted to ``y`` ``(..., n_b)`` (ln S/S0) at ``b`` ``(n_b,)`` (ms/um^2): a vectorised
    Gauss-Newton over every voxel at once on ``(MD, r = V / MD^2)``, MD in [1e-4, 5] um^2/ms, r in [1e-6, 50]. The
    first cumulant is MD, the second V; a model of every order, so a b-range that reaches the third cumulant does not
    fold it into the second (a cubic cumulant fit over few b-values does, and goes negative)."""
    sh = y.shape[:-1]; Y = y.reshape(-1, len(b)); n = Y.shape[0]
    md = np.clip(-np.nan_to_num(Y[:, 0] / b[0]), 1e-3, None); r = np.full(n, 0.1)

    def f(md, r):
        return -np.log1p(r[:, None] * b[None] * md[:, None]) / r[:, None]
    for _ in range(int(iters)):
        F = f(md, r); res = Y - F
        e1 = 1e-6 * np.maximum(md, 1e-3); e2 = 1e-6
        J1 = (f(md + e1, r) - F) / e1[:, None]; J2 = (f(md, r + e2) - F) / e2
        A11 = (J1 * J1).sum(1); A12 = (J1 * J2).sum(1); A22 = (J2 * J2).sum(1) + 1e-12
        g1 = (J1 * res).sum(1); g2 = (J2 * res).sum(1); det = A11 * A22 - A12 ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            md = np.clip(md + (A22 * g1 - A12 * g2) / det, 1e-4, 5.0); r = np.clip(r + (A11 * g2 - A12 * g1) / det, 1e-6, 50.0)
    return md.reshape(sh), (r * md ** 2).reshape(sh)


def md_ufa(pw, bvals):
    """``(MD (um^2/ms), muFA)`` per voxel from the powder averages: the gamma model (:func:`gamma_fit`) fitted to the
    linear and to the spherical powder signal over the b-values, MD their mean first cumulant, ``V_A = V_LTE - V_STE``
    the anisotropic variance (the orientation dispersion of the microscopic tensors, ``(4/5) (D_iso D_Delta)^2``), and
    muFA the FA of the axisymmetric microscopic tensor with ``D_iso = MD`` and that ``D_Delta``; 0 where ``V_A`` is not
    positive (a spherical signal whose variance exceeds the linear one's: on restricted tissue the two waveforms'
    spectra differ, and the spherical one sees more of the time dependence), NaN where a signal is not positive."""
    keys = sorted(float(x) for x in bvals)                                 # the powder averages' own keys, s/mm^2
    b = np.asarray(keys) * 1e-3                                            # ms/um^2
    with np.errstate(divide="ignore", invalid="ignore"):
        mL, vL = gamma_fit(b, np.stack([np.log(pw[("LTE", k)]) for k in keys], axis=-1))
        mS, vS = gamma_fit(b, np.stack([np.log(pw[("STE", k)]) for k in keys], axis=-1))
    bad = ~np.isfinite(np.stack([np.log(pw[(e, k)]) for e in ("LTE", "STE") for k in keys], axis=-1)).all(-1)
    md = 0.5 * (mL + mS)
    x = np.clip(1.25 * (vL - vS), 0.0, None)                              # (D_iso D_Delta)^2, (um^2/ms)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        ufa = np.sqrt(1.5) * np.sqrt(6.0 * x / (3.0 * md ** 2 + 6.0 * x))
    return np.where(bad, np.nan, md), np.where(bad, np.nan, ufa)
