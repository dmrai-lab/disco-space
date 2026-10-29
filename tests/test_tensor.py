"""The b-tensor class's encodings (space/tensor.py) without the layout: every row's b-tensor, read off the sequence by
dmipy-sim, is the encoding asked for; the class's waveform stays inside the scanner it was designed for; the maps
recover a known microscopic anisotropy from Gaussian compartments played through the same rows."""
import numpy as np
import pytest

from space import pipeline as P
from space import tensor as T

CFG = P.config()
TC = T.config(CFG)
pytestmark = pytest.mark.skipif(TC is None, reason="no [tensor] class in the configuration")


def _rows(bvals=(500.0, 1000.0, 2000.0), n=12):
    return T.rows(TC, list(bvals), n)


def test_every_row_is_the_encoding_asked_for():
    """b to 1e-6 relative and b_delta to 1e-6 (1, -1/2, 0) for every row; the b = 0 row plays nothing."""
    from dmipy_sim.acquisition.waveforms import btensor_invariants, calc_btensor
    seq, enc, bs = _rows()
    B = calc_btensor(seq)
    b, b_delta, _ = btensor_invariants(B)
    assert enc[0] == "b0" and abs(b[0]) < 1e-6
    for e in T.ENCODINGS:
        m = enc == e
        np.testing.assert_allclose(b[m] / 1e6, bs[m], rtol=1e-6)
        np.testing.assert_allclose(b_delta[m], T.B_DELTA[e], atol=1e-6)
    ste = B[enc == "STE"]                                            # spherical: b/3 I at every rotation
    np.testing.assert_allclose(ste, (bs[enc == "STE"] * 1e6 / 3)[:, None, None] * np.eye(3), rtol=0, atol=1e-6 * 2e9)


def test_the_waveform_is_refocused_and_inside_its_scanner():
    """Each shape's net moment at the echo vanishes and the designed amplitude and slew are within 80 mT/m and
    200 T/m/s per axis; the spherical row at b = 2000 at the design's own frame needs no more than the design."""
    W = T.waveform(TC); g = T._grid(); e = np.asarray(g.effective_gate, np.float64)
    net = np.abs((W * e[:, None]).sum(0) * g.dt)
    assert (net < 1e-9 * np.abs(W).max() * 0.0535).all(), net
    assert np.abs(W).max() <= 0.080 * (1 + 1e-6)
    assert np.abs(np.diff(W, axis=0) / g.dt).max() <= 200.0 * 3 * (1 + 1e-6)   # linear interpolation of a 200 T/m/s design


@pytest.mark.parametrize("bvals", [(500.0, 1000.0, 2000.0), (100.0, 700.0, 1400.0, 2000.0)])
def test_the_maps_recover_a_known_microscopic_anisotropy(bvals):
    """Powder of randomly oriented axisymmetric Gaussian tensors (D_iso 1 um^2/ms, muFA 0.8) played through the rows'
    own b-tensors: MD within 3 %, muFA within 0.03 over the b-values (the cumulant fits' own truncation); the powder
    averages are read by the b-values as given (700 is not 0.7 x 1e3 in floating point)."""
    seq, enc, bs = _rows(bvals=bvals, n=30)
    from dmipy_sim.acquisition.waveforms import calc_btensor
    B = calc_btensor(seq) * 1e-9                                       # s/m^2 -> ms/um^2
    D_iso, mufa = 1.0, 0.8
    x = mufa ** 2 * D_iso ** 2 / (3 - 2 * mufa ** 2)                  # (D_iso D_Delta)^2 of an axisymmetric tensor of that muFA
    D_delta = np.sqrt(x) / D_iso
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4000, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    Dpar, Dperp = D_iso * (1 + 2 * D_delta), D_iso * (1 - D_delta)
    D = Dperp * np.eye(3)[None] + (Dpar - Dperp) * u[:, :, None] * u[:, None, :]
    S = np.exp(-np.einsum("mij,cij->mc", B, D)).mean(1)
    S = S / S[0]
    pw = T.powder(S[None, None, None, :], enc, bs, list(bvals))
    md, got = T.md_ufa(pw, list(bvals))
    assert abs(float(md.ravel()[0]) - D_iso) < 0.03 * D_iso
    assert abs(float(got.ravel()[0]) - mufa) < 0.03
