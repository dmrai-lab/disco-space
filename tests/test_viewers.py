"""The page's views on constructed inputs (no GPU, no layout): the FOD peaks of a delta FOD are its direction, the slice
viewer draws with and without the overlay, the 3-D views hold the sample they say, the timings rows are the stages."""
import numpy as np

from space import pipeline as P
from space import viewers as V

CFG = P.config()


def test_the_peak_of_a_delta_fod_is_its_direction():
    from dmipy_tract import sh_matrix
    d = np.array([[0.6, 0.0, 0.8], [0.0, 1.0, 0.0]])
    sh = np.zeros((2, 1, 1, 45)); sh[0, 0, 0] = sh_matrix(8, d[:1])[0]; sh[1, 0, 0] = sh_matrix(8, d[1:])[0]
    peaks, amp = P.peaks(sh)
    assert peaks.shape == (2, 1, 1, 3) and (amp > 0).all()
    for k in range(2):
        assert abs(abs(peaks[k, 0, 0] @ d[k]) - 1.0) < 0.01           # within the hemisphere's angular resolution
    zero = np.zeros((1, 1, 1, 45)); zero[..., 0] = np.nan
    p0, a0 = P.peaks(zero)
    assert (p0 == 0).all() and (a0 == 0).all()


def _result():
    p = P.Protocol((P.Shell("d12-D24", 1000, 6),), n_b0=1)
    m = P.measurements(p, CFG["shapes"])
    rng = np.random.default_rng(0)
    dwi = rng.random((6, 6, 6, 7)).astype(np.float32); dwi[0, 0] = np.nan
    sh = np.zeros((6, 6, 6, 45)); sh[..., 0] = 1.0
    peaks, amp = P.peaks(sh)
    return dwi, m, peaks, amp


def test_the_slice_viewer_draws_with_and_without_the_overlay():
    dwi, m, peaks, amp = _result()
    a = V.dwi_slice(dwi, m, 3, 2, peaks=peaks, peak_amp=amp, overlay=True)
    b = V.dwi_slice(dwi, m, 3, 0)
    assert a.size[0] > 100 and b.size[0] > 100


def test_the_3d_views_hold_their_sample():
    import nibabel as nib
    import os
    rois = np.asarray(nib.load(os.path.join(P.DATA_DIR, "DiSCo_ROIs.nii.gz")).dataobj).astype(np.int32)
    strands = [np.cumsum(np.random.default_rng(i).normal(size=(20, 3)), 0) + 20 for i in range(50)]
    regions = V.region_markers(rois)
    assert regions.mode == "markers+text" and len(regions.x) == 16
    fig = V.strands3d(strands, np.full(50, 2e-6), regions, rois.shape, n=30)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert sum(int(np.isnan(np.asarray(t.x, float)).sum()) for t in lines) == 30    # one NaN gap per path
    assert "2.0-2.0 µm" in fig.layout.title.text
    from dmipy_tract import Tractogram
    pts = np.concatenate(strands[:10]).astype(np.float32); offsets = np.r_[0, np.cumsum([len(s) for s in strands[:10]])]
    tg = Tractogram(pts, offsets, np.arange(10), np.zeros((10, 2), np.int8))
    fig2 = V.tractogram3d(tg, regions, rois.shape, n=4, total=12345)
    assert sum(int(np.isnan(np.asarray(t.x, float)).sum()) for t in fig2.data if t.mode == "lines") == 4
    assert "4 of 12,345" in fig2.layout.title.text
    assert V.ground_truth_matrix(np.eye(16) * 0 + np.triu(np.ones((16, 16)), 1), np.triu(np.ones((16, 16)), 1) * 0.5, 120).size[0] > 100


def test_the_timings_rows_are_the_stages():
    rows = V.timings_rows(dict(replay=1.0, csd=2.0, total=3.0), 10.0)
    assert rows[0][0].startswith("source") and [r[0] for r in rows[1:]] == ["replay", "csd", "total"]


def test_the_floor_slice_draws_the_positive_voxels():
    floor = np.zeros((8, 8, 4), np.float32); floor[2:6, 2:6, :] = 0.01; floor[3, 3, 1] = 0.05
    img = V.floor_slice(floor, 1, 0.01, label="A: ")
    assert img.size[0] > 100 and img.size[1] > 100


def test_the_spread_matrices_draw():
    M = np.zeros((16, 16)); M[0, 1] = M[1, 0] = 10
    sp = P.pair_spread([M, M * 1.2], [dict(pearson_count=0.8), dict(pearson_count=0.82)])
    assert V.spread_matrices(sp).size[0] > 100
