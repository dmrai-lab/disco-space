"""The pipeline's pieces that need no GPU and no layout: protocols and their measurements, DiSCo's own table as
a protocol, the noise, the score on the ground truth itself, the config's consistency."""
import numpy as np
import pytest

from space import pipeline as P

CFG = P.config()


def test_the_config_names_every_shell_and_preset_shape():
    shapes = set(CFG["shapes"])
    assert {s["shape"] for s in CFG["disco"]["shells"]} <= shapes
    for name, preset in CFG["presets"].items():
        assert {s["shape"] for s in preset["shells"]} <= shapes, name
        P.Protocol(tuple(P.Shell(s["shape"], s["b"], s["n_dirs"]) for s in preset["shells"]), n_b0=preset["n_b0"], name=name)
    for s in CFG["shapes"].values():
        assert 0 < s["delta"] < s["Delta"] < s["TE"]


def test_a_protocols_measurements_are_its_rows_in_order():
    p = P.Protocol((P.Shell("d12-D24", 1000, 30), P.Shell("d8-D20", 6000, 12)), n_b0=2)
    m = P.measurements(p, CFG["shapes"])
    assert p.n_meas == 44 and m.bvals.shape == (44,) and m.dirs.shape == (44, 3)
    assert m.b0.sum() == 2 and list(m.shape[:2]) == ["d12-D24"] * 2      # the b = 0 rows take the first shell's timing
    assert (m.bvals[2:32] == 1000).all() and (m.bvals[32:] == 6000).all()
    np.testing.assert_allclose(np.linalg.norm(m.dirs, axis=1), 1.0, atol=1e-12)
    assert (m.dirs[2:, 2] > 0).all()                                     # a hemisphere: no antipodal pairs
    np.testing.assert_allclose(m.delta[32:], CFG["shapes"]["d8-D20"]["delta"])
    np.testing.assert_allclose(m.Delta[:32], CFG["shapes"]["d12-D24"]["Delta"])


def test_protocol_refusals():
    with pytest.raises(ValueError, match="at least one shell"):
        P.Protocol(())
    with pytest.raises(ValueError, match="b = 0"):
        P.Protocol((P.Shell("d12-D24", 1000, 30),), n_b0=0)
    with pytest.raises(ValueError, match="b > 0"):
        P.Protocol((P.Shell("d12-D24", 0, 30),))
    with pytest.raises(ValueError, match="directions must be"):
        P.Protocol((P.Shell("d12-D24", 1000, 3),), directions=np.zeros((2, 3)))
    with pytest.raises(KeyError, match="no timing class"):
        P.measurements(P.Protocol((P.Shell("d99-D99", 1000, 3),)), CFG["shapes"])


def test_discos_table_is_a_protocol_of_its_own_rows():
    """The dataset's 364 rows: 4 shells on 3 timing classes, every row's direction its own, b = 0 rows first, the
    rows' b-values those of the table."""
    p, bv = P.disco_protocol(CFG)
    m = P.measurements(p, CFG["shapes"])
    assert p.name == "DiSCo 364" and p.n_meas == 364 == len(bv)
    assert [s.b for s in p.shells] == [1000, 1925, 3094, 13192]          # the table's shells, rounded means
    assert [s.shape for s in p.shells] == ["d10.2-D16.7", "d10.2-D16.7", "d7.6-D45.9", "d17.7-D35.8"]
    assert m.b0.sum() == p.n_b0 == 4 and m.b0[:p.n_b0].all()
    np.testing.assert_array_equal(m.bvals, bv)                           # every row's own b-value, exactly
    assert sorted(np.unique(np.round(bv)).tolist()) == [0, 1000, 1925, 3094, 13192]
    np.testing.assert_allclose(np.linalg.norm(m.dirs, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(m.delta[m.shape == "d7.6-D45.9"], 7.6e-3)


def test_noise_is_rician_at_the_given_snr_and_leaves_nan_alone():
    dwi = np.full((5, 5, 5, 40), 0.5); dwi[0, 0, 0] = np.nan
    out = P.add_noise(dwi, 20.0, seed=1)
    assert np.isnan(out[0, 0, 0]).all() and np.isfinite(out[1:]).all()
    assert abs(out[1:].std() - 0.05) < 0.005 and (out[1:] >= 0).all()   # sigma = 1 / SNR, a magnitude
    assert P.add_noise(dwi, None) is dwi
    with pytest.raises(ValueError, match="positive"):
        P.add_noise(dwi, 0.0)


class _GroundTruth:
    """A layout stand-in holding the ground-truth matrices only."""
    def __init__(self):
        import os
        self.gt_count = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_Connectivity_Matrix_Strands_Count.txt"))
        self.gt_area = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt"))


def test_the_score_of_the_ground_truth_is_one_and_the_pairs_add_up():
    gt = _GroundTruth()
    s = P.score(gt.gt_count, gt)
    assert s["pearson_count"] == pytest.approx(1.0) and s["false_pairs"] == 0 == s["missed_pairs"]
    assert s["connected_pairs"] == s["gt_pairs"] == 25
    empty = P.score(np.zeros((16, 16)), gt)
    assert empty["missed_pairs"] == 25 and empty["connected_pairs"] == 0
