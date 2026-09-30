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
        assert P.preset_protocol(CFG, name).name == name
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
    p, idx = P.disco_protocol(CFG)
    m = P.measurements(p, CFG["shapes"])
    bv = np.loadtxt(P.DATA_DIR + "/DiSCo_gradients.bvals").ravel()[idx]  # the table's rows in the protocol's order
    assert p.name == "DiSCo 364" and p.n_meas == 364 == len(idx) and sorted(idx) == list(range(364))
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
    out_t = P.add_noise(dwi, 20.0, seed=1, backend="torch")                # numpy's stream: same distribution, another realisation
    assert np.isnan(out_t[0, 0, 0]).all() and abs(out_t[1:].std() - 0.05) < 0.005 and not np.array_equal(out_t[1:], out[1:])


class _Source(P.Source):
    """The ground truth and the config without any replay data (P.Source loads the files, nothing else)."""
    mode = "test"


def test_the_score_of_the_ground_truth_is_one_and_the_pairs_add_up():
    gt = _Source(CFG)
    s = P.score(gt.gt_count, gt)
    assert s["pearson_count"] == pytest.approx(1.0) and s["false_pairs"] == 0 == s["missed_pairs"]
    assert s["connected_pairs"] == s["gt_pairs"] == 25 == P.connected_pairs(gt.gt_count)
    empty = P.score(np.zeros((16, 16)), gt)
    assert empty["missed_pairs"] == 25 and empty["connected_pairs"] == 0 and np.isnan(empty["pearson_count"])   # a constant matrix: NaN, no warning
    assert np.isnan(P.pearson([1, 1, 1], [1, 2, 3])) and P.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_the_volumes_round_trip(tmp_path):
    """The DWI and FOD written as NIfTI read back with the table beside them."""
    import nibabel as nib
    p = P.Protocol((P.Shell("d12-D24", 1000, 6),), n_b0=1)
    m = P.measurements(p, CFG["shapes"])
    dwi = np.random.default_rng(0).random((4, 4, 4, 7)); dwi[0, 0, 0] = np.nan
    sh = np.random.default_rng(1).random((4, 4, 4, 45))
    res = P.Result(p, m, dwi, np.zeros((4, 4, 4)), sh, None, None, None, {}, {})
    paths = P.write_volumes(res, str(tmp_path), prefix="t")
    back = np.asarray(nib.load(paths["dwi"]).dataobj)
    np.testing.assert_allclose(back, np.nan_to_num(dwi).astype(np.float32))
    np.testing.assert_allclose(np.loadtxt(paths["bvals"]), m.bvals); np.testing.assert_allclose(np.loadtxt(paths["bvecs"]).T, m.dirs, atol=1e-8)
    assert np.asarray(nib.load(paths["fod"]).dataobj).shape == (4, 4, 4, 45)


def test_the_backend_and_residency_come_from_the_config_or_the_environment(monkeypatch):
    monkeypatch.delenv("DISCO_BACKEND", raising=False); monkeypatch.delenv("DISCO_RESIDENT", raising=False)
    assert P.backend(CFG) == CFG["compute"]["backend"] == "jax" and P.resident(CFG) is True
    monkeypatch.setenv("DISCO_BACKEND", "torch"); monkeypatch.setenv("DISCO_RESIDENT", "0")
    assert P.backend(CFG) == "torch" and P.resident(CFG) is False
    monkeypatch.setenv("DISCO_BACKEND", "numpy")
    with pytest.raises(ValueError, match="backend"):
        P.backend(CFG)


def test_the_mode_and_the_columns_come_from_the_config_or_the_environment(monkeypatch):
    monkeypatch.delenv("DISCO_MODE", raising=False); monkeypatch.delenv("DISCO_COLUMNS", raising=False)
    assert P.mode(CFG) == "demo" and P.columns_uri(CFG).startswith("hf://")
    monkeypatch.setenv("DISCO_MODE", "full"); monkeypatch.setenv("DISCO_COLUMNS", "/columns")
    assert P.mode(CFG) == "full" and P.columns_uri(CFG) == "/columns"
    monkeypatch.setenv("DISCO_MODE", "fast")
    with pytest.raises(ValueError, match="mode"):
        P.mode(CFG)


def test_a_free_pulse_timing_is_the_shells_own_in_full_mode():
    p = P.Protocol((P.Shell.free(2000, 12, 0.02, 0.05, 0.09),), n_b0=1)
    assert p.shells[0].shape == "d20-D50-TE90" and p.shells[0].free_timing
    m = P.measurements(p, CFG["shapes"])
    np.testing.assert_allclose(m.delta, 0.02); np.testing.assert_allclose(m.Delta, 0.05); np.testing.assert_allclose(m.TE, 0.09)
    with pytest.raises(ValueError, match="delta < Delta < TE"):
        P.Shell.free(2000, 12, 0.06, 0.05, 0.09)


def test_a_table_of_rows_becomes_a_protocol_b0_first_then_each_group_in_order():
    """protocol_from_rows: the one row-grouping used by DiSCo's table and by an uploaded scheme; its refusals."""
    b = np.array([1000, 0, 2000, 1000, 0]); dirs = np.array([[1, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 1], [0, 0, 0]], float)
    group = np.array([0, 0, 1, 0, 0])
    p, idx = P.protocol_from_rows(b, dirs, group, lambda g, rows: P.Shell("d12-D24", float(b[rows[0]]), len(rows)), name="t")
    assert list(idx) == [1, 4, 0, 3, 2] and p.n_b0 == 2 and [(s.b, s.n_dirs) for s in p.shells] == [(1000.0, 2), (2000.0, 1)]
    np.testing.assert_allclose(np.linalg.norm(p.directions, axis=1), 1.0)
    np.testing.assert_array_equal(p.directions[:2], [[0, 0, 1], [0, 0, 1]])
    with pytest.raises(ValueError, match="b = 0"):
        P.protocol_from_rows(b[[0, 2]], dirs[[0, 2]], group[[0, 2]], None, name="t")
    with pytest.raises(ValueError, match="zero direction"):
        P.protocol_from_rows(b, np.zeros((5, 3)), group, None, name="t")
    with pytest.raises(ValueError, match="dirs must be"):
        P.protocol_from_rows(b, dirs[:3], group, None, name="t")


def test_a_camino_scheme_becomes_a_protocol_with_every_rows_timing(tmp_path):
    """Two shells with their own delta / Delta at one TE plus a b = 0 row, written as Camino STEJSKALTANNER rows
    (|G| in T/m, times in s): one free shell per distinct timing, the rows' own b-values and directions, b = 0 first."""
    rows = ["VERSION: STEJSKALTANNER"]
    rng = np.random.default_rng(0)
    rows.append("0 0 0 0 0.0300 0.0100 0.0800")
    for delta, Delta, G in ((0.0100, 0.0300, 0.0400), (0.0150, 0.0400, 0.0500)):
        for _ in range(4):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            rows.append(f"{u[0]:.6f} {u[1]:.6f} {u[2]:.6f} {G:.4f} {Delta:.4f} {delta:.4f} 0.0800")
    path = tmp_path / "t.scheme"; path.write_text("\n".join(rows) + "\n")
    p = P.protocol_from_scheme(str(path))
    assert p.n_meas == 9 and p.n_b0 == 1 and len(p.shells) == 2 and all(s.free_timing for s in p.shells)
    m = P.measurements(p, CFG["shapes"])
    assert m.b0[0] and (m.delta[1:5] == 0.0100).all() and (m.delta[5:] == 0.0150).all() and np.allclose(m.TE, 0.08)
    assert np.allclose(np.linalg.norm(m.dirs, axis=1), 1.0)
    np.testing.assert_allclose(sorted(s.b for s in p.shells), [305, 1409], atol=1)   # Stejskal-Tanner b of 40 and 50 mT/m
    assert p.shells[0].shape == "d10-D30-TE80"


def test_the_physics_is_the_catalogue_at_the_nearest_field_with_the_tiers_as_switches():
    """3 T is cited; 0.064 T and 11.7 T take the nearest cited field (1.5 T, 7 T) and say so; a tier off leaves that
    part out of the tissue; all off is bare (no tissue); the refusals."""
    c3 = P.catalogue(3.0); assert c3["catalogue_field"] == 3.0 and set(c3["T2"]) == set(P.CATALOGUE_POOLS) and c3["T2"]["myelin"] < c3["T2"]["intra"]
    two = P.catalogue(3.0, ("intra", "extra")); assert set(two["T2"]) == {"intra", "extra"}
    with pytest.raises(ValueError, match="no relaxation for the pools"):
        P.catalogue(3.0, ("intra", "csf"))
    assert P.catalogue(0.064)["catalogue_field"] == 1.5 and P.catalogue(11.7)["catalogue_field"] == 7.0
    ph = P.Physics.at(3.0)
    assert ph.pools == ("intra", "extra")
    tis = ph.tissue()
    assert tis.T2 == two["T2"] and tis.T1 == two["T1"] and tis.rho == c3["rho"] and tis.chi_iso == c3["chi_iso"] and tis.chi_aniso == c3["chi_aniso"]
    filled = ph.tissue(unseeded=("myelin",))                              # the unseeded spec pool takes the first pool's value
    assert set(filled.T2) == {"intra", "extra", "myelin"} and filled.T2["myelin"] == two["T2"]["intra"]
    assert not ph.bare and "relaxation+contact+field" in ph.label()
    no_field = P.Physics.at(7.0, field=False, b0_direction=P.b0_direction(90, 0))
    assert no_field.tissue().chi_iso is None and no_field.tissue().chi_aniso == 0.0 and no_field.tissue().rho == c3["rho"]
    np.testing.assert_allclose(no_field.b0_direction, (1, 0, 0), atol=1e-12)
    bare = P.Physics.at(3.0, relaxation=False, contact=False, field=False)
    assert bare.bare and bare.tissue() is None and bare.label() == "bare diffusion"
    assert P.tissue_and_scanner(bare) == (None, None) and P.tissue_and_scanner(None) == (None, None)
    assert P.tissue_and_scanner(no_field)[1] is None and P.tissue_and_scanner(ph)[1] == 3.0    # the field goes with the field tier
    only_relax = P.Physics.at(3.0, contact=False, field=False, T2={"intra": 0.08, "extra": 0.08})
    assert only_relax.tissue().T2["intra"] == 0.08 and only_relax.tissue().rho is None
    with pytest.raises(ValueError, match="tesla"):
        P.Physics.at(3000.0)
    with pytest.raises(ValueError, match="same pools"):
        P.Physics.at(3.0, T2={"intra": 0.05})
    with pytest.raises(ValueError, match="per pool"):
        P.Physics.at(3.0, T2={"intra": -0.05, "extra": 0.05})
    with pytest.raises(ValueError, match="unit vector"):
        P.Physics.at(3.0, b0_direction=(0, 0, 2))
    np.testing.assert_allclose(np.linalg.norm(P.B0_TRANSVERSE), 1.0)
    assert P.Physics.at(3.0).along_z and not no_field.along_z


def test_a_tractogram_sample_keeps_whole_streamlines_in_order():
    from dmipy_tract.tractogram import Tractogram
    rng = np.random.default_rng(1)
    counts = rng.integers(2, 6, size=50)
    tg = Tractogram(rng.normal(size=(counts.sum(), 3)), np.concatenate([[0], np.cumsum(counts)]), np.arange(50), np.zeros((50, 2), np.int8))
    s = P.sample_tractogram(tg, 10, seed=0)
    assert len(s) == 10 and np.all(np.diff(s.seed_index) > 0)
    for k, i in enumerate(s.seed_index):
        np.testing.assert_array_equal(s[k], tg[int(i)])
    assert P.sample_tractogram(tg, 100) is tg


def test_the_gradient_a_shell_needs_and_the_scanners_that_can_play_it():
    """b = 1000 at δ 12 / Δ 24 ms needs 70 mT/m (γ² G² δ² (Δ − δ/3)); the Prisma (80 mT/m) plays it, the low-field
    class (23 mT/m) does not; DiSCo's b = 13183 shell needs the Connectom class; None means no limit."""
    G = P.gradient_needed(1000, 0.012, 0.024)
    assert 0.069 < G < 0.071
    classes = P.scanner_classes()
    assert classes["prisma"] == (0.08, 3.0) and classes["low_field"][0] < 0.03
    p = P.Protocol((P.Shell("d12-D24", 1000, 30),), n_b0=1)
    assert P.playable(p, CFG["shapes"], "prisma")[0][-1] and not P.playable(p, CFG["shapes"], "low_field")[0][-1]
    assert P.playable(p, CFG["shapes"], None)[0][5] is None and P.playable(p, CFG["shapes"], None)[0][-1]
    disco, _ = P.disco_protocol(CFG)
    rows = P.playable(disco, CFG["shapes"], "connectom")
    assert all(r[-1] for r in rows) and not all(r[-1] for r in P.playable(disco, CFG["shapes"], "prisma"))
    free = P.Protocol((P.Shell.free(1000, 30, 0.012, 0.024, 0.06),), n_b0=1)
    assert abs(P.playable(free, CFG["shapes"], None)[0][4] - G) < 1e-12


def test_the_pair_spread_over_repeated_runs():
    M1 = np.zeros((16, 16)); M1[0, 1] = M1[1, 0] = 10; M1[2, 3] = M1[3, 2] = 4
    M2 = M1.copy(); M2[0, 1] = M2[1, 0] = 14; M2[2, 3] = M2[3, 2] = 0; M2[4, 5] = M2[5, 4] = 2
    sp = P.pair_spread([M1, M2], [dict(pearson_count=0.8), dict(pearson_count=0.9)])
    assert sp["n"] == 2 and abs(sp["pearson_mean"] - 0.85) < 1e-12 and abs(sp["pearson_std"] - np.std([0.8, 0.9], ddof=1)) < 1e-12
    assert sp["pairs_any"] == 3 and sp["pairs_always"] == 1 and sp["mean"][0, 1] == 12 and abs(sp["std"][0, 1] - np.std([10, 14], ddof=1)) < 1e-12
    one = P.pair_spread([M1], [dict(pearson_count=0.8)])
    assert one["pearson_std"] == 0.0 and one["std"].max() == 0.0


def test_floor_stats_are_over_the_voxels_with_signal():
    dwi = np.ones((3, 3, 3, 2)); dwi[0, 0, 0] = np.nan
    floor = np.arange(27, dtype=float).reshape(3, 3, 3) / 27
    res = P.Result(None, None, dwi, floor, None, None, None, None, {})
    f = P.floor_stats(res)
    assert f["n"] == 26 and f["max"] == floor[2, 2, 2] and f["median"] == np.median(floor.ravel()[1:])


def test_the_layout_at_the_pinned_revision_holds_the_configs_classes_and_tiers():
    """The manifest the Space will download: every [shapes] class with the config's timing, and the tiers when the
    tissue panel starts on (needs the Hub; skipped offline)."""
    import json
    hf = pytest.importorskip("huggingface_hub")
    d = CFG["data"]
    import requests
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled
    try:
        path = hf.hf_hub_download(d["repo"], f"{d['moments']}/manifest.json", repo_type="dataset", revision=d["revision"])
    except (requests.ConnectionError, HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled) as e:   # offline, or the Hub unreachable
        pytest.skip(f"the Hub is not reachable: {e!r}"[:120])
    with open(path) as f:
        m = json.load(f)
    assert m["source"]["manifest_sha256"] == d["source_manifest_sha256"]
    missing = [c for c in CFG["shapes"] if c not in m["columns"]]
    assert not missing, f"the layout at {d['revision'][:12]} lacks {missing}"
    for name, t in CFG["shapes"].items():
        enc = m["shapes"][name].get("encoding") or {}
        for k in ("delta", "Delta", "TE"):
            v = enc.get(k); v = v[0] if isinstance(v, list) else v
            if v is not None:
                assert abs(v - t[k]) < 1e-9, (name, k, v, t[k])
    if CFG["physics"]["default_on"]:
        assert m.get("tiers"), "the tissue panel starts on but the layout has no tiers"
