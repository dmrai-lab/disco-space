"""The brain source: the asset's layout (the BATMAN fixture of ``tools/build_brain_fixture.py``, local only), the
kernel-route replay against dmipy-sim's own ``Phantom.compose(...).replay`` (the oracle), the connectome of a known
tractogram, the reservation model, the panel and its knobs, the refusals, and the page's chain on a small crop of the
fixture. Skipped where the fixture or the packs are not on this machine; ``DISCO_BRAIN_FIXTURE`` and
``DISCO_BRAIN_PACKS`` point at them."""
import json
import os

import numpy as np
import pytest

from space import pipeline as P
from space.sources import brain as B

FIXTURE = os.environ.get("DISCO_BRAIN_FIXTURE", os.path.expanduser("~/dmrai-ws/data/batman/brain_fixture"))
PACKS = os.environ.get("DISCO_BRAIN_PACKS", os.path.expanduser("~/dmrai-ws/packs"))
WM_PACK = os.path.join(PACKS, "cactus_single_bundle_100ms_xframe.spec.rpk")
GM_PACK = os.path.join(PACKS, "gm_spheres_100ms.rpk")
has_fixture = pytest.mark.skipif(not os.path.exists(os.path.join(FIXTURE, "manifest.json")), reason="the BATMAN brain fixture is not on this machine")
has_packs = pytest.mark.skipif(not (os.path.exists(WM_PACK) and os.path.exists(GM_PACK)), reason="the 100 ms packs are not on this machine")


def brain_cfg(asset=FIXTURE):
    """brain.toml with the asset and the packs local (the tests never touch the network)."""
    cfg = P.config(os.path.join(P.HERE, "brain.toml"))
    cfg["asset"] = {"local": asset}
    cfg["packs"] = {"wm": [{"label": "CACTUS single bundle, 100 ms", "uri": WM_PACK}],
                    "gm": [{"label": "grey-matter spheres, 100 ms", "uri": GM_PACK}]}
    return cfg


def default_values(cfg):
    panel = B.Brain.panel(cfg)
    return {c.name: c.value for row in panel.rows for c in row if c.name in panel.fields}


@pytest.fixture(scope="module")
def brain():
    if not os.path.exists(os.path.join(FIXTURE, "manifest.json")):
        pytest.skip("the BATMAN brain fixture is not on this machine")
    return B.Brain(brain_cfg())


@has_fixture
def test_the_fixture_loads_in_the_assets_layout(brain):
    """Every file of the contract, cropped to the brain with its affine shifted, 84 regions in 14 lobar groups, the
    FOD a unit-integral density per voxel, the fractions shares of a voxel, the scan's protocol in the image frame."""
    a = brain.asset
    man = a.manifest
    assert a.fod.shape == a.mask.shape + (45,) and a.fractions.shape == a.mask.shape + (3,) and a.labels.shape == a.mask.shape
    assert all(s < n for s, n in zip(a.mask.shape, man["grid"]["shape"])) and int(a.mask.sum()) == 90205
    lo = np.array([c.start for c in a.crop])
    np.testing.assert_allclose(a.affine[:3, 3], np.asarray(man["grid"]["affine"])[:3, :3] @ lo + np.asarray(man["grid"]["affine"])[:3, 3])
    assert brain.regions.n == 84 and len(brain.regions.groups) == 14 and sorted(i for _, ids in brain.regions.groups for i in ids) == list(range(1, 85))
    np.testing.assert_allclose(brain.fod_unit[brain.f[:, 0] > 0][:, 0], B.C00, rtol=1e-5)
    assert (a.fractions.sum(-1) <= 1 + 1e-6).all() and (a.stop <= a.mask).all()
    prot = B.Brain.protocol(brain.cfg, B.Brain.presets(brain.cfg)[0])
    assert prot.n_meas == len(man["protocol"]["bvals_s_mm2"]) and {s.shape for s in prot.shells} == {B.SCAN}
    g = np.asarray(man["protocol"]["bvecs"], float); b = np.asarray(man["protocol"]["bvals_s_mm2"], float)
    first = g[b > 50][0] / np.linalg.norm(g[b > 50][0])
    assert np.allclose(prot.directions[prot.n_b0:][0], a.R.T @ first) or np.isclose(np.abs(prot.directions[prot.n_b0:] @ (a.R.T @ first)).max(), 1.0)


@has_fixture
@has_packs
def test_the_replay_is_dmipy_sims_phantom_composition(brain):
    """At the scan's protocol, 3 T with every tier on (relaxation, contact, the sheath's field along the bore), noise
    off: the Space's float32 contraction over 200 voxels equals ``Phantom.compose(...).replay(seq, pose=...)`` of the
    same packs, tissue, fractions and FODs to 1e-5 in M0 units."""
    from dmipy_sim.phantom import FreeWater, Grid, Inert, ODF, PackSubstrate, Phantom
    cfg = brain.cfg
    ph = B.Brain.physics_from(cfg, default_values(cfg))
    assert ph.relaxation and ph.contact and ph.field and ph.field_T == 3.0
    prot = B.Brain.protocol(cfg, B.Brain.presets(cfg)[0])
    meas = brain.validate(prot, ph)
    k = brain.prepare(meas, ph)
    dwi, floor, factor, _ = brain.replay(meas, ph, k)
    pick = np.sort(np.random.default_rng(0).choice(len(brain.vox), 200, replace=False))
    ijk = brain.vox[pick]; n = len(ijk)
    fr = brain.asset.fractions[tuple(ijk.T)].astype(np.float64); fod = brain.asset.fod[tuple(ijk.T)].astype(np.float64)
    wm_pack, gm_pack = brain.pack("wm", ph.wm_pack, 1), brain.pack("gm", ph.gm_pack, 1)
    t_wm, t_gm, t_csf = ph.tissues(brain.pools(wm_pack), brain.pools(gm_pack))
    wm = PackSubstrate(wm_pack, m0=ph.m0["wm"], name="wm", tissue=t_wm)
    gm = PackSubstrate(gm_pack, m0=ph.m0["gm"], name="gm", tissue=t_gm)
    csf = FreeWater(m0=ph.m0["csf"], tissue=t_csf)
    iso = np.zeros((n, 45)); iso[:, 0] = B.C00
    grid = Grid(shape=(n, 1, 1), voxel_size_m=(2.5e-3,) * 3)
    phantom = Phantom.compose(grid, fractions={wm: fr[:, 0].reshape(n, 1, 1), gm: fr[:, 1].reshape(n, 1, 1), csf: fr[:, 2].reshape(n, 1, 1)},
                              remainder=Inert(), orientation={wm: ODF(fod.reshape(n, 1, 1, 45), basis="tournier07"),
                                                              gm: ODF(iso.reshape(n, 1, 1, 45), basis="tournier07")})
    pose = B.specimen_pose(brain.asset.R, ph.b0_direction)
    ref = phantom.replay(brain.sequence(meas, np.arange(len(meas.bvals)), pose), scanner=ph.scanner_field, pose=pose)[:, 0, 0, :]
    got = (dwi * factor[..., None])[tuple(ijk.T)]
    live = np.isfinite(ref[:, 0])
    assert live.sum() > 150 and (np.isfinite(got[:, 0]) == live).all()
    assert np.max(np.abs(got[live] - ref[live])) < 1e-5
    assert floor.max() == 0 and np.nanmax(dwi) <= 1 + 1e-5


@has_fixture
@has_packs
def test_prepare_is_kept_per_tissue_and_field_and_the_m0_is_the_devices(brain):
    """The pose responses are kept per (packs, protocol, tissue, field): the cache has them for other proton densities
    (zero seconds), not for another tissue, and the replay follows the new M0 linearly per tissue."""
    cfg = brain.cfg
    v = default_values(cfg)
    ph = B.Brain.physics_from(cfg, v)
    meas = brain.validate(P.preset_protocol(cfg, "clinical b1000 x 30"), ph)
    k1 = brain.prepare(meas, ph)
    k2 = brain.cached(meas, B.Brain.physics_from(cfg, {**v, "m0_gm": 0.5}))
    assert k2.seconds == 0.0 and k2.m0["gm"] == 0.5 and k1.m0["gm"] == v["m0_gm"] and k2.wm is k1.wm
    assert brain.cached(meas, B.Brain.physics_from(cfg, {**v, "rho": 2 * v["rho"]})) is None
    only_gm = {**v, "m0_wm": 0.0, "m0_csf": 0.0}
    a = brain.replay(meas, B.Brain.physics_from(cfg, only_gm), brain.cached(meas, B.Brain.physics_from(cfg, only_gm)))
    b = brain.replay(meas, B.Brain.physics_from(cfg, {**only_gm, "m0_gm": 0.5 * v["m0_gm"]}), brain.cached(meas, B.Brain.physics_from(cfg, {**only_gm, "m0_gm": 0.5 * v["m0_gm"]})))
    ok = np.isfinite(a[2])
    np.testing.assert_allclose(b[2][ok], 0.5 * a[2][ok], rtol=1e-5)                       # the b = 0 signal halves
    np.testing.assert_allclose(b[0][ok], a[0][ok], atol=1e-5)                              # the normalised signal does not move


def _regions():
    labels = np.zeros((6, 2, 2), np.int32); labels[0] = 1; labels[2] = 2; labels[4] = 3; labels[5] = 4
    return P.Regions(labels, np.diag([2.0, 2.0, 2.0, 1.0]), ("a", "b", "c", "d"), (("L x", (1, 2)), ("R x", (3, 4))))


def test_the_connectome_of_a_known_tractogram():
    """Four streamlines with known endpoint regions (mm through the affine): the symmetrised count, no diagonal, a
    streamline ending in no region uncounted, and the group sums (the diagonal the within-group count, once)."""
    from dmipy_tract import Tractogram
    reg = _regions()
    x = lambda i: np.array([2.0 * i, 0.0, 0.0])
    paths = [np.stack([x(0), x(2)]), np.stack([x(2), x(0)]), np.stack([x(0), x(4)]), np.stack([x(4), x(5)]), np.stack([x(1), x(5)])]
    pts = np.concatenate(paths).astype(np.float32); off = np.r_[0, np.cumsum([len(p) for p in paths])]
    tg = Tractogram(pts, off, np.arange(len(paths)), np.zeros((len(paths), 2), np.int8))
    M = reg.matrix(tg)
    expect = np.zeros((4, 4)); expect[0, 1] = expect[1, 0] = 2; expect[0, 2] = expect[2, 0] = 1; expect[2, 3] = expect[3, 2] = 1
    np.testing.assert_array_equal(M, expect)
    G = reg.grouped(M)
    np.testing.assert_array_equal(G, [[2, 1], [1, 1]])
    assert P.connected_pairs(M) == 3 and len(reg.pairs[0]) == 6


@has_fixture
def test_the_score_is_against_the_inputs_connectome(brain):
    """Pearson of log(1 + count): one for the truth itself, the pairs connected in one only counted, the lobar view;
    B against A the same numbers between two runs."""
    import types
    n = brain.regions.n
    rng = np.random.default_rng(0)
    T = np.triu(rng.poisson(3.0, (n, n)), 1).astype(float); T = T + T.T
    ref = dict(matrix=T, lobar=brain.regions.grouped(T), streamlines=100)
    s = brain.score(T, ref)
    assert s["pearson_log"] == pytest.approx(1.0) and s["only_run"] == 0 == s["only_truth"] and s["lobar_pearson_log"] == pytest.approx(1.0)
    M = T.copy(); M[0, 1] = M[1, 0] = 0; M[2, 3] = M[3, 2] = T[2, 3] + 5
    s2 = brain.score(M, ref)
    assert s2["only_truth"] == int(T[0, 1] > 0) and s2["pearson_log"] < 1
    a = types.SimpleNamespace(matrix=T, score=s); b = types.SimpleNamespace(matrix=M, score=s2)
    c = brain.compare(a, b)
    assert c["only_a"] == int(T[0, 1] > 0) and c["only_b"] == 0 and c["delta_log"] == pytest.approx(s2["pearson_log"] - 1.0)


def test_the_reservation_grows_with_the_run_and_the_default_fits_a_logged_out_visitor():
    """The brain's model from ``[budget]``: more measurements, the ladder, B, a denser seeding and more keys each
    cost more; the default acquisition with A + B + ladder at the default density reserves at most 120 s."""
    cfg = P.config(os.path.join(P.HERE, "brain.toml"))
    small = P.preset_protocol(cfg, "clinical b1000 x 30"); big = P.preset_protocol(cfg, "research 3-shell x 90")
    est = lambda prot, **kw: B.Brain.estimated_seconds(cfg, prot, **{**dict(density=cfg["tracking"]["density"], knob=P.NO_KNOB, n_keys=1, ladder=False), **kw})
    knob = "SNR → 10"
    assert 30 <= est(small) < est(big) < est(big, ladder=True) < est(big, ladder=True, knob=knob) <= 480
    assert est(big, density=1) < est(big, density=2) < est(big, density=4) and est(big) < est(big, n_keys=3)
    scan_like = P.Protocol((P.Shell(B.SCAN, 1000, 96), P.Shell(B.SCAN, 1500, 96), P.Shell(B.SCAN, 2000, 96), P.Shell(B.SCAN, 2500, 96), P.Shell(B.SCAN, 3000, 96)), n_b0=15)
    assert est(scan_like, ladder=True, knob=knob) <= 120      # MASiVar's five shells x 96 at the default density, A + B + ladder


@has_fixture
def test_the_panel_is_a_brain_physics_and_every_knob_changes_one_thing(brain):
    cfg = brain.cfg
    panel = B.Brain.panel(cfg)
    assert set(panel.fields) <= set(panel.controls()) and set(panel.catalogue) <= set(panel.fields)
    v = default_values(cfg)
    ph = B.Brain.physics_from(cfg, v)
    assert ph.m0 == {"wm": 0.7, "gm": 0.85, "csf": 1.0} and ph.T2["wm"]["intra"] == pytest.approx(0.05) and ph.scanner_field == 3.0
    off = B.Brain.physics_from(cfg, {**v, "on": False})
    assert off.bare and off.m0 == ph.m0 and off.scanner_field is None and off.wm_pack == ph.wm_pack
    prot = P.preset_protocol(cfg, "clinical b1000 x 30")
    for label, change in B.Brain.knobs(cfg).items():
        if change is None:
            continue
        pb, on, snr, vb = B.Brain.apply_knob(cfg, change, prot, True, 30.0, v)
        diffs = [k for k in panel.fields if vb[k] != v[k]]
        kind = change[0]
        if kind == "field":
            assert "field_T" in diffs or change[1] == v["field_T"]
        elif kind in ("tier", "bare", "b0", "m0_gm", "pack"):
            assert len(diffs) <= 1, (label, diffs)
        elif kind in ("snr", "shape"):
            assert diffs == []
    steps = brain.ladder_steps(ph)
    assert [l for l, _ in steps] == ["bare diffusion", "+ relaxation", "+ relaxation + contact"] and steps[0][1].bare and steps[0][1].m0 == ph.m0


@has_fixture
@has_packs
def test_the_refusals_name_what_the_packs_cannot_play(brain):
    """A class whose echo lies beyond the 100 ms walk (the configuration's classes for a longer walk among them), and
    the multi-tissue reconstruction on a dmipy-fit without it."""
    cfg = brain.cfg
    ph = B.Brain.physics_from(cfg, default_values(cfg))
    brain.pack("wm", ph.wm_pack, 1); brain.pack("gm", ph.gm_pack, 1)
    brain.shapes = {**brain.shapes, "long": dict(label="long", delta=0.03, Delta=0.06, TE=0.12)}
    with pytest.raises(ValueError, match="100 ms walk: it cannot play the class 'long'"):
        brain.validate(P.Protocol((P.Shell("long", 1000, 6),)), ph)
    for name in ("ste-d12-TM150", "long-TE-d30-D120"):
        with pytest.raises(ValueError, match=f"100 ms walk: it cannot play the class '{name}'"):
            brain.validate(P.Protocol((P.Shell(name, 1000, 6),)), ph)
    import inspect
    from dmipy_fit.tissue_response.three_tissue_response import three_tissue_response_dhollander16
    if "mask" not in inspect.signature(three_tissue_response_dhollander16).parameters:
        with pytest.raises(RuntimeError, match="dmipy-fit#39"):
            P.csd_msmt(np.ones((2, 1, 1, 7)), P.measurements(P.Protocol((P.Shell("d12-D24", 1000, 6),)), cfg["shapes"]), np.ones((2, 1, 1), bool), backend="torch")


def mini_asset(tmp_path, size=(18, 18, 10)):
    """A crop of the fixture around the brain's centre, in the asset's layout (every region name kept)."""
    d = tmp_path / "asset"; d.mkdir()
    man = json.load(open(os.path.join(FIXTURE, "manifest.json")))
    mask = np.load(os.path.join(FIXTURE, "mask.npy"))
    c = np.argwhere(mask).mean(0).astype(int)
    sl = tuple(slice(int(ci - s // 2), int(ci - s // 2 + s)) for ci, s in zip(c, size))
    for f in ("fod_wm.npy", "fractions.npy", "mask.npy", "labels.npy", "stop_mask.npy", "mean_b0.npy"):
        np.save(d / f, np.load(os.path.join(FIXTURE, f))[sl])
    A = np.asarray(man["grid"]["affine"]); A[:3, 3] = A[:3, :3] @ np.array([s.start for s in sl]) + A[:3, 3]
    man["grid"] = dict(shape=list(size), voxel_size_mm=man["grid"]["voxel_size_mm"], affine=A.tolist())
    json.dump(man, open(d / "manifest.json", "w"))
    import shutil
    shutil.copy(os.path.join(FIXTURE, "regions.json"), d / "regions.json")
    return str(d)


@has_fixture
@has_packs
def test_the_brain_page_runs_a_to_b_on_a_small_crop(tmp_path, monkeypatch):
    """The page's chain on a crop of the fixture (single-tissue reconstruction, the torch backend on the CPU): the cache
    lookup, the device part (the packs' responses, A, B = A with the field off, the ladder), the drawing; every output there, the headline
    against the input's connectome, the round trip and the timings naming the reconstruction path, the files."""
    pytest.importorskip("gradio")
    from space import app as A
    monkeypatch.setenv("DISCO_RECONSTRUCTION", "tournier07")
    asset = mini_asset(tmp_path)
    monkeypatch.setenv("DISCO_CONFIG", "brain.toml"); monkeypatch.setenv("DISCO_BRAIN_ASSET", asset)     # what estimated_seconds reads
    cfg = brain_cfg(asset)
    src = B.Brain(cfg)
    state = dict(error=None, source=src, cfg=cfg, load_seconds=0.1, regions=__import__("space.viewers", fromlist=["x"]).region_markers(src.regions), gt_views=None)
    monkeypatch.setattr(A, "_load", lambda: state)
    panel = B.Brain.panel(cfg)
    physics = [default_values(cfg)[k] for k in panel.fields]
    shells = [True, "d12-D24", 1000, 12, 10.0, 20.0, 60.0] + [False, "d12-D24", 2000, 12, 10.0, 20.0, 60.0] * 3
    args = ["research 3-shell x 90", 1, True, 30, 1, 45.0, 1.25, 0, None, "field tier → off", A.IDEAL, 2, True, *physics, *shells]
    out = list(A.run_pipeline(A.compute, *args))
    final = out[-1]
    assert len(final) == len(A.OUTPUTS)
    named = dict(zip(A.OUTPUTS, final))
    assert "connectome vs the input's" in named["headline"] and "A vs B" in named["headline"] and "tracker keys" in named["headline"]
    rows = dict((r[0], r[1]) for r in named["timings"])
    assert any("tournier07" in k for k in rows) and any(k.startswith("A · truth") for k in rows) and float(rows[A.RESPONSES_ROW]) > 0
    assert any(r[0] == "reconstruction" and "tournier07" in r[1] for r in named["roundtrip"])
    for key in ("dwi_view", "mats", "lobar_view", "truth_view", "fractions_view", "response_view", "ingredient_pool", "explore_view"):
        assert named[key] is not None, key
    assert any(p.endswith("_connectome.csv") for p in named["volumes"]) and all(os.path.exists(p) for p in named["tck"])
    labels = named["layer_choice"]["choices"]
    assert [l if isinstance(l, str) else l[0] for l in labels][:3] == ["bare diffusion", "+ relaxation", "+ relaxation + contact"]
    assert A.estimated_seconds(*args) >= 30
    fig, img = src.truth_views()                              # the input tab
    assert len(fig.data) == 1 and img.size[0] > 100
    demo = A.build(cfg=cfg)                                   # the page itself builds from the brain's class alone
    assert demo.title == cfg["describe"]["title"]


@has_fixture
@has_packs
def test_the_worker_computes_what_the_page_has_not_cached_and_the_page_keeps_it(monkeypatch):
    """prepare_runs is a lookup: None for every entry the page has not cached (A, B, the ladder's rungs). The worker
    (another process: here another Brain) computes exactly those inside the call, says so in a stage text, and hands
    them over by key; the page keeps them, after which the same run finds every entry cached and equal."""
    pytest.importorskip("gradio")
    from space import app as A
    cfg = brain_cfg()
    page, worker = B.Brain(cfg), B.Brain(cfg)
    monkeypatch.setattr(A, "_load", lambda: dict(error=None, source=page, cfg=cfg))
    v = default_values(cfg)
    physics = [v[k] for k in B.Brain.panel(cfg).fields]
    args = ["clinical b1000 x 30", 1, True, 30, 1, 45.0, 1.25, 0, None, "B0 direction → " + list(B.B0_MODES)[1], A.IDEAL, 1, True, *physics] + [False] * 28
    prepared = A.prepare_runs(*args)
    assert set(prepared["runs"]) == {"A", "B"} and all(k is None for k in [*prepared["runs"].values(), *prepared["ladder"]]) and len(prepared["ladder"]) == 3
    runs = A.plan_runs(cfg, page, *args[:4], args[8], args[9], args[10], v, [False] * 28)
    gen = A._fill(worker, runs, True, prepared)
    texts = []
    try:
        while True:
            texts.append(next(gen))
    except StopIteration as stop:
        filled, computed, seconds = stop.value
    assert len(texts) == 1 and "computing the packs' responses on the worker" in texts[0][0] and seconds > 0
    assert len(computed) == 5 and all(k is not None for k in [*filled["runs"].values(), *filled["ladder"]])
    assert page.cached(P.measurements(runs[0][1], page.shapes), runs[0][4]) is None      # the worker's work is not the page's yet
    page.keep(computed)
    again = A.prepare_runs(*args)
    for got, want in zip([*again["runs"].values(), *again["ladder"]], [*filled["runs"].values(), *filled["ladder"]]):
        assert got.seconds == 0.0 and np.array_equal(got.wm, want.wm) and np.array_equal(got.csf, want.csf)
    gen = A._fill(worker, runs, True, again)                  # nothing missing: no stage text, nothing computed
    try:
        next(gen); raise AssertionError("a stage text with nothing to compute")
    except StopIteration as stop:
        assert stop.value[1] == {} and stop.value[2] < 0.01


@has_fixture
@has_packs
def test_the_reservation_prices_the_responses_the_page_has_not_cached(monkeypatch):
    """Nothing cached: A is a band to compile (cold), the ladder's rungs compile nothing (no field tier). With A's
    entries cached on the page, B's knob decides the extra: a new field is a band to compile (cold), a new field
    direction at A's band is warm, the field tier off is the ladder's top rung, cached; the reservation rises by the priced
    entry and falls back once the page keeps B's responses (as after a run). Another band compiled by an earlier
    entry of the same run is warm."""
    from space import app as A
    cfg = brain_cfg()
    page = B.Brain(cfg)
    monkeypatch.setitem(A._state, "source", page); monkeypatch.setitem(A._state, "cfg", cfg)
    monkeypatch.setattr(A, "_load", lambda: dict(error=None, source=page, cfg=cfg))
    monkeypatch.setattr(A.P, "config", lambda path=None: cfg)
    v = default_values(cfg)
    physics = [v[k] for k in B.Brain.panel(cfg).fields]
    args = lambda knob, ladder=True: ["clinical b1000 x 30", 1, True, 30, 1, 45.0, 1.25, 0, None, knob, A.IDEAL, 1, ladder, *physics] + [False] * 28
    seven = "field → 7 T (catalogue tissue at that field)"
    cold = A.uncached_responses(*args(P.NO_KNOB))
    saves = 1002                                              # the 100 ms WM pack's one window
    assert cold == [("cold", 31, saves), ("no_field", 31, saves), ("no_field", 31, saves), ("no_field", 31, saves)]
    prot = P.preset_protocol(cfg, "clinical b1000 x 30")
    for (_, meas, ph) in A.response_entries(page, A.plan_runs(cfg, page, *args(P.NO_KNOB)[:4], None, P.NO_KNOB, A.IDEAL, v, [False] * 28), True):
        page.prepare(meas, ph)
    assert A.uncached_responses(*args(P.NO_KNOB)) == []
    assert A.uncached_responses(*args(seven)) == [("cold", 31, saves)]
    assert A.uncached_responses(*args("B0 direction → " + list(B.B0_MODES)[1])) == [("warm", 31, saves)]
    assert A.uncached_responses(*args("field tier → off")) == []                    # A's top rung of the ladder
    b = cfg["budget"]
    base = B.Brain.estimated_seconds(cfg, prot, density=1, knob=seven, n_keys=1, ladder=True)
    before = A.estimated_seconds(*args(seven))
    cold_price = b["response_cold"]["fixed"] + 31 * b["response_cold"]["per_meas"] + saves * b["response_cold"]["per_save"]
    assert abs((before - base) - b["margin"] * cold_price) <= 1
    runs = A.plan_runs(cfg, page, *args(seven)[:4], None, seven, A.IDEAL, v, [False] * 28)
    meas_b, ph_b = A.response_entries(page, runs, False)[1][1:]
    worker = B.Brain(cfg)
    page.keep({page.response_key(meas_b, ph_b): worker.prepare(meas_b, ph_b)})
    assert A.uncached_responses(*args(seven)) == [] and A.estimated_seconds(*args(seven)) == base < before
    seven_both = dict(v, field_T=7.0)                         # A at 7 T (cold), B its other direction: warm after A
    physics7 = [seven_both[k] for k in B.Brain.panel(cfg).fields]
    a7 = ["clinical b1000 x 30", 1, True, 30, 1, 45.0, 1.25, 0, None, "B0 direction → " + list(B.B0_MODES)[1], A.IDEAL, 1, False, *physics7] + [False] * 28
    assert A.uncached_responses(*a7) == [("cold", 31, saves), ("warm", 31, saves)]


@has_fixture
def test_the_asset_as_the_masivar_build_writes_it_loads(tmp_path):
    """The MASiVar build's spellings (tools/build_brain_asset.py, build_parcellation.py): the basis named
    ``dmipy-fit ...``, the pulse timing null (the sidecars carry none), regions.json with ``lobes`` names instead of
    groups, fractions of an unconstrained fit summing above one, a WM fraction where the FOD is zero: it loads, the
    scan's class takes the configuration's timing and says so, and the two corrections are counted."""
    d = mini_asset(tmp_path)
    man = json.load(open(os.path.join(d, "manifest.json")))
    man["fod"]["basis"] = "dmipy-fit tournier07 (dipy real_sh_tournier real-SH ordering)"
    man["protocol"].update(TE_s=None, delta_s=None, Delta_s=None)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    regions = json.load(open(os.path.join(d, "regions.json")))["regions"]
    json.dump(dict(convention="fixture", regions=regions, lobes=[f"{r['lobe']}-{r['hemisphere']}" for r in regions]), open(os.path.join(d, "regions.json"), "w"))
    fr = np.load(os.path.join(d, "fractions.npy")).astype(np.float32); fod = np.load(os.path.join(d, "fod_wm.npy")); mask = np.load(os.path.join(d, "mask.npy"))
    i, j, k = np.argwhere(mask & (fr[..., 0] > 0.3))[0]
    fr[i, j, k] = [0.8, 0.3, 0.1]
    i2, j2, k2 = np.argwhere(mask & (fr[..., 0] > 0.3))[1]
    fod[i2, j2, k2] = 0.0
    np.save(os.path.join(d, "fractions.npy"), fr.astype(np.float16)); np.save(os.path.join(d, "fod_wm.npy"), fod)
    B.manifest.cache_clear()
    src = B.Brain(brain_cfg(d))
    assert "configuration's timing" in src.shapes[B.SCAN]["label"] and src.shapes[B.SCAN]["TE"] == src.cfg["scan"]["TE"]
    notes = dict(src.asset.notes)
    assert any("sum above one" in n for n in notes) and any("no FOD" in n for n in notes)
    assert (src.asset.fractions.sum(-1) <= 1 + 1e-6).all() and len(src.regions.groups) == 14
    wrong = dict(man, fod=dict(man["fod"], basis="descoteaux07"))
    json.dump(wrong, open(os.path.join(d, "manifest.json"), "w")); B.manifest.cache_clear()
    with pytest.raises(ValueError, match="required basis"):
        B.Brain(brain_cfg(d))
    B.manifest.cache_clear()


# ---- the pack's windows (RPK.md 4.3): a run holds the windows its acquisition reaches ----

@pytest.fixture(scope="module")
def two_windows(tmp_path_factory):
    """A cylinder walk of 200 ms stored in two 100 ms windows (300 walkers, every tier, built with the bank) and
    the same pack truncated to its first window (no re-encode), both on disk: ``(two, one)`` paths."""
    import dmipy_sim as d
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.replay.bank import build_replay_pack
    out = tmp_path_factory.mktemp("windows")
    walk = d.simulate_trajectories(300, 2e-9, d.Cylinder(radius=3e-6, orientation=(0.0, 0.0, 1.0)), 0.2, 2e-4, seed=0, require_gpu=False)
    two = str(out / "two.rpk"); one = str(out / "one.rpk")
    build_replay_pack(walk, id="test/two-windows", license="x", citation="x", K=48, out_path=two, segment_T=0.1)
    read_rpk(two).truncate(1, out_path=one)
    return two, one


@pytest.fixture(scope="module")
def windowed(two_windows):
    """A brain whose packs are the two-window fixture (``two``) and its first window (``one``), for WM and GM."""
    if not os.path.exists(os.path.join(FIXTURE, "manifest.json")):
        pytest.skip("the BATMAN brain fixture is not on this machine")
    two, one = two_windows
    cfg = brain_cfg()
    menu = [{"label": "two", "uri": two}, {"label": "one", "uri": one}]
    cfg["packs"] = {"wm": list(menu), "gm": list(menu)}
    return B.Brain(cfg)


def _class(name, n=6, b=1000):
    return P.Protocol((P.Shell(name, b, n),), n_b0=1)


@has_fixture
def test_the_windows_a_class_reaches(windowed):
    """On a pack of two 100 ms windows: the 60 / 70 ms classes and the 100 ms class reach window 0 alone (the save at
    100 ms ends window 0, window 1 starts there), the 160 ms PGSE and the 184 ms stimulated echo reach both; the table
    reads the same from the pack's header and from the loaded pack's meta; the first window alone is one window
    whatever the class."""
    two = windowed.segments("wm", "two")
    assert two == dict(n=2, n_t=501, T=pytest.approx(0.1))
    for name, k in (("d12-D24", 1), ("ste-d12-TM36", 1), ("d25-D55", 1), ("long-TE-d30-D120", 2), ("ste-d12-TM150", 2)):
        meas = P.measurements(_class(name), windowed.shapes)
        assert windowed.windows_reached(two, meas) == {name: k}, name
        assert windowed.windows_needed("wm", "two", meas) == k
        assert windowed.windows_needed("wm", "one", meas) == 1
    both = P.measurements(P.Protocol((P.Shell("d12-D24", 1000, 6), P.Shell("long-TE-d30-D120", 2000, 6)), n_b0=1), windowed.shapes)
    assert windowed.windows_reached(two, both) == {"d12-D24": 1, "long-TE-d30-D120": 2}
    pk = windowed.pack("wm", "two", 1)
    assert windowed.windows_reached(pk.meta, both) == windowed.windows_reached(two, both)
    assert B.Brain.saves_spanned(two, 1) == 501 and B.Brain.saves_spanned(two, 2) == 1001


@has_fixture
def test_a_class_beyond_the_declared_windows_is_refused_by_name(windowed):
    """A 250 ms echo on the 200 ms pack of two windows reaches window 2: refused by name before anything loads."""
    cfg = windowed.cfg
    ph = B.Brain.physics_from(cfg, {**default_values(cfg), "wm_pack": "two", "gm_pack": "two"})
    windowed.shapes = {**windowed.shapes, "beyond": dict(label="beyond", delta=0.03, Delta=0.12, TE=0.25)}
    with pytest.raises(ValueError, match=r"'two' declares 2 window\(s\) of 100 ms: the class 'beyond' \(TE 250 ms\) reaches window 2"):
        windowed.validate(_class("beyond"), ph)
    meas = windowed.validate(_class("long-TE-d30-D120"), ph)
    assert len(meas.bvals) == 7


@has_fixture
def test_a_hub_pack_loads_the_windows_reached_and_reloads_for_more(windowed, two_windows, monkeypatch):
    """A Hub pack declaring two windows: the first request loads ``windows=range(1)``, a request within what is held
    loads nothing, a request for two reloads with ``range(2)`` and replaces the cached pack; a local path loads whole.
    (The library's window-ranged read is stood in for by the fixture stamped with the windows asked for;
    ``test_the_hub_fixture_loads_by_window`` reads the Hub.)"""
    import dmipy_sim.phantom as phantom
    from dmipy_sim.replay import read_rpk
    two, _ = two_windows
    uri = "hf://SubstrateCommons/test/packs/two.rpk"
    calls = []

    class Windowed:
        def __init__(self, pack, *, m0, name=None, windows=None):
            calls.append((pack, None if windows is None else list(windows)))
            self.pack = read_rpk(two)
            if windows is not None:
                self.pack.meta["windows_present"] = len(windows)

    real = B.declared_segments
    monkeypatch.setattr(B, "declared_segments", lambda u: real(two) if u == uri else real(u))
    monkeypatch.setattr(phantom, "PackSubstrate", Windowed)
    src = B.Brain.__new__(B.Brain)
    src.cfg = {**windowed.cfg, "packs": {"wm": [{"label": "hub", "uri": uri}, {"label": "local", "uri": two}], "gm": []}}
    src.packs = {}
    seg = src.segments("wm", "hub")
    a = src.pack("wm", "hub", 1)
    assert calls == [(uri, [0])] and src.windows_held(a, seg) == 1
    assert src.pack("wm", "hub", 1) is a and len(calls) == 1
    b = src.pack("wm", "hub", 2)
    assert calls[-1] == (uri, [0, 1]) and src.packs[("wm", "hub")] is b and src.windows_held(b, seg) == 2
    assert src.pack("wm", "hub", 1) is b and len(calls) == 2
    whole = src.pack("wm", "local", 1)
    assert calls[-1] == (two, None) and src.windows_held(whole, seg) == 2


HUB_FIXTURE = "hf://SubstrateCommons/parity-fixtures/test-fixtures/windows-fixture.rpk"


@pytest.mark.skipif(not os.environ.get("DISCO_HUB_TESTS"), reason="reads the Hub: set DISCO_HUB_TESTS=1")
@has_fixture
def test_the_hub_fixture_loads_by_window(windowed, monkeypatch):
    """The public three-window fixture on the Hub, read by byte range: ``windows=range(1)`` for a class within
    window 0, then ``range(2)`` when a class reaching window 1 asks, which replaces the cached pack. The fixture has
    no record in its dataset's manifest, so its declared table is read from its own header (a window-0 read keeps
    the parent's table)."""
    from dmipy_sim.replay import ReplayPack
    table = ReplayPack.load(HUB_FIXTURE, windows=range(1)).segments
    real = B.declared_segments
    monkeypatch.setattr(B, "declared_segments", lambda u: dict(n=int(table["n"]), n_t=int(table["n_t"]), T=float(table["T"])) if u == HUB_FIXTURE else real(u))
    src = B.Brain.__new__(B.Brain)
    src.cfg = {**windowed.cfg, "packs": {"wm": [{"label": "hub", "uri": HUB_FIXTURE}], "gm": []}}
    src.packs = {}; src.shapes = windowed.shapes
    seg = src.segments("wm", "hub")
    assert seg["n"] == 3
    T = float(seg["T"])                                        # classes whose echo sits in window 0, then in window 1
    src.shapes = {**src.shapes, "w0": dict(label="w0", delta=0.2 * T, Delta=0.5 * T, TE=T), "w1": dict(label="w1", delta=0.3 * T, Delta=0.75 * T, TE=1.5 * T)}
    short, long = P.measurements(_class("w0"), src.shapes), P.measurements(_class("w1"), src.shapes)
    assert src.windows_needed("wm", "hub", short) == 1
    assert src.windows_needed("wm", "hub", long) == 2
    one = src.pack("wm", "hub", 1)
    assert one.windows_present == 1 and one.n_segments == 3
    two = src.pack("wm", "hub", 2)
    assert two.windows_present == 2 and src.packs[("wm", "hub")] is two


@has_fixture
def test_the_reservation_rises_with_the_windows_reached(windowed):
    """A class within the first window and one reaching the second: the WM pack's saves spanned (501, then 1001);
    the reservation of the responses the page has not cached rises by ``per_save`` x the saves a second window adds."""
    cfg = windowed.cfg
    ph = B.Brain.physics_from(cfg, {**default_values(cfg), "wm_pack": "two", "gm_pack": "two"})
    short = windowed.validate(_class("d17-D30"), ph); long = windowed.validate(_class("long-TE-d30-D120"), ph)
    assert windowed.responses([(short, ph)]) == [("cold", 7, 501)] and windowed.responses([(long, ph)]) == [("cold", 7, 1001)]
    prot = P.preset_protocol(cfg, "research 3-shell x 90")                  # above the 30 s floor
    swoop = dict(n=10, n_t=1302, T=0.09995)                                   # the 1 s pack's windows: 1,302 and 2,603 saves
    one, both = B.Brain.saves_spanned(swoop, 1), B.Brain.saves_spanned(swoop, 2)
    est = lambda r: B.Brain.estimated_seconds(cfg, prot, density=2, knob=P.NO_KNOB, n_keys=1, ladder=False, responses=r)
    b = cfg["budget"]
    for state in ("cold", "warm", "no_field"):
        per_save = b[f"response_{state}"]["per_save"]
        lo, hi = est([(state, prot.n_meas, one)]), est([(state, prot.n_meas, both)])
        assert per_save > 0 and lo < hi and abs((hi - lo) - b["margin"] * (both - one) * per_save) <= 1, state


@has_fixture
def test_a_one_window_class_on_two_windows_is_the_first_window_bit_for_bit(windowed):
    """Every tier on, a class within the first window: the pose responses of the two-window pack equal those of the
    pack truncated to its first window to the bit (the replay reads window 0 alone), the explorer's curves and the
    tiers' b = 0 weights too; the accuracy table says the windows each holds."""
    cfg = windowed.cfg
    v = default_values(cfg)
    ph_two = B.Brain.physics_from(cfg, {**v, "wm_pack": "two", "gm_pack": "two"})
    ph_one = B.Brain.physics_from(cfg, {**v, "wm_pack": "one", "gm_pack": "one"})
    assert ph_two.relaxation and ph_two.contact and ph_two.field
    prot = P.Protocol((P.Shell("d17-D30", 1000, 12), P.Shell("d17-D30", 2500, 12)), n_b0=2)
    k2 = windowed.prepare(windowed.validate(prot, ph_two), ph_two)
    k1 = windowed.prepare(windowed.validate(prot, ph_one), ph_one)
    for x in ("wm", "gm", "csf"):
        assert np.array_equal(getattr(k2, x), getattr(k1, x)), x
    for x in ("wm", "gm", "csf"):
        assert np.array_equal(k2.profile[x], k1.profile[x]), x
    assert k2.b0 == k1.b0 and windowed.windows_held(windowed.packs[("wm", "two")], windowed.segments("wm", "two")) == 2
    rows = dict(windowed.accuracy())
    assert "windows held / declared 2 / 2 of 100 ms (1,001 of 1,001 saves at 200 µs)" in rows["WM pack two"]
    assert "windows held / declared 1 / 1 of 100 ms (501 of 501 saves" in rows["WM pack one"]


@has_fixture
@has_packs
def test_the_warm_up_runs_in_its_own_process_and_the_page_keeps_what_arrives(monkeypatch):
    """``Brain.warm_in_background`` loads the packs here, spawns the worker on ``warm_entries`` and keeps every
    ``(key, Kernels)`` the worker sends; the page then finds the entries cached and equal to its own ``prepare``.
    Two entries stand for the set."""
    cfg = brain_cfg()
    full = B.Brain(cfg, asset=FIXTURE)
    two = full.warm_entries()[:2]
    assert len(two) == 2
    monkeypatch.setattr(B.Brain, "warm_entries", lambda self: two)
    page = B.Brain(cfg, asset=FIXTURE)
    assert all(page.cached(m, ph) is None for m, ph in two)
    drain = page.warm_in_background()
    assert ("wm", two[0][1].wm_pack) in page.packs and ("gm", two[0][1].gm_pack) in page.packs   # loaded here, before the worker
    drain.join(timeout=600)
    assert not drain.is_alive(), "the warm-up did not finish in time"
    for m, ph in two:
        got = page.cached(m, ph)
        assert got is not None and np.array_equal(got.wm, full.prepare(m, ph).wm)
    assert P.Source.warm_in_background(P.Source.__new__(P.Source)) is None   # the base: warm() itself, nothing to join


@has_fixture
@has_packs
def test_the_warm_set_has_one_entry_per_response_and_the_high_fields_last():
    """``Brain.warm_entries``: no two entries share a response key (the noise, M0, default-pack and default-direction
    knobs repeat the default run's response), the first entry is the scan's protocol at the panel's defaults, and
    the field presets close the set in ascending field, so the minutes 7 T and 11.7 T cost come after everything
    cheaper, the pulse-timing classes among it."""
    cfg = brain_cfg()
    src = B.Brain(cfg, asset=FIXTURE)
    entries = src.warm_entries()
    keys = [src.response_key(m, ph) for m, ph in entries]
    assert len(keys) == len(set(keys))
    assert len(entries) < sum(1 for c in B.Brain.knobs(cfg).values() if c is not None) + 3 * 4
    first_meas, first_ph = entries[0]
    assert set(first_meas.shape.tolist()) == {"scan"} and first_ph == B.Brain.physics_from(cfg, default_values(cfg))
    fields = [ph.field_T for m, ph in entries]
    catalogue_fields = sorted(float(f) for f in cfg["physics"]["fields"])
    high = [f for f in fields if f > 3.0]
    assert high == sorted(high) and fields[-2:] == [catalogue_fields[-1]] * 2
    shapes = [i for i, (m, ph) in enumerate(entries) if set(m.shape.tolist()) != {"scan"} and len(m.bvals) == len(entries[0][0].bvals)]
    assert shapes and max(shapes) < min(i for i, f in enumerate(fields) if f > 3.0)


@has_fixture
@has_packs
def test_a_machine_plays_every_pulse_at_its_slew_and_field():
    """``BrainPhysics.scanner`` names a machine of dmipy-sim's catalogue: the pulses are trapezoids with ramps at its
    slew (the same b), its field and direction replace the panel's, the response key tells the machines apart, the label
    names it, a magnet with its own gradient (the Swoop) is refused by name, and a name outside the catalogue too."""
    cfg = brain_cfg()
    src = B.Brain(cfg, asset=FIXTURE)
    values = default_values(cfg)
    ideal = B.Brain.physics_from(cfg, values)
    real = B.Brain.physics_from(cfg, dict(values, field_T=1.5), scanner="siemens_magnetom_terra_7T")
    assert ideal.scanner is None and np.isinf(ideal.slew_rate)
    assert real.scanner == "siemens_magnetom_terra_7T" and real.slew_rate == pytest.approx(200.0) and real.field_T == 7.0
    assert real.b0_direction == (0.0, 0.0, 1.0) and "on the siemens_magnetom_terra_7T" in real.label() and "on the" not in ideal.label()
    with pytest.raises(ValueError, match="two-directional"):
        B.Brain.physics_from(cfg, values, scanner="hyperfine_swoop_64mT")
    with pytest.raises(Exception):
        B.Brain.physics_from(cfg, values, scanner="a magnet nobody built")
    prot = src.protocol(cfg, src.presets(cfg)[0])
    meas = P.measurements(prot, src.shapes)
    rows = np.flatnonzero(meas.shape == meas.shape[0])
    square = src.sequence(meas, rows, np.eye(3), ideal.slew_rate)
    trap = src.sequence(meas, rows, np.eye(3), real.slew_rate)
    assert np.max(square.encoding.ramp_time) == 0.0 and np.max(trap.encoding.ramp_time) > 0.0
    np.testing.assert_allclose(trap.b(), square.b(), rtol=1e-6)                 # measured 1.8e-7: the trapezoid sampled on the 1000-sample grid
    assert src.response_key(meas, real) != src.response_key(meas, ideal)


@has_fixture
def test_the_head_sits_at_isocentre_at_its_physical_offsets():
    """The bore grid puts every brain voxel at its affine's place relative to the head centre (the left-handed
    affine's third index reversed, not mirrored), with non-negative indices a phantom can take."""
    src = B.Brain(brain_cfg(), asset=FIXTURE)
    grid, idx = src.bore_grid
    A = src.affine
    truth = (A[:3, :3] @ src.vox.T).T * 1e-3 + A[:3, 3] * 1e-3
    np.testing.assert_allclose(grid.offset_m(idx) @ np.asarray(grid.to_scanner).T, truth - truth.mean(0), atol=1e-12)
    assert idx.min() >= 0 and np.all(idx.max(0) < np.asarray(grid.shape))


@has_fixture
@has_packs
def test_a_machine_plays_each_voxel_its_encoding_class_and_a_representative_exactly():
    """On the Prisma, the clinical preset: every voxel is played its encoding class (the delivered b of every member
    within the configured tolerance of its class's), and a class's first member -- the voxel whose exact values the
    class was composed from -- gets exactly what its own delivered acquisition, expanded by the packs and composed with
    its FOD and fractions, gives: the replay to 1e-5 in M0 units; the machine moves the image off the ideal scanner's."""
    from dmipy_sim.phantom import FreeWater
    from dmipy_sim.phantom.bore import encoding_classes
    from dmipy_sim.acquisition.waveforms import rotate_waveform
    cfg = brain_cfg()
    src = B.Brain(cfg, asset=FIXTURE)
    values = default_values(cfg)
    ph = B.Brain.physics_from(cfg, values, scanner="siemens_magnetom_prisma_3T")
    ideal = B.Brain.physics_from(cfg, values)
    prot = B.Brain.protocol(cfg, "clinical b1000 x 30")
    meas = src.validate(prot, ph)
    k = src.prepare(meas, ph)
    dwi, _, factor, _ = src.replay(meas, ph, k)
    dwi0, _, factor0, _ = src.replay(meas, ideal, src.prepare(meas, ideal))
    rows = np.flatnonzero(meas.shape == meas.shape[0])
    cls, played = src.encoding(meas, ph, rows)
    assert 1 < len(played) < 50 and k.wm.shape[0] == len(played)
    b_class = np.array([p.b() for p in played])                                # (n_classes, n_meas)
    grid, idx = src.bore_grid
    tol = float(cfg["scanners"]["tolerance"])
    rng = np.random.default_rng(1)
    for v in rng.choice(len(idx), 40, replace=False):
        _, own = encoding_classes(P.limits(ph.scanner), grid, src.sequence(meas, rows, np.eye(3), ph.slew_rate), idx[v:v + 1], tolerance=None)
        b = own[0].b(); nz = meas.bvals[rows] > 50
        assert np.max(np.abs(b[nz] / b_class[cls[v]][nz] - 1)) < 2 * tol
    pose = B.specimen_pose(src.asset.R, ph.b0_direction)
    wm_pack, gm_pack = src.pack("wm", ph.wm_pack, 1), src.pack("gm", ph.gm_pack, 1)
    t_wm, t_gm, t_csf = ph.tissues(src.pools(wm_pack), src.pools(gm_pack))
    reps = [int(np.flatnonzero(cls == c)[0]) for c in range(len(played))]
    reps = [v for v in reps if np.isfinite(factor[tuple(src.vox[v])])][:4]          # a class's first member may hold no signal
    assert reps
    for v in reps:
        _, own = encoding_classes(P.limits(ph.scanner), grid, src.sequence(meas, rows, np.eye(3), ph.slew_rate), idx[v:v + 1], tolerance=None)
        seq = rotate_waveform(own[0], pose)
        Cw = wm_pack.pose_responses([seq], tissue=t_wm, scanner=ph.scanner_field, pose=pose, keep=(B.LMAX, 0))[0].retained(B.LMAX, 0) @ B.axis_map()
        eg = gm_pack.pose_responses([seq], tissue=t_gm, scanner=ph.scanner_field, pose=pose, keep=(0, 0))[0].retained(0, 0)[:, 0] * B.axis_map()[0, 0] * B.C00
        ec = FreeWater(m0=1.0, tissue=t_csf).response(seq)
        f = src.f[v].astype(np.float64); m0 = ph.m0
        ref = np.abs(f[0] * m0["wm"] * (src.fod_unit[v].astype(np.float64) @ Cw.T) + f[1] * m0["gm"] * eg + f[2] * m0["csf"] * ec)
        got = (dwi * factor[..., None])[tuple(src.vox[v])][rows]
        assert np.max(np.abs(got - ref)) < 1e-5, (v, np.max(np.abs(got - ref)))
    moved = np.nanmax(np.abs(dwi * factor[..., None] - dwi0 * factor0[..., None]))
    assert moved > 1e-3


@has_fixture
@has_packs
def test_the_warm_set_holds_the_scan_on_every_machine_that_can_play_it():
    """After the knobs, ``warm_entries`` adds the scan's protocol on each machine of the menu that can play its
    shells, at the machine's field with its slew; a machine that cannot play the scan is left out."""
    cfg = brain_cfg()
    src = B.Brain(cfg, asset=FIXTURE)
    scan = src.protocol(cfg, src.presets(cfg)[0])
    menu = P.machines(cfg).values()
    can = {key for key in menu if all(r[-1] for r in P.playable(scan, src.shapes, key))}
    got = {ph.scanner: ph for m, ph in src.warm_entries() if ph.scanner is not None}
    assert set(got) == can and can
    for key, ph in got.items():
        assert ph.field_T == pytest.approx(P.limits(key).field_T) and ph.slew_rate == pytest.approx(P.limits(key).slew_max)