"""The page's acquisition choice without a browser: DiSCo's table, a config preset, the shell rows (a stored class in
demo mode, the row's own delta / Delta / TE in full mode), a scheme upload only in full mode; an unknown choice
refused (no Gradio needed: the mapping lives beside the page, not in a widget)."""
import types

import numpy as np
import pytest

from space import app as A
from space import pipeline as P
from space.sources import disco as D

CFG = P.config()
S = D.Disco
ROWS = [True, "d12-D24", 1000, 30, 10.0, 20.0, 60.0, True, "d8-D20", 3000, 45, 8.0, 20.0, 60.0,
        False, "d17-D30", 3000, 90, 17.0, 30.0, 60.0, False, "d17-D30", 6000, 60, 17.0, 30.0, 60.0]


def test_the_three_kinds_of_acquisition():
    disco = A._protocol_from_inputs(S, CFG, "DiSCo 364", 5, *ROWS)
    assert disco.name == "DiSCo 364" and disco.n_meas == 364 and disco.n_b0 == 4        # the table's own, not the slider's
    name = next(iter(CFG["presets"]))
    preset = A._protocol_from_inputs(S, CFG, name, 5, *ROWS)
    assert preset.name == name and preset.n_b0 == CFG["presets"][name]["n_b0"]
    assert [s.shape for s in preset.shells] == [s["shape"] for s in CFG["presets"][name]["shells"]]
    custom = A._protocol_from_inputs(S, CFG, A.CUSTOM, 2, *ROWS)
    assert custom.n_b0 == 2 and [(s.shape, s.b, s.n_dirs) for s in custom.shells] == [("d12-D24", 1000.0, 30), ("d8-D20", 3000.0, 45)]
    assert not any(s.free_timing for s in custom.shells)
    with pytest.raises(ValueError, match="unknown acquisition"):
        A._protocol_from_inputs(S, CFG, "something else", 2, *ROWS)
    with pytest.raises(ValueError, match="at least one shell"):
        A._protocol_from_inputs(S, CFG, A.CUSTOM, 2, *([False] + ROWS[1:7] + [False] + ROWS[8:14] + ROWS[14:]))


def test_full_mode_takes_the_rows_own_timing_and_demo_mode_refuses_the_upload():
    full = A._protocol_from_inputs(S, CFG, A.CUSTOM, 2, *ROWS, full=True)
    assert all(s.free_timing for s in full.shells)
    assert [(s.delta, s.Delta, s.TE) for s in full.shells] == [(0.010, 0.020, 0.060), (0.008, 0.020, 0.060)]
    with pytest.raises(ValueError, match="full mode"):
        A._protocol_from_inputs(S, CFG, A.UPLOADED, 2, *ROWS, scheme="x.scheme")
    with pytest.raises(ValueError, match="upload"):
        A._protocol_from_inputs(S, CFG, A.UPLOADED, 2, *ROWS, scheme=None, full=True)


def _values(on=True, field_T=3.0, **over):
    nums = S.catalogue_numbers(CFG, field_T)[:-1]
    v = dict(zip(D.physics_fields(CFG), [on, field_T, next(iter(D.B0_MODES)), 0, 0] + nums + [True, True, True, 7.9, next(iter(D.OFFSET_AXES))]))
    v.update(over)
    return v


def test_the_physics_panel_is_a_physics_in_si_or_none_when_off():
    nums = S.catalogue_numbers(CFG, 3.0)
    assert nums[-1].startswith("catalogue values at 3 T") and S.catalogue_numbers(CFG, 0.064)[-1].startswith("the catalogue has no cited")
    ph = S.physics_from(CFG, _values(field=False))
    assert ph.field_T == 3.0 and ph.T2["intra"] == nums[0] * 1e-3 and ph.rho == nums[4] * 1e-6 and not ph.field and ph.relaxation and ph.pools == D.pools(CFG)
    assert S.physics_from(CFG, _values(on=False)) is None
    free = S.physics_from(CFG, _values(field_T=7.0, b0_mode=D.FREE_B0, theta=90, phi=90))
    assert abs(free.b0_direction[1] - 1.0) < 1e-12
    with pytest.raises(ValueError, match="17 inputs"):
        A.physics_values(S, CFG, 1, 2, 3)


def test_a_knob_changes_one_thing_of_a():
    """Every knob maps to B = A with that one change: the field takes the catalogue tissue with it, a tier goes off,
    the tissue goes off, the noise changes, every shell is retimed; A's other settings survive; a panel knob on an
    A whose panel is off is refused (B would differ in two things)."""
    v = _values()
    prot = A._protocol_from_inputs(S, CFG, A.CUSTOM, 2, *ROWS)
    ks = S.knobs(CFG)
    assert ks[A.NO_KNOB] is None and len(ks) == 1 + 5 + 2 + 3 + 1 + 3 + len(CFG["shapes"])
    for name, change in ks.items():
        if change is None:
            continue
        pb, on, snr, vb = S.apply_knob(CFG, change, prot, True, 30.0, v)
        diffs = [k for k in D.physics_fields(CFG) if vb[k] != v[k]]
        kind = change[0]
        if kind == "field":
            assert vb["field_T"] == change[1] and [vb[k] for k in D.tissue_numbers(CFG)] == S.catalogue_numbers(CFG, change[1])[:-1] and pb is prot
        elif kind == "b0":
            assert diffs == ["b0_mode"] or (diffs == [] and change[1] == v["b0_mode"])
        elif kind == "tier":
            assert diffs == [change[1]] and vb[change[1]] is False
        elif kind == "bare":
            assert diffs == ["on"] and S.physics_from(CFG, vb) is None
        elif kind == "snr":
            assert diffs == [] and (on, snr) == ((False, 30.0) if change[1] is None else (True, change[1]))
        elif kind == "shape":
            assert diffs == [] and all(s.shape == change[1] for s in pb.shells) and [s.b for s in pb.shells] == [s.b for s in prot.shells]
        if kind in ("field", "b0", "tier"):
            with pytest.raises(ValueError, match="which is off for A"):
                S.apply_knob(CFG, change, prot, True, 30.0, _values(on=False))
    with pytest.raises(ValueError, match="unknown knob"):
        S.apply_knob(CFG, ("what", 1), prot, True, 30.0, v)


def test_compare_is_symmetric_in_its_pairs_and_zero_for_the_same_run():
    M = np.zeros((16, 16)); M[0, 1] = M[1, 0] = 5; M[2, 3] = M[3, 2] = 2
    N = M.copy(); N[4, 5] = N[5, 4] = 1; N[2, 3] = N[3, 2] = 0
    a = types.SimpleNamespace(matrix=M, score=dict(pearson_count=0.9, pearson_area=0.8))
    b = types.SimpleNamespace(matrix=N, score=dict(pearson_count=0.85, pearson_area=0.8))
    c = D.compare(a, b)
    assert c["only_a"] == 1 and c["only_b"] == 1 and abs(c["delta_count"] + 0.05) < 1e-12 and c["delta_area"] == 0
    same = D.compare(a, a)
    assert same["only_a"] == 0 and same["pearson_ab"] == 1.0


PRISMA = "Siemens Prisma 3 T"
SWOOP = next(k for k in P.machines(CFG) if k.startswith("Hyperfine Swoop"))


def test_the_gradient_table_names_the_shell_the_scanner_cannot_play():
    prot = A._protocol_from_inputs(S, CFG, A.CUSTOM, 2, *ROWS)
    table, ok = A.gradient_text(CFG, prot, CFG["shapes"], PRISMA)
    assert "70 mT/m" in table and "194 mT/m" in table and "cannot play" in table and not ok      # shell 2: b 3000 at δ 8 / Δ 20
    assert A.gradient_text(CFG, prot, CFG["shapes"], A.IDEAL)[1]
    with pytest.raises(ValueError, match="no scanner"):
        A.gradient_text(CFG, prot, CFG["shapes"], "a magnet nobody built")


def test_the_scanner_menu_names_every_term_and_where_it_comes_from():
    """The term table of each machine: the Swoop applies every term (its field law measured, its nonlinearity's
    diagonal derived from a measurement), the cylinders their class-model nonlinearity and the Maxwell term, with
    their own gradient and transmit map absent and said so; the ideal scanner none."""
    assert list(P.machines(CFG).values()) == ["hyperfine_swoop_64mT", "siemens_magnetom_prisma_3T", "siemens_magnetom_terra_7T"]
    sw = {t: (on, src) for t, on, src in P.scanner_terms("hyperfine_swoop_64mT")}
    assert all(on for on, _ in sw.values()) and "0.064 T along (0, 1, 0)" in sw["field strength and direction"][1]
    assert "measured" in sw["transmit scale B1"][1] and "derived from a measurement" in sw["gradient nonlinearity L"][1]
    pr = {t: (on, src) for t, on, src in P.scanner_terms("siemens_magnetom_prisma_3T")}
    assert pr["gradient nonlinearity L"][0] and "inferred from the class" in pr["gradient nonlinearity L"][1]
    assert not pr["field law: its gradient g0 (and its value, a phase per voxel)"][0] and "absent" in pr["field law: its gradient g0 (and its value, a phase per voxel)"][1]
    assert not pr["transmit scale B1"][0] and "0.7-1.2" in pr["transmit scale B1"][1] and pr["Maxwell (concomitant) gradient"][0]
    assert "ideal" in P.scanner_text(CFG, A.IDEAL) and "| transmit scale B1 | no |" in P.scanner_text(CFG, PRISMA)


def test_a_machine_sets_the_field_its_direction_and_the_placement():
    """DiSCo's physics on a machine: the machine's field and direction whatever the panel says, the phantom's centre at
    the panel's distance along its axis, the bare diffusion on the machine when the tissue is off; the ideal scanner
    keeps the panel's field and no placement."""
    v = dict(_values(), field_T=1.5, offset_cm=7.9, offset_axis="A-P (y)")
    ph = S.physics_from(CFG, v, scanner="hyperfine_swoop_64mT")
    assert ph.field_T == 0.064 and ph.b0_direction == (0.0, 1.0, 0.0) and np.allclose(ph.offset_m, (0.0, 0.079, 0.0))
    assert ph.scanner == "hyperfine_swoop_64mT" and "hyperfine_swoop_64mT" in ph.label()
    off = S.physics_from(CFG, dict(v, on=False), scanner="siemens_magnetom_prisma_3T")
    assert off.bare and off.scanner == "siemens_magnetom_prisma_3T" and off.field_T == 3.0
    assert S.physics_from(CFG, dict(v, on=False)) is None
    ideal = S.physics_from(CFG, v)
    assert ideal.field_T == 1.5 and ideal.scanner is None and ideal.offset_m == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="T magnet"):
        P.Physics.at(3.0, scanner="hyperfine_swoop_64mT")


class _Demo(D.Disco):
    """A demo source without replay data: validates as a tiered layout would."""
    mode = "demo"
    tiers = True

    def validate(self, protocol, physics):
        if physics and not physics.bare and not self.tiers:
            raise ValueError("bare only")
        if physics and set(physics.pools) != set(self.pools):
            raise ValueError("pools")
        return P.measurements(protocol, self.shapes)


def test_the_runs_are_planned_and_refused_before_any_work():
    """plan_runs: A alone, A + B with a knob, the scanner's refusal naming the run, the source's refusal, and the
    upload refused in demo mode."""
    src = _Demo(CFG)
    v = _values()
    runs = A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, A.IDEAL, v, ROWS)
    assert [r[0] for r in runs] == ["A"] and runs[0][4].field_T == 3.0
    runs = A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, "SNR → 10", A.IDEAL, v, ROWS)
    assert [r[0] for r in runs] == ["A", "B"] and runs[1][3] == 10.0 and runs[1][1] is runs[0][1]
    with pytest.raises(ValueError, match="cannot play run A"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, SWOOP, v, ROWS)
    with pytest.raises(ValueError, match="cannot play run B"):
        A.plan_runs(CFG, src, "clinical b1000 x 30", 1, True, 30.0, None, "every shell's pulse timing → d8-D20 (Connectome 2.0 δ 8 / Δ 20 ms)", PRISMA, v, ROWS)
    with pytest.raises(ValueError, match="fixes the field"):
        A.plan_runs(CFG, src, "clinical b1000 x 30", 1, True, 30.0, None, "field → 7 T (catalogue tissue at that field)", PRISMA, v, ROWS)
    runs = A.plan_runs(CFG, src, "clinical b1000 x 30", 1, True, 30.0, None, A.NO_KNOB, PRISMA, v, ROWS)
    assert runs[0][4].scanner == "siemens_magnetom_prisma_3T" and runs[0][4].field_T == 3.0
    src.tiers = False
    with pytest.raises(ValueError, match="bare only"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, A.IDEAL, v, ROWS)
    with pytest.raises(ValueError, match="full mode"):
        A.plan_runs(CFG, src, A.UPLOADED, 2, True, 30.0, "x.scheme", A.NO_KNOB, A.IDEAL, _values(on=False), ROWS)
    with pytest.raises(ValueError, match="unknown knob"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, "twist", A.IDEAL, v, ROWS)


def test_disco_prepares_nothing_so_the_lookup_and_the_worker_have_nothing_to_do(monkeypatch):
    """DiSCo has no share outside the device: prepare_runs gives None for every entry, the worker computes nothing and
    says nothing, the payload has no response seconds and the reservation has no response part."""
    src = _Demo(CFG)
    monkeypatch.setattr(A, "_load", lambda: dict(error=None, source=src, cfg=CFG))
    v = _values()
    args = [A.CUSTOM, 2, True, 30.0, 2, 30.0, 0.5, 0, None, "SNR → 10", A.IDEAL, 1, True, *v.values(), *ROWS]
    prepared = A.prepare_runs(*args)
    assert prepared["runs"] == {"A": None, "B": None} and all(p is None for p in prepared["ladder"])
    runs = A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, "SNR → 10", A.IDEAL, v, ROWS)
    gen = A._fill(src, runs, True, prepared)
    with pytest.raises(StopIteration) as stop:
        next(gen)
    assert stop.value.value == (prepared, {}, None)
    assert src.responses([(m, ph) for _, m, ph in A.response_entries(src, runs, True)]) == []


def test_the_estimated_seconds_grow_with_the_run_and_stay_in_the_pools_window():
    v = list(_values().values())
    one = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 1, False, *v, *ROWS)
    two = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, "SNR → 10", A.IDEAL, 1, False, *v, *ROWS)
    keys = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 8, False, *v, *ROWS)
    ladder = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 1, True, *v, *ROWS)
    small = A.estimated_seconds("clinical b1000 x 30", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 1, False, *v, *ROWS)
    assert 30 <= small < one < two <= 480 and one < keys <= 480 and one < ladder <= 480
    both = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, "SNR → 10", A.IDEAL, 1, True, *v, *ROWS)
    assert one < ladder < both <= 120          # DiSCo alone, with the ladder, and A + B + ladder all fit a logged-out visitor's quota
    assert A.estimated_seconds("nonsense", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 1, True, *v, *ROWS) == 480
    text = A.gpu_seconds_text("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.IDEAL, 1, True, *v, *ROWS)
    assert f"reserves {ladder} s" in text and "logged out" in text.split("started by")[1].split(".")[0]
    assert len(A.OUTPUTS) == 30


def test_the_explorer_draws_layers_differences_and_metrics_from_its_state():
    """A two-layer explorer state (bare, A) with B: every mode and quantity returns an image; the table has one row
    per layer step per shell plus the floor; an empty state draws nothing."""
    p = P.Protocol((P.Shell("d12-D24", 1000, 12),), n_b0=1)
    m = P.measurements(p, CFG["shapes"])
    rng = np.random.default_rng(0)
    bare = rng.uniform(0.3, 1.0, (4, 4, 2, 13)).astype(np.float16); a = (bare * 0.98).astype(np.float16); b = (bare * 0.97).astype(np.float16)
    md = rng.random((4, 4, 2)).astype(np.float16); fa = rng.random((4, 4, 2)).astype(np.float16)
    ex = dict(layers=[("bare diffusion", bare), ("A = bare + field", a), ("B = A with field → 7 T", b)], metrics={k: (md, fa) for k in ("bare diffusion", "A = bare + field", "B = A with field → 7 T")},
              floor=np.full((4, 4, 2), 0.01, np.float32), meas=m, mask=np.ones((4, 4, 2), bool),
              ingredients=S.ingredient_layers(dict(intra_fraction=rng.random((4, 4, 2)), wall_contact_um=rng.random((4, 4, 2)), contact_survival=None, field_rad=None, D_walk=6e-10)),
              differences=P.layer_differences([("bare diffusion", bare.astype(np.float32)), ("A", a.astype(np.float32)), ("B", b.astype(np.float32))], m, np.ones((4, 4, 2), bool)),
              snr=None, physics=None)
    for mode in A.EXPLORE_MODES:
        for metric in A.METRICS:
            img = A.explore(ex, "A = bare + field", mode, metric, 1, 5)
            assert img is not None and img.size[0] > 100, (mode, metric)
    assert A.explore(ex, "bare diffusion", "minus the previous layer", A.METRICS[0], 0, 0) is not None
    pool, contact, fld = A.ingredient_views(ex, 1)
    assert pool is not None and contact is not None and fld is None
    rows = A.layer_table(ex)
    assert len(rows) == 2 + 1 and rows[-1][0] == "replay floor"
    assert A._layer_labels(P.Physics.at(3.0, contact=False), "SNR → 10") == ("A = bare + relaxation + field", "B = A with SNR → 10")
    assert A._layer_labels(None, A.NO_KNOB)[0] == "A = bare diffusion"
    assert A.explore(None, "A", "signal", A.METRICS[0], 0, 0) is None and A.ingredient_views(None, 0) == (None, None, None) and A.layer_table(None) == []


def test_the_run_button_chains_the_devices_texts_into_the_headline_then_the_pages_outputs(monkeypatch):
    """run_pipeline: every (text, fraction) of the compute function becomes a headline-only update (the progress bar
    following), the payload goes to present once the device part has ended, and its outputs are the last yield."""
    pytest.importorskip("gradio")
    kept = []
    source = types.SimpleNamespace(keep=kept.append)
    monkeypatch.setattr(A, "_load", lambda: dict(error=None, load_seconds=1.0, regions=None, source=source))
    monkeypatch.setattr(A, "prepare_runs", lambda *args: dict(runs={"A": args}, ladder=None))
    seen = []
    monkeypatch.setattr(A, "present", lambda payload, state: seen.append(payload) or ("drawn",) * len(A.OUTPUTS))

    def fake_compute(prepared, *args):
        assert prepared["runs"]["A"] == args
        yield ("**A · 1/5 replay** …", 0.1)
        yield ("**A · 2/5 noise** …", 0.3)
        yield dict(results={}, compute_seconds=12.0, kernels={"key": "computed on the worker"})
    bars = []
    out = list(A.run_pipeline(fake_compute, "x", 1, progress=lambda f, desc: bars.append((f, desc))))
    assert "starting" in out[0][1] and [o[1] for o in out[1:3]] == ["**A · 1/5 replay** …", "**A · 2/5 noise** …"] and all(len(o) == len(A.OUTPUTS) for o in out[:4])
    assert "drawing" in out[3][1] and out[4] == ("drawn",) * len(A.OUTPUTS)
    assert bars == [(0.0, "starting"), (0.1, "A · 1/5 replay"), (0.3, "A · 2/5 noise")] and seen[0]["compute_seconds"] == 12.0
    assert kept == [{"key": "computed on the worker"}]          # the page keeps what the worker computed, before drawing


def test_the_payload_carries_float32_volumes():
    """_light: the DWI and the clean replay cross the fork as float32, everything else of the result as it was."""
    p = P.Protocol((P.Shell("d12-D24", 1000, 6),), n_b0=1)
    m = P.measurements(p, CFG["shapes"])
    dwi = np.random.default_rng(0).random((2, 2, 2, 7)); floor = np.zeros((2, 2, 2))
    res = P.Result(p, m, dwi, floor, np.ones((2, 2, 2)), np.zeros((2, 2, 2, 45)), None, None, np.zeros((16, 16)), {}, clean=dwi * 0.5, snr=30.0)
    light = A._light(res)
    assert light.dwi.dtype == np.float32 and light.clean.dtype == np.float32 and res.dwi.dtype == np.float64
    np.testing.assert_allclose(light.dwi, dwi, rtol=1e-6); assert light.snr == 30.0 and light.meas is m
    assert A._light(P.Result(p, m, dwi, floor, None, None, None, None, None, {})).clean is None
