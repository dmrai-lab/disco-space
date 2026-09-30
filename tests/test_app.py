"""The page's acquisition choice without a browser: DiSCo's table, a config preset, the shell rows (a stored class in
demo mode, the row's own delta / Delta / TE in full mode), a scheme upload only in full mode; an unknown choice
refused (no Gradio needed: the mapping lives beside the page, not in a widget)."""
import numpy as np
import pytest

from space import app as A
from space import pipeline as P

CFG = P.config()
ROWS = [True, "d12-D24", 1000, 30, 10.0, 20.0, 60.0, True, "d8-D20", 3000, 45, 8.0, 20.0, 60.0,
        False, "d17-D30", 3000, 90, 17.0, 30.0, 60.0, False, "d17-D30", 6000, 60, 17.0, 30.0, 60.0]


def test_the_three_kinds_of_acquisition():
    disco = A._protocol_from_inputs(CFG, "DiSCo 364", 5, *ROWS)
    assert disco.name == "DiSCo 364" and disco.n_meas == 364 and disco.n_b0 == 4        # the table's own, not the slider's
    name = next(iter(CFG["presets"]))
    preset = A._protocol_from_inputs(CFG, name, 5, *ROWS)
    assert preset.name == name and preset.n_b0 == CFG["presets"][name]["n_b0"]
    assert [s.shape for s in preset.shells] == [s["shape"] for s in CFG["presets"][name]["shells"]]
    custom = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS)
    assert custom.n_b0 == 2 and [(s.shape, s.b, s.n_dirs) for s in custom.shells] == [("d12-D24", 1000.0, 30), ("d8-D20", 3000.0, 45)]
    assert not any(s.free_timing for s in custom.shells)
    with pytest.raises(ValueError, match="unknown acquisition"):
        A._protocol_from_inputs(CFG, "something else", 2, *ROWS)
    with pytest.raises(ValueError, match="at least one shell"):
        A._protocol_from_inputs(CFG, A.CUSTOM, 2, *([False] + ROWS[1:7] + [False] + ROWS[8:14] + ROWS[14:]))


def test_full_mode_takes_the_rows_own_timing_and_demo_mode_refuses_the_upload():
    full = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS, full=True)
    assert all(s.free_timing for s in full.shells)
    assert [(s.delta, s.Delta, s.TE) for s in full.shells] == [(0.010, 0.020, 0.060), (0.008, 0.020, 0.060)]
    with pytest.raises(ValueError, match="full mode"):
        A._protocol_from_inputs(CFG, A.UPLOADED, 2, *ROWS, scheme="x.scheme")
    with pytest.raises(ValueError, match="upload"):
        A._protocol_from_inputs(CFG, A.UPLOADED, 2, *ROWS, scheme=None, full=True)


def _values(on=True, field_T=3.0, **over):
    nums = A.catalogue_numbers(field_T)[:-1]
    v = dict(zip(A.PHYSICS_FIELDS, [on, field_T, next(iter(A.B0_MODES)), 0, 0] + nums + [True, True, True]))
    v.update(over)
    return v


def test_the_physics_panel_is_a_physics_in_si_or_none_when_off():
    nums = A.catalogue_numbers(3.0)
    assert nums[-1].startswith("catalogue values at 3 T") and A.catalogue_numbers(0.064)[-1].startswith("the catalogue has no cited")
    ph = A.physics_from(_values(field=False))
    assert ph.field_T == 3.0 and ph.T2["intra"] == nums[0] * 1e-3 and ph.rho == nums[4] * 1e-6 and not ph.field and ph.relaxation and ph.pools == A.POOLS
    assert A.physics_from(_values(on=False)) is None
    free = A.physics_from(_values(field_T=7.0, b0_mode=A.FREE_B0, theta=90, phi=90))
    assert abs(free.b0_direction[1] - 1.0) < 1e-12
    with pytest.raises(ValueError, match="15 inputs"):
        A.physics_values(1, 2, 3)


def test_a_knob_changes_one_thing_of_a():
    """Every knob maps to B = A with that one change: the field takes the catalogue tissue with it, a tier goes off,
    the tissue goes off, the noise changes, every shell is retimed; A's other settings survive; a panel knob on an
    A whose panel is off is refused (B would differ in two things)."""
    v = _values()
    prot = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS)
    ks = A.knobs(CFG)
    assert ks[A.NO_KNOB] is None and len(ks) == 1 + 5 + 2 + 3 + 1 + 3 + len(CFG["shapes"])
    for name, change in ks.items():
        if change is None:
            continue
        pb, on, snr, vb = A.apply_knob(change, prot, True, 30.0, v)
        diffs = [k for k in A.PHYSICS_FIELDS if vb[k] != v[k]]
        kind = change[0]
        if kind == "field":
            assert vb["field_T"] == change[1] and [vb[k] for k in A.TISSUE_NUMBERS] == A.catalogue_numbers(change[1])[:-1] and pb is prot
        elif kind == "b0":
            assert diffs == ["b0_mode"] or (diffs == [] and change[1] == v["b0_mode"])
        elif kind == "tier":
            assert diffs == [change[1]] and vb[change[1]] is False
        elif kind == "bare":
            assert diffs == ["on"] and A.physics_from(vb) is None
        elif kind == "snr":
            assert diffs == [] and (on, snr) == ((False, 30.0) if change[1] is None else (True, change[1]))
        elif kind == "shape":
            assert diffs == [] and all(s.shape == change[1] for s in pb.shells) and [s.b for s in pb.shells] == [s.b for s in prot.shells]
        if kind in ("field", "b0", "tier"):
            with pytest.raises(ValueError, match="which is off for A"):
                A.apply_knob(change, prot, True, 30.0, _values(on=False))
    with pytest.raises(ValueError, match="unknown knob"):
        A.apply_knob(("what", 1), prot, True, 30.0, v)


def test_compare_is_symmetric_in_its_pairs_and_zero_for_the_same_run():
    import types
    M = np.zeros((16, 16)); M[0, 1] = M[1, 0] = 5; M[2, 3] = M[3, 2] = 2
    N = M.copy(); N[4, 5] = N[5, 4] = 1; N[2, 3] = N[3, 2] = 0
    a = types.SimpleNamespace(matrix=M, score=dict(pearson_count=0.9, pearson_area=0.8))
    b = types.SimpleNamespace(matrix=N, score=dict(pearson_count=0.85, pearson_area=0.8))
    c = P.compare(a, b)
    assert c["only_a"] == 1 and c["only_b"] == 1 and abs(c["delta_count"] + 0.05) < 1e-12 and c["delta_area"] == 0
    same = P.compare(a, a)
    assert same["only_a"] == 0 and same["pearson_ab"] == 1.0


def test_the_gradient_table_names_the_shell_the_scanner_cannot_play():
    prot = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS)
    table, ok = A.gradient_text(prot, CFG["shapes"], "prisma")
    assert "70 mT/m" in table and "194 mT/m" in table and "cannot play" in table and not ok      # shell 2: b 3000 at δ 8 / Δ 20
    assert A.gradient_text(prot, CFG["shapes"], "connectom")[1] and A.gradient_text(prot, CFG["shapes"], A.NO_SCANNER)[1]


class _Demo(P.Source):
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
    runs = A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, A.NO_SCANNER, v, ROWS)
    assert [r[0] for r in runs] == ["A"] and runs[0][4].field_T == 3.0
    runs = A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, "SNR → 10", A.NO_SCANNER, v, ROWS)
    assert [r[0] for r in runs] == ["A", "B"] and runs[1][3] == 10.0 and runs[1][1] is runs[0][1]
    with pytest.raises(ValueError, match="cannot play run A"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, "low_field", v, ROWS)
    with pytest.raises(ValueError, match="cannot play run B"):
        A.plan_runs(CFG, src, "clinical b1000 x 30", 1, True, 30.0, None, "every shell's pulse timing → d8-D20 (Connectome 2.0 δ 8 / Δ 20 ms)", "prisma", v, ROWS)
    src.tiers = False
    with pytest.raises(ValueError, match="bare only"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, A.NO_KNOB, A.NO_SCANNER, v, ROWS)
    with pytest.raises(ValueError, match="full mode"):
        A.plan_runs(CFG, src, A.UPLOADED, 2, True, 30.0, "x.scheme", A.NO_KNOB, A.NO_SCANNER, _values(on=False), ROWS)
    with pytest.raises(ValueError, match="unknown knob"):
        A.plan_runs(CFG, src, A.CUSTOM, 2, True, 30.0, None, "twist", A.NO_SCANNER, v, ROWS)


def test_the_estimated_seconds_grow_with_the_run_and_stay_in_the_pools_window():
    v = list(_values().values())
    one = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 1, False, *v, *ROWS)
    two = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, "SNR → 10", A.NO_SCANNER, 1, False, *v, *ROWS)
    keys = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 8, False, *v, *ROWS)
    ladder = A.estimated_seconds("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 1, True, *v, *ROWS)
    small = A.estimated_seconds("clinical b1000 x 30", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 1, False, *v, *ROWS)
    assert 60 <= small < one < two <= 480 and one < keys <= 480 and one < ladder <= 480
    assert one <= 120 < ladder <= 300          # DiSCo alone fits a logged-out visitor's quota; with the ladder a free account's
    assert A.estimated_seconds("nonsense", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 1, True, *v, *ROWS) == 480
    text = A.gpu_seconds_text("DiSCo 364", 1, True, 30, 2, 30.0, 0.5, 0, None, A.NO_KNOB, A.NO_SCANNER, 1, True, *v, *ROWS)
    assert f"reserves {ladder} s" in text and "free account" in text and "logged out" not in text.split("started by")[1].split(".")[0]
    assert len(A.OUTPUTS) == 25


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
              ingredients=dict(intra_fraction=rng.random((4, 4, 2)), wall_contact_um=rng.random((4, 4, 2)), contact_survival=None, field_rad=None, D_walk=6e-10),
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
