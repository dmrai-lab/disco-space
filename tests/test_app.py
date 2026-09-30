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


def test_the_physics_panel_is_a_physics_in_si_or_none_when_off():
    nums = A.catalogue_numbers(3.0)
    assert nums[-1].startswith("catalogue values at 3 T") and A.catalogue_numbers(0.064)[-1].startswith("the catalogue has no cited")
    inputs = [True, 3.0, next(iter(P.B0_PRESETS)), 0, 0] + nums[:9] + [True, True, False]
    ph = A._physics_from_inputs(CFG, *inputs)
    assert ph.field_T == 3.0 and ph.T2["intra"] == nums[0] * 1e-3 and ph.rho == nums[6] * 1e-6 and not ph.field and ph.relaxation
    assert A._physics_from_inputs(CFG, False, *inputs[1:]) is None
    free = A._physics_from_inputs(CFG, *([True, 7.0, A.FREE_B0, 90, 90] + nums[:9] + [True, True, True]))
    assert abs(free.b0_direction[1] - 1.0) < 1e-12


def test_a_knob_changes_one_thing_of_a():
    """Every knob maps to B = A with that one change: the field takes the catalogue tissue with it, a tier goes off,
    the tissue goes off, the noise changes, every shell is retimed; A's other settings survive."""
    nums = A.catalogue_numbers(3.0)[:9]
    ph = [True, 3.0, next(iter(P.B0_PRESETS)), 0, 0] + nums + [True, True, True]
    prot = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS)
    ks = A.knobs(CFG)
    assert ks[A.NO_KNOB] is None and len(ks) == 1 + 5 + 2 + 3 + 1 + 3 + len(CFG["shapes"])
    for name, change in ks.items():
        if change is None:
            continue
        pb, on, snr, phb = A.apply_knob(CFG, change, prot, True, 30.0, ph)
        diffs = [i for i in range(len(ph)) if phb[i] != ph[i]]
        kind = change[0]
        if kind == "field":
            assert phb[1] == change[1] and phb[5:14] == A.catalogue_numbers(change[1])[:9] and pb is prot
        elif kind == "b0":
            assert diffs == [2] or (diffs == [] and change[1] == ph[2])
        elif kind == "tier":
            assert diffs == [change[1]] and phb[change[1]] is False
        elif kind == "bare":
            assert diffs == [0] and A._physics_from_inputs(CFG, *phb) is None
        elif kind == "snr":
            assert diffs == [] and (on, snr) == ((False, 30.0) if change[1] is None else (True, change[1]))
        elif kind == "shape":
            assert diffs == [] and all(s.shape == change[1] for s in pb.shells) and [s.b for s in pb.shells] == [s.b for s in prot.shells]
    with pytest.raises(ValueError, match="unknown knob"):
        A.apply_knob(CFG, ("what", 1), prot, True, 30.0, ph)


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
