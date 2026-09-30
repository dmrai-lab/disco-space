"""The page's acquisition choice without a browser: DiSCo's table, a config preset, the shell rows (a stored class in
demo mode, the row's own delta / Delta / TE in full mode), a scheme upload only in full mode; an unknown choice
refused (no Gradio needed: the mapping lives beside the page, not in a widget)."""
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
