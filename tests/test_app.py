"""The page's acquisition choice without a browser: DiSCo's table, a config preset, the shell rows; an unknown
choice refused (no Gradio needed: the mapping lives beside the page, not in a widget)."""
import pytest

from space import app as A
from space import pipeline as P

CFG = P.config()
ROWS = [True, "d12-D24", 1000, 30, True, "d8-D20", 3000, 45, False, "d17-D30", 3000, 90, False, "d17-D30", 6000, 60]


def test_the_three_kinds_of_acquisition():
    disco = A._protocol_from_inputs(CFG, "DiSCo 364", 5, *ROWS)
    assert disco.name == "DiSCo 364" and disco.n_meas == 364 and disco.n_b0 == 4        # the table's own, not the slider's
    name = next(iter(CFG["presets"]))
    preset = A._protocol_from_inputs(CFG, name, 5, *ROWS)
    assert preset.name == name and preset.n_b0 == CFG["presets"][name]["n_b0"]
    assert [s.shape for s in preset.shells] == [s["shape"] for s in CFG["presets"][name]["shells"]]
    custom = A._protocol_from_inputs(CFG, A.CUSTOM, 2, *ROWS)
    assert custom.n_b0 == 2 and [(s.shape, s.b, s.n_dirs) for s in custom.shells] == [("d12-D24", 1000.0, 30), ("d8-D20", 3000.0, 45)]
    with pytest.raises(ValueError, match="unknown acquisition"):
        A._protocol_from_inputs(CFG, "something else", 2, *ROWS)
    with pytest.raises(ValueError, match="at least one shell"):
        A._protocol_from_inputs(CFG, A.CUSTOM, 2, *([False] + ROWS[1:4] + [False] + ROWS[5:8] + ROWS[8:]))
