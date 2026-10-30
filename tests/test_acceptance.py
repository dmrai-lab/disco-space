"""The gate: the page's replay from the time-segment layout equals the replay pack's own replay of the same rows, and
the noiseless pipeline scores the connectome within the reference's band with no ground-truth pair missed.

The reference (:mod:`space.reference`) is the replay pack the layout was contracted from (``[data] pack``), replayed
walker by walker -- every band, every field mode, the contact -- at ``[gate] reference_voxels`` voxels spread over its
index; the page's signal at those voxels, over the voxel's bare b = 0 signal (the quantity both certificates bound;
S/S0 divides it by the relaxed b = 0 signal, down to 0.3 at 7 T, and is recorded beside it), must be within
``[gate] reference_max_abs_diff`` of the reference's, for DiSCo's
own protocol bare and in the catalogue's white matter at 3 T and 7 T, and for every preset (the paper's flagship
acquisitions, the cosine OGSE among them) bare and at 3 T. It runs where the layout is (``DISCO_LAYOUT`` a local copy,
else the Hub) and reads the pack's rows at the gate's voxels (``DISCO_PACK`` a local copy, else the Hub); on a CPU
host every image streams the layout chunk by chunk (minutes per class), so it runs when asked:

    DISCO_GATE=1 DISCO_LAYOUT=/local/layout DISCO_PACK=/local/coefficients pytest tests/test_acceptance.py -s
    DISCO_GATE=1 JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_deterministic_ops=true pytest tests/test_acceptance.py -s

The per-case worst difference and timings go to ``gate.json`` (``GATE_OUT``).
"""
import json
import os
import time

import numpy as np
import pytest

from space import pipeline as P
from space import reference as R
from space import sources
from space.sources import disco as D

CFG = P.config()
OUT = os.environ.get("GATE_OUT", os.path.join(P.ROOT, "gate.json"))
pytestmark = pytest.mark.skipif(not os.environ.get("DISCO_GATE"), reason="the gate runs when asked (DISCO_GATE=1)")


def _record(key, value):
    table = json.load(open(OUT)) if os.path.exists(OUT) else {}
    table[key] = value
    with open(OUT, "w") as f:
        json.dump(table, f, indent=1, default=float)


@pytest.fixture(scope="module")
def layout():
    t0 = time.perf_counter()
    lay = sources.source(CFG)
    lay.load_seconds = time.perf_counter() - t0
    return lay


@pytest.fixture(scope="module")
def pack():
    pk = R.open_pack(os.environ.get("DISCO_PACK") or CFG["data"]["pack"])
    return pk, R.gate_voxels(pk, CFG["gate"]["reference_voxels"])


CASES = [(D.DISCO_364, run) for run in ("bare", "3T", "7T")] + [(name, run) for name in CFG["presets"] for run in ("bare", "3T")]


@pytest.mark.parametrize("preset,run", CASES)
def test_the_layout_is_the_replay_packs_replay_at_the_gates_voxels(layout, pack, preset, run):
    """At the gate's voxels: the page's replay through the layout (its classes' waveforms, its tiers) is the pack's replay
    of the same rows, to ``reference_max_abs_diff`` in units of the voxel's bare b = 0 signal."""
    pk, voxels = pack
    protocol = D.Disco.protocol(CFG, preset)
    physics = None if run == "bare" else P.Physics.at(float(run[:-1]))
    meas = layout.validate(protocol, physics)
    t0 = time.perf_counter()
    dwi, floor, factor, secs = layout.replay(meas, physics)
    tissue, scanner = P.tissue_and_scanner(physics, layout.unseeded)
    ref = np.zeros((len(voxels), len(meas.bvals)))
    t1 = time.perf_counter()
    for name in np.unique(meas.shape):
        rows = meas.shape == name
        ref[:, rows] = R.replay_at(pk, voxels, layout.sequence(name, meas.bvals[rows], meas.dirs[rows]), tissue, scanner)
    t_ref = time.perf_counter() - t1
    ref_S0 = ref[:, meas.b0].mean(1, keepdims=True)
    got = np.stack([dwi[tuple(v)] for v in voxels])
    d = np.abs(got - ref / ref_S0)                                   # S/S0
    d_S = np.abs(got * ref_S0 - ref)                                 # the signal over the voxel's bare b = 0 signal: what the certificates bound
    worst, worst_S = float(np.nanmax(d)), float(np.nanmax(d_S))
    print(f"\n{preset} {run}: max |dS| {worst_S:.2e} (S over the bare b = 0 signal), max |d(S/S0)| {worst:.2e} over "
          f"{len(voxels)} voxels x {len(meas.bvals)} measurements (S0 down to {float(ref_S0.min()):.3f}); "
          f"layout replay {secs:.1f} s, reference {t_ref:.1f} s")
    _record(f"{preset} | {run}", dict(worst_S=worst_S, worst_S_over_S0=worst, median_S=float(np.nanmedian(d_S)), S0_min=float(ref_S0.min()),
                                      voxels=len(voxels), n_meas=int(len(meas.bvals)), replay_seconds=secs, reference_seconds=t_ref,
                                      bound=CFG["gate"]["reference_max_abs_diff"], manifest_sha256=layout.manifest_sha256))
    assert np.isfinite(got).all(), "a gate voxel the pack has rows in has no signal from the layout"
    assert worst_S < CFG["gate"]["reference_max_abs_diff"]


def test_the_noiseless_pipeline_scores_within_the_references_band(layout):
    """Pearson vs strand count at or above the gate's floor and no ground-truth pair missed on DiSCo's own protocol;
    the timings recorded (first call with every compile, then steady)."""
    p, _ = D.disco_protocol(CFG)
    tr = P.Tracking(**{k: CFG["tracking"][k] for k in ("density", "step_mm", "max_angle", "max_steps")})
    first = P.run(layout, p, snr=None, tracking=tr)
    steady = P.run(layout, p, snr=None, tracking=tr)
    s = steady.score
    _record("pipeline", dict(layout_load_seconds=layout.load_seconds, first_call=first.seconds, steady=steady.seconds, score=s,
                             streamlines=len(steady.tractogram), seeds=int(len(steady.seeds)), n_meas=p.n_meas))
    assert s["pearson_count"] >= CFG["gate"]["pearson_count_min"], s
    assert s["missed_pairs"] <= CFG["gate"]["missed_pairs_max"], s
    assert first.score["pearson_count"] == s["pearson_count"]


@pytest.mark.parametrize("machine", [m["key"] for m in CFG["scanners"]["machines"]])
def test_a_machine_on_discos_protocol_moves_the_image(layout, machine):
    """Each machine of the menu at 7.9 cm from isocentre along R-L on DiSCo's own protocol against the ideal scanner at
    the machine's field and direction: the delivered image is finite where the ideal one is and moves it."""
    from dataclasses import replace
    L = P.limits(machine)
    p, _ = D.disco_protocol(CFG)
    ideal = P.Physics.at(float(L.field_T), b0_direction=tuple(L.b0_axis or (0.0, 0.0, 1.0)))
    far = replace(ideal, scanner=machine, offset_m=(0.079, 0.0, 0.0))
    m = layout.validate(p, far)
    ref, *_ = layout.replay(m, ideal)
    dwi, _, _, secs = layout.replay(m, far)
    both = np.isfinite(ref)
    assert (np.isfinite(dwi) == both).all()
    moved = float(np.max(np.abs(dwi - ref)[both]))
    _record(f"machine | {machine}", dict(max_abs_dS=moved, seconds=secs))
    assert moved > 1e-3
