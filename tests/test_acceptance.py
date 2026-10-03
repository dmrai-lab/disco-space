"""The gate: on a GPU with the layout reachable, DiSCo's own protocol replayed from the layout equals the published
bare reference volume, and the noiseless pipeline scores the connectome within the reference's band with no ground-
truth pair missed. Prints the per-stage timings (first call and steady) and writes them to ``gate.json``.

    JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_deterministic_ops=true pytest tests/test_acceptance.py -s
    DISCO_MOMENTS=/local/dir pytest ...        # a local copy of the layout instead of the Hub
"""
import json
import os
import time

import numpy as np
import pytest

from space import pipeline as P
from space import sources
from space.sources import disco as D

CFG = P.config()
OUT = os.environ.get("GATE_OUT", os.path.join(P.ROOT, "gate.json"))


@pytest.fixture(scope="module")
def layout():
    import jax
    if jax.devices()[0].platform != "gpu":
        pytest.skip("the gate runs on a GPU")
    pytest.importorskip("huggingface_hub")
    t0 = time.perf_counter()
    lay = sources.source(CFG)
    lay.warm()
    lay.load_seconds = time.perf_counter() - t0
    return lay


@pytest.mark.parametrize("run", ["bare", "3T", "7T"])
def test_discos_protocol_from_the_layout_is_the_published_reference_volume(layout, run):
    """S0-normalised, on every voxel the reference has signal: the replay through the moments is the replay the
    reference was made with, to the float32 arithmetic of the layout; bare, and in the catalogue's white matter at
    3 T and 7 T with every tier on and the field along z (the references' setting)."""
    import nibabel as nib
    from huggingface_hub import hf_hub_download
    p, order = D.disco_protocol(CFG)                                   # order: the table's rows in the protocol's order
    physics = None if run == "bare" else P.Physics.at(float(run[:-1]))
    m = layout.validate(p, physics)
    dwi, floor, factor, secs = layout.replay(m, physics)
    ref_path = hf_hub_download(CFG["data"]["repo"], f"disco/reference/disco_replay_{run}.nii.gz", repo_type="dataset", revision=CFG["data"]["revision"])
    ref = np.asarray(nib.load(ref_path).dataobj, np.float64)[..., order]
    both = np.isfinite(dwi) & (ref != 0)
    d = np.abs(dwi - ref)[both]
    print(f"\nreference check {run}: max |dS| {d.max():.2e}, 99.9 % {np.quantile(d, 0.999):.2e}, median {np.median(d):.2e} "
          f"over {both.any(-1).sum()} voxels; replay {secs:.2f} s")
    assert (~np.isfinite(dwi) & (ref != 0)).any(-1).sum() == 0, "voxels the reference has signal in and the layout has no rows"
    assert d.max() < CFG["gate"]["reference_max_abs_diff"]


def test_the_noiseless_pipeline_scores_within_the_references_band(layout):
    """Pearson vs strand count at or above the gate's floor (the replay reference's band is 0.912 +- 0.003; the
    tracker on the reference volume measured 0.927) and no ground-truth pair missed; the timings recorded."""
    p, _ = D.disco_protocol(CFG)
    tr = P.Tracking(**{k: CFG["tracking"][k] for k in ("density", "step_mm", "max_angle", "max_steps")})
    first = P.run(layout, p, snr=None, tracking=tr)                    # includes every compile
    steady = P.run(layout, p, snr=None, tracking=tr)
    s = steady.score
    table = dict(device=str(__import__("jax").devices()[0]), layout_load_seconds=layout.load_seconds,
                 first_call=first.seconds, steady=steady.seconds, score=s, streamlines=len(steady.tractogram),
                 seeds=int(len(steady.seeds)), n_meas=p.n_meas)
    print("\ngate: " + json.dumps(table, indent=1))
    with open(OUT, "w") as f:
        json.dump(table, f, indent=1)
    assert s["pearson_count"] >= CFG["gate"]["pearson_count_min"], s
    assert s["missed_pairs"] <= CFG["gate"]["missed_pairs_max"], s
    assert first.score["pearson_count"] == s["pearson_count"]   # the same key: the same tractogram (deterministic XLA ops)


@pytest.mark.parametrize("machine", [m["key"] for m in CFG["scanners"]["machines"]])
def test_a_machine_on_discos_protocol_moves_the_image_and_is_timed(layout, machine):
    """Each machine of the menu at 7.9 cm from isocentre along R-L on DiSCo's own protocol (none of them can play it:
    the gradient limit is the page's, the replay's cost is measured here at the gate's protocol), against the ideal
    scanner at the machine's field and direction: the delivered image is finite where the ideal one is and moves it;
    the first and steady replay seconds of both go to ``gate_scanners.json`` beside ``gate.json``. A machine with its
    own gradient needs the layout's background moments and is skipped by name on a layout without them."""
    from dataclasses import replace
    L = P.limits(machine)
    if L.has_field_law and not layout.moments.background:
        pytest.skip(f"{machine}'s own gradient needs the layout's background moments (shape_moments.stamp_background)")
    p, _ = D.disco_protocol(CFG)
    ideal = P.Physics.at(float(L.field_T), b0_direction=tuple(L.b0_axis or (0.0, 0.0, 1.0)))
    far = replace(ideal, scanner=machine, offset_m=(0.079, 0.0, 0.0))
    m = layout.validate(p, far)
    secs = {}
    for tag, ph in (("ideal", ideal), ("machine", far)):
        for k in ("first", "steady"):
            dwi, _, factor, s = layout.replay(m, ph)
            secs[f"{tag}_{k}"] = s
        if tag == "ideal":
            ref = dwi
    both = np.isfinite(ref)
    assert (np.isfinite(dwi) == both).all()
    moved = float(np.max(np.abs(dwi - ref)[both]))
    print(f"\n{machine} at 7.9 cm on DiSCo 364: max |dS| vs ideal {moved:.3e}; seconds {json.dumps(secs)}")
    assert moved > 1e-3
    path = os.path.join(os.path.dirname(OUT), "gate_scanners.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    table[machine] = dict(seconds=secs, max_abs_dS=moved, n_meas=p.n_meas, offset_m=list(far.offset_m), device=str(__import__("jax").devices()[0]))
    with open(path, "w") as f:
        json.dump(table, f, indent=1)
