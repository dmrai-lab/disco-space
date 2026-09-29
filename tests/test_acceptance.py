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

CFG = P.config()
OUT = os.environ.get("GATE_OUT", os.path.join(P.ROOT, "gate.json"))


@pytest.fixture(scope="module")
def layout():
    import jax
    if jax.devices()[0].platform != "gpu":
        pytest.skip("the gate runs on a GPU")
    pytest.importorskip("huggingface_hub")
    t0 = time.perf_counter()
    lay = P.Layout(CFG, local=os.environ.get("DISCO_MOMENTS"))
    lay.warm()
    lay.load_seconds = time.perf_counter() - t0
    return lay


def test_discos_protocol_from_the_layout_is_the_published_reference_volume(layout):
    """S0-normalised, on every voxel the reference has signal: the replay through the moments is the replay the
    reference was made with, to the float32 arithmetic of the layout."""
    import nibabel as nib
    from huggingface_hub import hf_hub_download
    p, _ = P.disco_protocol(CFG)
    m = P.measurements(p, layout.shapes)
    dwi, floor, secs = P.replay(layout, m)
    ref_path = hf_hub_download(CFG["data"]["repo"], "disco/reference/disco_replay_bare.nii.gz", repo_type="dataset")
    ref = np.asarray(nib.load(ref_path).dataobj, np.float64)
    # the reference's rows follow the table's order; ours put the b = 0 rows first, then the shells in table order
    bv = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_gradients.bvals")).ravel()
    b0 = bv < 50
    centres = np.array([s["b"] for s in CFG["disco"]["shells"]])
    which = np.argmin(np.abs(bv[:, None] - centres[None, :]), axis=1)
    order = np.concatenate([np.flatnonzero(b0)] + [np.flatnonzero((which == k) & ~b0) for k in range(len(centres))])
    ref = ref[..., order]
    both = np.isfinite(dwi) & (ref != 0)
    d = np.abs(dwi - ref)[both]
    print(f"\nreference check: max |dS| {d.max():.2e}, 99.9 % {np.quantile(d, 0.999):.2e}, median {np.median(d):.2e} "
          f"over {both.any(-1).sum()} voxels; replay {secs:.2f} s")
    assert (~np.isfinite(dwi) & (ref != 0)).any(-1).sum() == 0, "voxels the reference has signal in and the layout has no rows"
    assert d.max() < CFG["gate"]["reference_max_abs_diff"]


def test_the_noiseless_pipeline_scores_within_the_references_band(layout):
    """Pearson vs strand count at or above the gate's floor (the replay reference's band is 0.912 +- 0.003; the
    tracker on the reference volume measured 0.927) and no ground-truth pair missed; the timings recorded."""
    p, _ = P.disco_protocol(CFG)
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
