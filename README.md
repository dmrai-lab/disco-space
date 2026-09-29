---
title: DiSCo replay to tractogram
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: DiSCo replay, any acquisition, a connectome, while you wait
---

# The DiSCo Space

The application behind the Hugging Face Space **SubstrateCommons/disco** (dmrai-lab/dmipy-sim#505): the Docker
image, the pinned dependency stack, the Gradio page, the acceptance gate, and the deploy workflow. The libraries
(dmipy-sim, dmipy-fit, dmipy-tract) stay free of all of it; this repository is the one place the world is pinned.

**What it does.** The DiSCo phantom's walkers were simulated once and published as a replay pack
([SubstrateCommons/disco-replay](https://huggingface.co/datasets/SubstrateCommons/disco-replay)). A user picks an
acquisition (DiSCo's own 364-measurement protocol, a preset, or their own shells on the stored timing classes), an
SNR and the tracker's settings; the Space replays the whole 40³ grid from the stored walk, adds Rician noise, fits
CSD (order 8, the single-fibre response from the volume), tracks probabilistically from the sixteen regions and
scores the 16 × 16 connectome against the dataset's strand-count and area matrices. Every stage's time is on screen.

**The data path.** `disco/moments/` on the dataset is dmipy-sim's *shape-moment layout*: the replay pack contracted
once against each PGSE timing class (δ/Δ at TE 53.5 ms, square pulses), so that any b-value and direction on a
class is a three-term phase per walker. Six classes are stored (`space/config.toml`, `[shapes]`); the layout is
written by `dmipy_sim.replay.shape_moments.write_shape_moments` from the certified columnar rows and names the
columnar manifest's sha256 it was contracted from. The replay of DiSCo's protocol from the layout is checked
against the published reference volume by the gate.

## Layout

```
Dockerfile, entrypoint.sh   python:3.11-slim, the pinned stack, ldconfig for the CUDA wheels, caches on /data
requirements.txt            the pins (dmipy packages by commit)
space/config.toml           dataset + layout revision, the timing classes, DiSCo's shells, presets, tracking, the gate
space/pipeline.py           replay -> noise -> CSD -> track -> score, plain functions with timings
space/app.py                Gradio Blocks over pipeline.py
data/                       DiSCo mask, regions, gradient table, ground-truth matrices (CC BY 4.0, see SOURCE.md)
tests/test_pipeline.py      CPU: protocol construction, DiSCo's table, the score on the ground truth itself
tests/test_acceptance.py    GPU + data: the gate (Pearson, missed pairs, the reference volume, timings)
.github/workflows/gate.yml  the gate as a Hugging Face Job on an L4 (GitHub runners have no GPU)
```

## Run it

```bash
pip install -r requirements.txt
python -m space.app                              # http://localhost:7860; the layout downloads into the Hub cache
pytest tests/test_pipeline.py                    # CPU
JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_deterministic_ops=true pytest tests/test_acceptance.py -s   # the gate, on a GPU
```

**Determinism.** The image runs with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (the Dockerfile sets it): without it
the GPU scatter-add of the replay differs between runs by up to 5e-7 on the normalised signal, which the CSD
amplifies to 1e-2 on the SH coefficients and the tracker into a few different streamlines (Pearson moved by 1e-5).
With it two runs are bit-identical at a 7 % cost on the replay (2.48 s against 2.31 s on the L40S).

## Measured

The gate on the DiSCo 364 protocol, noiseless, 659,840 seeds (`tests/test_acceptance.py`, deterministic XLA ops):

| stage | L40S first call (s) | L40S steady (s) | L4 first call (s) | L4 steady (s) |
|---|---|---|---|---|
| layout on the device (once per process, from a local copy) | 8.1 | | | |
| replay, 364 measurements on 3 timing classes | 2.5 | 2.4 | | |
| CSD (response + `csd_tournier07_jax`, order 8) | 4.6 | 2.9 | | |
| tracking (probabilistic, density 4) | 18.7 | 2.5 | | |
| whole pipeline | 26.0 | 7.9 | | |

Score: Pearson 0.927 vs strand count, 0.929 vs cross-sectional area; 120 of 120 pairs connected (25 in the ground
truth: 95 false, 0 missed). The replay of the DiSCo 364 protocol from the layout is within 2.0e-7 of the published
reference volume on every voxel (K = 64 bands, band error 1.4e-4 against the pack's floor 4.8e-3). The L4 columns
are filled by the gate job (`tools/gate_job.py`).
