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

The application behind the Hugging Face Space **rfick/disco** (to move to SubstrateCommons once the organisation has
compute credits; dmrai-lab/dmipy-sim#505): the Docker
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
Dockerfile, entrypoint.sh   python:3.11-slim, the pinned stack, ldconfig for the CUDA wheels, caches on /data (self-contained)
Dockerfile.base / .space    the same split in two: the stack as ghcr.io/dmrai-lab/disco-space (image.yml), the app on top
requirements.txt            the pins (dmipy packages by commit)
space/config.toml           dataset + layout revision, the timing classes, DiSCo's shells, presets, tracking, the gate
space/pipeline.py           replay -> noise -> CSD -> track -> score, plain functions with timings
space/app.py                Gradio Blocks over pipeline.py: three tabs (acquisition; ground truth: the strands in 3-D; results:
                            DWI slice viewer with FOD peaks, 3-D tractogram, connectome, timings, downloads); the progress bar names the stage
space/viewers.py            the figures (matplotlib slices and matrices, Plotly 3-D views); nothing derived here
data/                       DiSCo mask, regions, gradient table, ground-truth matrices and strands (CC BY 4.0, see SOURCE.md)
tests/test_pipeline.py      CPU: protocol construction, DiSCo's table, the score on the ground truth itself
tests/test_acceptance.py    GPU + data: the gate (Pearson, missed pairs, the reference volume, timings)
.github/workflows/gate.yml  the gate as a Hugging Face Job on an L4 (GitHub runners have no GPU)
.github/workflows/image.yml the base image to GHCR on every change to the pins (the package must be public for HF to pull it)
```

## Run it

Deploy on the prebuilt base: `python tools/deploy.py --base-tag <commit sha or latest>` (a Space build then copies files
instead of installing the stack; the gate job runs in the Space's own image either way). Measured 2026-09-29: the Space
build on `ghcr.io/dmrai-lab/disco-space` takes 91 s (82 s pulling the base layers) against 5-8 min installing the stack
from PyPI, and 61 s after the trim in `Dockerfile.base` (the unused nccl + nvshmem, cudnn's convolution engines and
the packages' test suites: compressed image 3.68 -> 2.42 GB; the gate in the trimmed image reproduces the Pearson to
the last digit, 0.9270634171621878, steady 18.4 s on an A10G). The app then needs another ~150 s to download the
13 GB layout and compile before it serves, which is the storage item, not the image's.

**Storage.** Hugging Face replaced the persistent-storage tiers with Storage Buckets mounted as volumes: the layout
sits in the private bucket `rfick/disco-space-data` (11.9 GB, copied server-side from the dataset with
`HfApi.copy_files` in 2 s, about $0.20 a month at $18 per TB), mounted at `/data` with `set_space_volumes`, and the
Space variable `DISCO_MOMENTS=/data/moments` points the app at it; the JAX compile cache lives on the same volume.
Measured 2026-09-29 (A10G): a wake without a build serves in 16 s (was ~250 s), and the first DiSCo 364 request after
that wake answers in 86 s wall (the layout read from the mount plus the run; 132 s on the first wake, when the compile
cache is still empty).

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

## ZeroGPU

`rfick/disco-zero` is the same application on Hugging Face's shared GPU pool (Gradio SDK, `zero-a10g`; the pool
handed an RTX PRO 6000 Blackwell MIG slice with 51 GB): `DISCO_BACKEND=torch` switches the layout image, the CSD
solver and the tracker to their PyTorch kernels (dmipy-sim#510, dmipy-fit#37, dmipy-tract#4), `spaces.GPU` wraps the
run, the layout is preloaded into the parent process from the mounted bucket and the forked GPU worker inherits it
(the mount reads at 80 MB/s, the host-to-device transfer at 8 GB/s), and nothing in the worker touches JAX (the noise
draw comes from numpy there; a JAX call in the forked worker aborts the task). Measured 2026-09-29 through the API:
DiSCo 364 at SNR 30 in 51 s wall, 21 s in the pipeline (replay 12.6 s with the per-call transfer and kernel compile,
noise 0.5, CSD 3.6, tracking 4.5); a custom two-shell protocol in 44 s wall, 18.6 s in the pipeline. No idle cost, no
sleep; per-visitor quotas instead of a queue on one card. Deploy with `tools/deploy.py --zero`
(`README-zero.md`, `requirements-zero.txt`, the root `app.py`).

## Measured

The gate on the DiSCo 364 protocol, noiseless, 659,840 seeds (`tests/test_acceptance.py`, deterministic XLA ops):

| stage | L40S first call (s) | L40S steady (s) | L4 first call (s) | L4 steady (s) |
|---|---|---|---|---|
| layout on the device (once per process; L40S from a local copy, L4 from the Hub, 13 GB) | 8.1 | | 84.0 | |
| replay, 364 measurements on 3 timing classes | 2.5 | 2.4 | 4.6 | 5.0 |
| CSD (response + `csd_tournier07_jax`, order 8) | 4.6 | 2.9 | 5.4 | 3.6 |
| tracking (probabilistic, density 4) | 18.7 | 2.5 | 24.7 | 7.1 |
| whole pipeline | 26.0 | 7.9 | 34.9 | 15.9 |

Score on both: Pearson 0.927 vs strand count, 0.929 vs cross-sectional area; 120 of 120 pairs connected (25 in the
ground truth: 95 false, 0 missed); the two machines agree to the last digit (deterministic ops). The replay of the
DiSCo 364 protocol from the layout is within 2.0e-7 of the published reference volume on every voxel (K = 64 bands,
band error 1.4e-4 against the pack's floor 4.8e-3). The L4 run is the gate job in the Space's own image
(`tools/gate_job.py`, 2026-09-29).

**Live Space** (`rfick/disco`, A10G small after two "not enough hardware capacity" failures on the L4 on 2026-09-29,
through its API with `tools/live_check.py`): first request after a cold start 150 s wall (layout on the device 104 s
once per process, then 49.6 s for DiSCo 364 at SNR 30 including every compile: replay 9.4, noise 1.1, CSD 6.5,
tracking 32.4, score 0.3; Pearson 0.924 / 0.926). A custom two-shell protocol (77 measurements) right after: 34 s
wall, 11.8 s in the pipeline (replay 1.4, CSD 4.3, tracking 5.2), Pearson 0.904. The `.tck` (290 MB), the DWI with
its table and the FOD field download from the page.
