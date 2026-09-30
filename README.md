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
acquisition, an SNR and the tracker's settings; the Space replays the whole 40³ grid from the stored walk, adds
Rician noise, fits CSD (order 8, the single-fibre response from the volume), tracks probabilistically from the
sixteen regions and scores the 16 × 16 connectome against the dataset's strand-count and area matrices. Every
stage's time is on screen.

**The physics as knobs.** The walk carries more than positions: each walker's pool, its wall contacts and the
susceptibility field it saw are stored as tiers, so the same run can be evaluated bare or in tissue at a scanner.
The *tissue and scanner* panel sets the field (presets 0.064, 1.5, 3, 7, 11.7 T or any value), its direction in the
phantom's frame (along z, transverse, or free angles), T2 and T1 per pool, the surface relaxivity and myelin's
susceptibility, each tier switchable on its own; the defaults are dmipy-sim's cited white-matter catalogue at the
nearest field. A stimulated-echo pulse timing (δ 7.6 / TM 38.3 ms) is the twin of DiSCo's b = 3091 class: the same
diffusion time with the magnetisation stored along z, so the field acts only during the two δ.

**The Replay DWI Explorer** (the third tab) shows what the replay made before the noise and the tractography: the
ingredient of each tier per voxel (the intra-axonal weight fraction; the walkers' wall contact, a boundary local
time of 0–58 µm on DiSCo with a contact survival of 0.89–1 at ρ = 1.16 µm/s; the spread of the sheath field's
dephasing phase at the echo, 0.10 rad in strand voxels at 3 T and 0.24 rad at 7 T), then the same walk replayed
noise-free with the tiers switched on one at a time, A and B, and any layer's signal, its difference to the
previous layer or B minus A, in the DWI or in the tensor's MD and FA, divided by the replay floor when asked.

**Two modes, one image.** The hosted Spaces run in *demo* mode: the acquisition's pulse timing is one of a few
stored classes (the shape-moment layout, `disco/moments/` of the dataset, one pass over the pack per class), so a
run takes seconds and the b-values, directions, SNR, tissue and scanner stay free. The same image beside the
columnar pack runs in *full* mode: every run reads the pack, so δ, Δ, TE and a Camino `.scheme` upload are free,
and a run takes minutes (the plan on screen says how many). The mode is the container's configuration:

```bash
docker run -p 7860:7860 ghcr.io/dmrai-lab/disco-space                                   # demo: the stored classes
docker run -p 7860:7860 -e DISCO_MODE=full -e DISCO_COLUMNS=hf://SubstrateCommons/disco-replay/disco \
    ghcr.io/dmrai-lab/disco-space                                                        # full: the pack from the Hub
docker run -p 7860:7860 -e DISCO_MODE=full -e DISCO_COLUMNS=/columns -v /path/to/disco:/columns:ro \
    ghcr.io/dmrai-lab/disco-space                                                        # full: the pack on disk
```

**The data path.** `disco/moments/` on the dataset is dmipy-sim's *shape-moment layout*: the replay pack contracted
once against each PGSE timing class (δ/Δ at TE 53.5 ms, square pulses), so that any b-value and direction on a
class is a three-term phase per walker. Seven classes are stored (`space/config.toml`, `[shapes]`: six PGSE and one stimulated echo); the layout is
written by `dmipy_sim.replay.shape_moments.write_shape_moments` from the certified columnar rows and names the
columnar manifest's sha256 it was contracted from. The replay of DiSCo's protocol from the layout is checked
against the published reference volume by the gate.

## Layout

```
Dockerfile, entrypoint.sh   python:3.11-slim, the pinned stack, ldconfig for the CUDA wheels, caches on /data (self-contained)
Dockerfile.base / .space    the same split in two: the stack as ghcr.io/dmrai-lab/disco-space (image.yml), the app on top
requirements.txt            the pins (dmipy packages by commit)
space/config.toml           the DiSCo Space: dataset + layout revision, the timing classes, DiSCo's shells, presets, tracking, the gate
space/brain.toml            the brain Space (DISCO_CONFIG=brain.toml): the asset, the pack menu, the timing classes, presets, the GPU budget
space/pipeline.py           replay -> noise -> reconstruction -> track -> score, plain functions with timings; the Source interface
space/sources/disco.py      the DiSCo source: its layout (demo) and columnar pack (full), texts, panel, regions, ground truth, budget
space/sources/brain.py      the brain source: an asset (FOD, fractions, parcellation) composed with WM / GM packs on the device
space/app.py                Gradio Blocks over the configured source: four tabs (acquisition, tissue and scanner; the truth; the
                            Replay DWI Explorer; results: DWI slice viewer with FOD peaks, 3-D tractogram, connectome, timings,
                            downloads); the progress bar names the stage
space/viewers.py            the figures (matplotlib slices and matrices, Plotly 3-D views); nothing derived here
data/                       DiSCo mask, regions, gradient table, ground-truth matrices and strands (CC BY 4.0, see SOURCE.md)
tests/test_pipeline.py      CPU: protocol construction, DiSCo's table, the score on the ground truth itself
tests/test_acceptance.py    GPU + data: the gate (Pearson, missed pairs, the reference volume, timings)
tests/test_brain.py         the brain source on its local fixture: the replay against dmipy-sim's Phantom, the page's chain
tools/build_brain_fixture.py  the BATMAN subject in the brain asset's layout (local development fixture, never published)
tools/measure_brain.py      the brain's costs per stage (the [budget] of brain.toml is refit from it)
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

**Storage.** Both Spaces download the layout from the dataset at the config's pinned revision when their container
starts (the Docker Space into `/data/hf` on its own disk, the ZeroGPU parent process into the Hub cache before the
page serves), so a wake costs one download per container start, not per visitor; nothing is mounted. The Storage
Bucket of the first deployment (`rfick/disco-space-data`) is retired by `tools/deploy.py`, which removes the
`DISCO_MOMENTS` variable and any volume it finds. Measured 2026-09-29 (A10G, with the bucket): a wake without a
build served in 16 s and the first DiSCo 364 request after it answered in 86 s wall; the Hub-download figures are
measured on the next deploy.

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
run, the layout is downloaded from the Hub into the parent process once per container and the forked GPU worker inherits
it (the host-to-device transfer runs at 8 GB/s), and nothing in the worker touches JAX (the noise
draw comes from numpy there; a JAX call in the forked worker aborts the task). Measured 2026-09-30 through the API
on the tiered layout (3 T, every tier on, SNR 30): the container built and started in 364 s (the 31.6 GB layout
downloaded from the Hub and preloaded in 55 s of that); DiSCo 364 in 84 s wall, 30 s in the pipeline (replay 22.8 s
with the per-call transfer of the moment and tier columns, noise 0.5, CSD 3.3, tracking 3.3), Pearson 0.924; a
custom two-shell protocol (77 measurements) in 54 s wall, 9.6 s in the pipeline; DiSCo 364 with B (the transverse
field), the tier ladder and the ingredient maps in 240 s wall. No idle cost, no sleep; per-visitor quotas instead of
a queue on one card.

**The SNR is defined at M0**, the bare signal of the fullest water voxel, and each voxel's noise follows its own
b = 0 signal: on DiSCo 364 at 3 T with every tier the b = 0 signal is 0.32–0.38 of M0 (T2 near TE, the walls'
contact in the densest strand voxels), so SNR 30 at M0 is SNR 11 at b = 0 and the connectome's Pearson vs strand
count goes from 0.924 (bare, or SNR 80 at M0) to 0.912; noiseless it is 0.927 (measured on the L40S, 2026-09-30). Deploy with `tools/deploy.py --zero`
(`README-zero.md`, `requirements-zero.txt`, the root `app.py`).

**The brain Space** (`rfick/brain-zero`, disco-space#8) is this repository deployed with
`tools/deploy.py --zero --space rfick/brain-zero --config brain.toml`: the deploy sets `DISCO_CONFIG=brain.toml` on the
Space and takes `README-brain-zero.md` as its README (its measured table). The page, the pipeline and the quota
machinery are the same; the source (`space/sources/brain.py`) is a real brain composed from an asset and two packs.

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
through its API with `tools/live.py`): first request after a cold start 150 s wall (layout on the device 104 s
once per process, then 49.6 s for DiSCo 364 at SNR 30 including every compile: replay 9.4, noise 1.1, CSD 6.5,
tracking 32.4, score 0.3; Pearson 0.924 / 0.926). A custom two-shell protocol (77 measurements) right after: 34 s
wall, 11.8 s in the pipeline (replay 1.4, CSD 4.3, tracking 5.2), Pearson 0.904. The `.tck` (290 MB), the DWI with
its table and the FOD field download from the page.
