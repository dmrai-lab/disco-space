---
title: "Brain replay: MASiVar"
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: A real brain replayed from packs, to an 84-region connectome
---

# The brain Space on ZeroGPU

The same application as `rfick/disco-zero` (dmrai-lab/disco-space), serving `space/brain.toml` (`DISCO_CONFIG`):
a real brain as a compositional replay phantom. The input is the multi-tissue CSD of one scan, MASiVar `sub-cIs1`
(OpenNeuro ds003416, Cai et al. MRM 2021, CC0): its white-matter FOD field, its WM / GM / CSF fractions and an
84-region parcellation, made once offline and published as the dataset `SubstrateCommons/masivar-brain`. Each run
composes every brain voxel from two replay packs (CACTUS axons for WM, packed spheres for GM; free water in closed
form for CSF) at the chosen acquisition, tissue and scanner, adds Rician noise, estimates the tissue responses from
the replayed data, fits CSD, tracks from the white matter and scores the 84 x 84 connectome (and its 14 lobar groups)
against the connectome of the input FOD tracked with the same tracker, settings, seeds and key. Source and pins:
https://github.com/dmrai-lab/disco-space (`requirements-zero.txt`, `app.py`, `space/sources/brain.py`).

**Where the work runs.** The packs' pose responses (the only physics that depends on the tissue and the field) are
computed on the CPU of the page's process before the GPU is held, and kept for the container's lifetime per pack,
protocol, tissue and field; the GPU call gets the small response arrays. Inside the call, torch contracts them with
every voxel's FOD, fractions and proton densities (voxels x measurements x 45), then the noise, the reconstruction
(dmipy-fit), the tracking and the truth's tracking (dmipy-tract), the scoring. Deterministic algorithms are on and
TF32 off inside the call.

## GPU quota per visitor

Every run reserves the GPU for the seconds the page shows under the run button. ZeroGPU charges that reservation
against the *visitor's* daily quota, not the Space's: 2 minutes logged out, 5 with a free Hugging Face account, 40
with PRO; a single request above the visitor's quota is refused before it starts, and a run that outlives its
reservation is killed. Only the device part of a run holds the GPU; the packs' responses before it and the page's
files and figures after it do not.

Measured on the L40S with the BATMAN development fixture (96 x 96 x 60 at 2.5 mm, 90,205 brain voxels, 36,605 in the
WM stop mask; dmipy-sim 7f6f1fa, dmipy-fit 0c7dde8 with the single-tissue reconstruction, dmipy-tract 7da22c3;
`tools/measure_brain.py`, 2026-09-30), steady state, seconds:

| stage | 113 measurements | 495 measurements (five shells x 96) |
|---|---|---|
| the packs' responses, every tier, on the host CPU (not charged) | 4.9-6.1 (gaia, 8 threads) | 7.5-7.9 (gaia, 8 threads) |
| the replay (the contraction on the device) | 0.04 | 0.17 |
| noise | 0.35 | 1.4 |
| reconstruction (single-tissue: tournier07 response + CSD) | 3.7 | 8.0 |
| the round trip against the input FOD | 0.85 | 0.85 |
| tracking, density 1 / 2 / 4 (36,605 / 292,840 / 2,342,720 seeds) | 0.7 / 1.9 / 13.6 | 0.7 / 1.9 / 13.6-16.4 |
| the truth's tracking, density 1 / 2 / 4 | 0.8 / 2.1 / 14.8 | 0.7 / 2.1 / 23.2 |
| the ladder (three rungs of contraction) | 0.07 | 0.5 |

The reservation (`space/brain.toml`, `[budget]`) is each device stage x 1.5 for the pool, 12 s for the worker's
start and the handoff, times 1.3: at the default density 2 and 495 measurements, A alone 49 s, A with the ladder 50
s, A + B + ladder 79 s, so every default configuration with one B fits a logged-out visitor's 2 minutes; density 4
reserves 131 s for A alone and is for logged-in visitors. These are L40S numbers scaled, not the pool's: the pool's
own table replaces them after the first deployment (`tools/live.py rfick/brain-zero --config brain.toml`).

The payload that crosses from the GPU worker back to the page is one Result per run (the DWI before and after the
noise in float32 on the brain's bounding box, the FOD field, the streamlines): 1.5 GB per run at 495 measurements and
density 2 (0.27 GB of it streamlines; 3.4 GB at density 4), pickled in 1.8 s on the L40S host, plus the ladder's
rungs; its handoff on the pool is the number to watch on the first deployment.

The truth connectome is reproducible bit for bit at one key; between two keys at the same density its Pearson of
log(1 + count) over the 3,486 region pairs is 0.834 / 0.945 / 0.982 at density 1 / 2 / 4, the floor of the score.
