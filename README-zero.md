---
title: DiSCo replay to tractogram (ZeroGPU)
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: DiSCo replay to connectome on the shared GPU pool
---

# The DiSCo Space on ZeroGPU

The same application as `rfick/disco` (dmrai-lab/disco-space) on Hugging Face's shared GPU pool: the compute backend
is PyTorch (`DISCO_BACKEND=torch`: dmipy-sim's shape-moment image, dmipy-fit's `csd_tournier07_torch`, dmipy-tract's
torch tracker), the GPU exists for the duration of one run (`spaces.GPU`), and the layout's tiles are rebuilt on the
device inside every call from the layout downloaded from the Hub when the container started. Deterministic algorithms are on and TF32 off inside the
call. Source and pins: https://github.com/dmrai-lab/disco-space (`requirements-zero.txt`, `app.py`).

## GPU quota per visitor

Every run reserves the GPU for the seconds the page shows under the run button. ZeroGPU charges that reservation
against the *visitor's* daily quota, not the Space's: 2 minutes logged out, 5 with a free Hugging Face account, 40
with PRO; a single request above the visitor's quota is refused before it starts ("The requested GPU duration (N s)
is larger than the maximum allowed"), and a run that outlives its reservation is killed. Only the device part of a
run (`space.app.compute`: the replays, the noise, the CSD, the tracking, the scoring, the explorer's maps) holds the
GPU; the page writes the files and draws in its own process after the device is released (disco-space#7).

Measured on the pool (DiSCo 364, every tier, density 4, 2026-09-30, commit 67de79d):

| run | device held | handoff | client wall | reserved |
|---|---|---|---|---|
| DiSCo 364 | 31 s | 2 s | 93 s | 51 s |
| DiSCo 364 + the explorer's ladder | 63 s | 4 s | 115 s | 91 s |

Before the split the device was held 77 s and 100 s for the same runs (the files, the states, the figures and a
15 s host pass for the ingredient maps were inside the window). The reservation is 16 s + 0.065 s per measurement
+ 0.084 s per measurement for the ladder + (8 s + 0.028 s per measurement) for B + 10 s per extra tracker key,
times 1.3. DiSCo 364 alone, with the ladder, and A + B + ladder all fit a logged-out visitor's 2 minutes.
