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

Every run reserves the GPU for the seconds the page shows under the run button (measured on the pool: 30 s fixed plus
0.19 s per measurement per run, times 1.2). ZeroGPU charges that reservation against the *visitor's* daily quota, not
the Space's: 2 minutes logged out, 5 with a free Hugging Face account, 40 with PRO. A single request above the visitor's
quota is refused before it starts with "The requested GPU duration (N s) is larger than the maximum allowed", so a
DiSCo 364 run alone fits a logged-out visitor, DiSCo with the explorer's ladder or a B run needs a free account, and A + B
+ ladder (about 4 min) needs PRO. Log in to Hugging Face in the same browser to use your own quota.
