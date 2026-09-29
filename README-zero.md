---
title: DiSCo replay to tractogram (ZeroGPU)
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: DiSCo replay to connectome on the shared GPU pool
---

# The DiSCo Space on ZeroGPU

The same application as `rfick/disco` (dmrai-lab/disco-space) on Hugging Face's shared GPU pool: the compute backend
is PyTorch (`DISCO_BACKEND=torch`: dmipy-sim's shape-moment image, dmipy-fit's `csd_tournier07_torch`, dmipy-tract's
torch tracker), the GPU exists for the duration of one run (`spaces.GPU`), and the layout's tiles are rebuilt on the
device inside every call from the bucket mounted at `/data`. Deterministic algorithms are on and TF32 off inside the
call. Source and pins: https://github.com/dmrai-lab/disco-space (`requirements-zero.txt`, `app.py`).
