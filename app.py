"""The ZeroGPU entry (disco-space#2): the same page as ``space/app.py`` on Hugging Face's shared GPU pool, which runs
PyTorch only and gives a GPU to a call decorated with ``spaces.GPU`` for its duration. The compute backend is torch
(``DISCO_BACKEND=torch``), the layout's tiles are rebuilt on the device inside every call (``resident = false``:
nothing survives between calls), deterministic algorithms are on and TF32 off inside the call. ``spaces`` must be
imported before torch."""
import os

os.environ.setdefault("DISCO_BACKEND", "torch")
os.environ.setdefault("DISCO_RESIDENT", "0")             # the pool drops the device between calls: tiles rebuilt per call
os.environ.setdefault("JAX_PLATFORMS", "cpu")            # jax is a library dependency here; it never sees the GPU

import spaces                                           # noqa: E402  (before torch, as ZeroGPU requires)
import torch                                            # noqa: E402

from space import app as A                              # noqa: E402

DURATION = int(os.environ.get("DISCO_GPU_SECONDS", "240"))


def gpu_runner(fn):
    """``fn`` under the GPU for ``DURATION`` seconds, deterministic, full precision. The progress bar is declared
    on the decorated function itself (``progress=gr.Progress()``), which is how ZeroGPU forwards it into the GPU
    worker; a progress object passed in from outside cannot cross the process boundary."""
    import gradio as gr

    @spaces.GPU(duration=DURATION)
    def run(*args, progress=gr.Progress()):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return fn(*args, progress=progress)
    return run


if __name__ == "__main__":
    A._load()                                            # the layout from the mounted bucket, on the CPU side
    A.build(runner=gpu_runner).queue(max_size=16).launch()
