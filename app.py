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

DURATION = int(os.environ.get("DISCO_GPU_SECONDS", "480"))


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


@spaces.GPU(duration=60)
def probe():
    """What the GPU worker sees: the device, the memory, the mounted layout and how fast it reads (an API endpoint
    for the deployment check, `/probe`)."""
    import glob, json, time
    import numpy as np
    out = dict(torch=torch.__version__, cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
               data=sorted(os.path.basename(p) for p in glob.glob("/data/moments/*"))[:12], backend=os.environ.get("DISCO_BACKEND"),
               resident=os.environ.get("DISCO_RESIDENT"))
    try:
        with open("/proc/meminfo") as f:
            mem = dict(l.split(":") for l in f.read().splitlines() if ":" in l)
        out["host_ram_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1e6, 1); out["host_ram_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1e6, 1)
    except Exception as e:
        out["meminfo"] = repr(e)[:100]
    files = sorted(glob.glob("/data/moments/m_*.npy"))
    if files:
        t0 = time.perf_counter(); m = np.load(files[0], mmap_mode="r"); chunk = np.array(m[:100000]); dt = time.perf_counter() - t0
        out["read_mb"] = round(chunk.nbytes / 1e6, 1); out["read_mb_per_s"] = round(chunk.nbytes / 1e6 / dt, 1)
        t0 = time.perf_counter(); dev = torch.as_tensor(chunk, device="cuda"); torch.cuda.synchronize(); out["upload_mb_per_s"] = round(chunk.nbytes / 1e6 / (time.perf_counter() - t0), 1)
    # the moment image inside this worker: the first compiled call, a second one (the kernel alone), the eager kernel
    try:
        from space import pipeline as P
        from dmipy_sim.replay import shape_moments as SM
        st = A._load(); lay = st["layout"]
        b = np.full(184, 1e9); u = np.tile([0.6, 0.0, 0.8], (184, 1)); name = "d10.2-D16.7"
        cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR"); out["inductor_cache_dir"] = cache
        out["inductor_cache_files_before"] = len(glob.glob(os.path.join(cache, "**", "*"), recursive=True)) if cache and os.path.isdir(cache) else None
        t0 = time.perf_counter(); lay.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_first_compiled_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter(); lay.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_second_compiled_s"] = round(time.perf_counter() - t0, 2)
        SM._TORCH_KERNEL["fn"] = SM._tile_sums_torch
        t0 = time.perf_counter(); lay.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_eager_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter(); lay.moments.image(name, b, u, backend="torch", resident=False); torch.cuda.synchronize(); out["image_eager_nonresident_s"] = round(time.perf_counter() - t0, 2)
        out["inductor_cache_files_after"] = len(glob.glob(os.path.join(cache, "**", "*"), recursive=True)) if cache and os.path.isdir(cache) else None
    except Exception as e:
        out["image_probe_error"] = repr(e)[:300]
    return json.dumps(out)


if __name__ == "__main__":
    import gradio as gr
    A._load()                                            # the layout from the mounted bucket, on the CPU side
    demo = A.build(runner=gpu_runner)
    with demo:
        probe_out = gr.Textbox(visible=False)
        gr.Button("probe", visible=False).click(probe, outputs=probe_out, api_name="probe")
    demo.queue(max_size=16).launch()
