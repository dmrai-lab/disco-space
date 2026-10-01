"""The ZeroGPU entry (disco-space#2): the same page as ``space/app.py`` on Hugging Face's shared GPU pool, which runs
PyTorch only and gives a GPU to a call decorated with ``spaces.GPU`` for its duration: here the compute alone (with
the source's share a run needs that the page has not cached, a brain's pack responses, so the call is entered as soon
as the request arrives); the cache lookup before it and the page's drawing and files after it stay in this process
(disco-space#7). The configuration is ``DISCO_CONFIG`` (``space/config.toml``, the DiSCo Space, unless the
Space sets another: ``brain.toml`` for the brain Space). The compute backend is torch (``DISCO_BACKEND=torch``), the
layout's tiles are rebuilt on the device inside every call (``resident = false``: nothing survives between calls),
deterministic algorithms are on and TF32 off inside the call. The source's data come from the Hub at the config's
revision, downloaded once when the container starts (the parent process loads it before the page serves; the forked
GPU worker inherits it). ``spaces`` must be imported before torch."""
import os

os.environ.setdefault("DISCO_BACKEND", "torch")
os.environ.setdefault("DISCO_RESIDENT", "0")             # the pool drops the device between calls: tiles rebuilt per call
os.environ.setdefault("JAX_PLATFORMS", "cpu")            # jax is a library dependency here; it never sees the GPU

import spaces                                           # noqa: E402  (before torch, as ZeroGPU requires)
import torch                                            # noqa: E402

from space import app as A                              # noqa: E402

def gpu_runner(fn):
    """``fn`` (:func:`space.app.compute`) under the GPU for the seconds :func:`space.app.estimated_seconds` reads off
    the inputs (``DISCO_GPU_SECONDS`` overrides with a fixed number), deterministic, full precision. Its first
    argument is the source's share of the run the page has cached (:func:`space.app.prepare_runs`, a lookup), the
    page's inputs follow. A generator: its stage texts and its payload cross from the worker to the page's process,
    where the page draws and writes files after the device is released."""
    fixed = os.environ.get("DISCO_GPU_SECONDS")
    duration = (lambda *args, **kw: int(fixed)) if fixed else (lambda prepared, *args, **kw: A.estimated_seconds(*args))

    @spaces.GPU(duration=duration)
    def run(*args):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield from fn(*args)
    return run


@spaces.GPU(duration=60)
def probe():
    """What the GPU worker sees: the device, the memory, the layout's files and how fast they read (an API endpoint
    for the deployment check, `/probe`)."""
    import glob, json, time
    import numpy as np
    st = A._load(); src = st["source"]; where = getattr(getattr(src, "moments", None), "path", "")
    out = dict(torch=torch.__version__, cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
               layout=where, data=sorted(os.path.basename(p) for p in glob.glob(os.path.join(where, "*")))[:12], backend=os.environ.get("DISCO_BACKEND"),
               resident=os.environ.get("DISCO_RESIDENT"), load_seconds=round(st.get("load_seconds", float("nan")), 1), error=st.get("error"),
               cpus=len(os.sched_getaffinity(0)), cpu_count=os.cpu_count())
    try:
        with open("/proc/meminfo") as f:
            mem = dict(l.split(":") for l in f.read().splitlines() if ":" in l)
        out["host_ram_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1e6, 1); out["host_ram_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1e6, 1)
    except Exception as e:
        out["meminfo"] = repr(e)[:100]
    files = sorted(glob.glob(os.path.join(where, "m_*.npy")))
    if files:
        t0 = time.perf_counter(); m = np.load(files[0], mmap_mode="r"); chunk = np.array(m[:100000]); dt = time.perf_counter() - t0
        out["read_mb"] = round(chunk.nbytes / 1e6, 1); out["read_mb_per_s"] = round(chunk.nbytes / 1e6 / dt, 1)
        t0 = time.perf_counter(); torch.as_tensor(chunk, device="cuda"); torch.cuda.synchronize(); out["upload_mb_per_s"] = round(chunk.nbytes / 1e6 / (time.perf_counter() - t0), 1)
    # the moment image inside this worker: the first compiled call, a second one (the kernel alone), the eager kernel
    try:
        from dmipy_sim.replay import shape_moments as SM
        kernel = dict(SM._TORCH_KERNEL)
        b = np.full(184, 1e9); u = np.tile([0.6, 0.0, 0.8], (184, 1)); name = "d10.2-D16.7"
        try:
            t0 = time.perf_counter(); src.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_first_compiled_s"] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter(); src.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_second_compiled_s"] = round(time.perf_counter() - t0, 2)
            SM._TORCH_KERNEL["fn"] = SM._tile_sums_torch
            t0 = time.perf_counter(); src.moments.image(name, b, u, backend="torch", resident=True); torch.cuda.synchronize(); out["image_eager_s"] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter(); src.moments.image(name, b, u, backend="torch", resident=False); torch.cuda.synchronize(); out["image_eager_nonresident_s"] = round(time.perf_counter() - t0, 2)
        finally:
            SM._TORCH_KERNEL.update(kernel)                  # the probe leaves the kernel as it found it
    except Exception as e:
        out["image_probe_error"] = repr(e)[:300]
    return json.dumps(out)


if __name__ == "__main__":
    import gradio as gr
    A._load()                                            # the layout from the Hub, once per container, on the CPU side
    demo = A.build(runner=gpu_runner)
    with demo:
        probe_out = gr.Textbox(visible=False)
        gr.Button("probe", visible=False).click(probe, outputs=probe_out, api_name="probe")
    demo.queue(max_size=16).launch()
