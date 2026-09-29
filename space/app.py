"""The DiSCo Space's page: a Gradio Blocks over :mod:`space.pipeline`. The user sets the acquisition (a preset or
their own shells on the layout's timing classes), the SNR and the tracker's settings; the page shows the DWI, the
tractogram, the connectome beside the ground truth with its Pearson scores, every stage's time, and offers the
tractogram as ``.tck``. Nothing scientific lives here: every number comes from the pipeline."""
from __future__ import annotations

import io
import os
import tempfile
import threading
import time

import numpy as np

from . import pipeline as P

MAX_SHELLS = 4
STREAMLINES_DRAWN = 4000
_state = {"layout": None, "error": None}
_lock = threading.Lock()


def _load():
    """The layout, loaded once per process (the first call downloads the moments into the Hub cache)."""
    with _lock:
        if _state["layout"] is None and _state["error"] is None:
            try:
                t0 = time.perf_counter()
                cfg = P.config()
                layout = P.Layout(cfg, local=os.environ.get("DISCO_MOMENTS"))    # a local copy of the layout, else the Hub
                layout.warm()
                _state["layout"] = layout
                _state["cfg"] = cfg
                _state["load_seconds"] = time.perf_counter() - t0
            except Exception as e:                       # shown on the page instead of a dead Space
                _state["error"] = repr(e)
        return _state


def _figure_to_image(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def dwi_figure(res):
    """The mid-slice of the b = 0 volume and of each shell's mean DWI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = res.dwi.shape[2] // 2
    panels = [("b = 0", np.nanmean(res.dwi[..., res.meas.b0], -1))]
    for s in res.protocol.shells:
        rows = (res.meas.shape == s.shape) & (np.abs(res.meas.bvals - s.b) < 1)
        panels.append((f"b = {s.b:g}, {s.n_dirs} dirs", np.nanmean(res.dwi[..., rows], -1)))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (title, vol) in zip(np.atleast_1d(axes), panels):
        ax.imshow(np.nan_to_num(vol[:, :, z]).T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"axial slice {z} of the replayed DWI (S0-normalised)", fontsize=10)
    return _figure_to_image(fig)


def tractogram_figure(res, n=STREAMLINES_DRAWN, seed=0):
    """A random sample of streamlines projected on the axial plane, coloured by local direction, over the regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    tg = res.tractogram
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tg), min(n, len(tg)), replace=False)
    segs, cols = [], []
    for i in pick:
        pts = tg[i]
        if len(pts) < 2:
            continue
        d = np.diff(pts, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        segs.append(np.stack([pts[:-1, :2], pts[1:, :2]], axis=1)); cols.append(np.abs(d))
    fig, ax = plt.subplots(figsize=(6, 6))
    background = np.nan_to_num(np.nanmean(res.dwi[..., res.meas.b0], -1)).max(axis=2)      # the b = 0 volume, projected along z
    ax.imshow(background.T, origin="lower", cmap="gray", alpha=0.35,
              extent=(-0.5, res.dwi.shape[0] - 0.5, -0.5, res.dwi.shape[1] - 0.5))
    if segs:
        lc = LineCollection(np.concatenate(segs), colors=np.concatenate(cols), linewidths=0.4, alpha=0.6)
        ax.add_collection(lc)
    ax.set_xlim(-0.5, res.dwi.shape[0] - 0.5); ax.set_ylim(-0.5, res.dwi.shape[1] - 0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{len(tg):,} streamlines from {len(res.seeds):,} seeds ({len(pick):,} drawn), colour = |direction| (x, y, z)", fontsize=9)
    return _figure_to_image(fig)


def matrix_figure(res, layout):
    """Ours beside the strand-count ground truth, both on a log colour scale, with the Pearson numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    s = res.score
    for ax, (title, M) in zip(axes, ((f"streamline counts (Pearson {s['pearson_count']:.3f} vs count, {s['pearson_area']:.3f} vs area)", res.matrix),
                                     ("ground truth: strand count", layout.gt_count))):
        im = ax.imshow(np.where(M > 0, M, np.nan), norm=LogNorm(vmin=1, vmax=max(M.max(), 2)), cmap="viridis")
        ax.set_title(title, fontsize=9); ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3))
        ax.set_xticklabels(range(1, 17, 3)); ax.set_yticklabels(range(1, 17, 3))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{s['connected_pairs']} of 120 pairs connected, ground truth {s['gt_pairs']}; {s['false_pairs']} false, {s['missed_pairs']} missed", fontsize=9)
    return _figure_to_image(fig)


def timings_table(res, state):
    rows = [["layout on device (once per process)", f"{state.get('load_seconds', float('nan')):.1f}"]]
    rows += [[k, f"{v:.2f}"] for k, v in res.seconds.items()]
    return rows


CUSTOM = "custom shells"


def _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs):
    """The acquisition the page asks for: DiSCo's own table, a config preset, or the shell rows."""
    if preset == "DiSCo 364":
        return P.disco_protocol(cfg)[0]
    if preset in cfg["presets"]:
        p = cfg["presets"][preset]
        return P.Protocol(tuple(P.Shell(s["shape"], float(s["b"]), int(s["n_dirs"])) for s in p["shells"]), n_b0=int(p["n_b0"]), name=preset)
    if preset != CUSTOM:
        raise ValueError(f"unknown acquisition {preset!r}")
    shells = []
    for k in range(MAX_SHELLS):
        on, shape, b, n = shell_inputs[4 * k: 4 * k + 4]
        if on:
            shells.append(P.Shell(str(shape), float(b), int(n)))
    return P.Protocol(tuple(shells), n_b0=int(n_b0), name=CUSTOM)


def run_pipeline(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, *shell_inputs):
    state = _load()
    if state["error"]:
        raise RuntimeError(f"the layout did not load: {state['error']}")
    layout, cfg = state["layout"], state["cfg"]
    protocol = _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs)
    tracking = P.Tracking(density=int(density), step_mm=float(step_mm), max_angle=float(max_angle), key=int(key))
    res = P.run(layout, protocol, snr=(float(snr) if snr_on else None), tracking=tracking)
    tck = os.path.join(tempfile.mkdtemp(), f"disco_{protocol.name.replace(' ', '_')}.tck")
    res.tractogram.to_tck(tck)
    s = res.score
    headline = (f"**Pearson vs strand count {s['pearson_count']:.3f}, vs cross-sectional area {s['pearson_area']:.3f}** "
                f"({protocol.n_meas} measurements, {len(res.tractogram):,} streamlines, "
                f"{res.seconds['total']:.1f} s in total; replay floor median {np.nanmedian(res.floor[layout.mask]):.4f})")
    return (headline, dwi_figure(res), tractogram_figure(res), matrix_figure(res, layout), timings_table(res, state), tck)


def build():
    import gradio as gr
    cfg = P.config()
    shapes = cfg["shapes"]; presets = ["DiSCo 364"] + list(cfg["presets"]) + [CUSTOM]
    shape_names = list(shapes)
    labels = {n: shapes[n]["label"] for n in shape_names}
    with gr.Blocks(title="DiSCo replay to tractogram") as demo:
        gr.Markdown(
            "# DiSCo: one Monte-Carlo walk, any acquisition, a connectome\n"
            "The DiSCo phantom's walkers were simulated once (SubstrateCommons/disco-replay). Choose an acquisition; the "
            "Space replays the whole 40³ grid from the stored walk, adds noise, fits constrained spherical deconvolution, "
            "tracks from the sixteen regions and scores the connectome against the ground truth. Every stage's time is shown: "
            "the wait is the demonstration.")
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(presets, value="DiSCo 364", label="acquisition")
                n_b0 = gr.Slider(1, 10, value=1, step=1, label="b = 0 measurements (custom shells)")
                shell_inputs = []
                for k in range(MAX_SHELLS):
                    with gr.Row():
                        on = gr.Checkbox(value=(k == 0), label=f"shell {k + 1}")
                        shape = gr.Dropdown(shape_names, value=shape_names[min(k, 0)], label="timing class",
                                            info=None)
                        b = gr.Number(value=[1000, 2000, 3000, 6000][k], label="b (s/mm²)")
                        n = gr.Slider(6, 128, value=[30, 60, 90, 60][k], step=1, label="directions")
                    shell_inputs += [on, shape, b, n]
                gr.Markdown("Timing classes: " + "; ".join(f"`{n}` = {labels[n]}" for n in shape_names))
                with gr.Row():
                    snr_on = gr.Checkbox(value=True, label="add Rician noise")
                    snr = gr.Slider(5, 100, value=30, step=1, label="SNR at b = 0")
                with gr.Row():
                    density = gr.Slider(1, 4, value=cfg["tracking"]["density"], step=1, label="seeds per region voxel (density³)")
                    max_angle = gr.Slider(10, 60, value=cfg["tracking"]["max_angle"], step=1, label="max angle (°)")
                    step_mm = gr.Slider(0.25, 1.0, value=cfg["tracking"]["step_mm"], step=0.05, label="step (voxels)")
                    key = gr.Number(value=0, precision=0, label="random key")
                go = gr.Button("replay → CSD → track → score", variant="primary")
            with gr.Column(scale=2):
                headline = gr.Markdown()
                dwi = gr.Image(label="DWI", type="pil")
                tract = gr.Image(label="tractogram", type="pil")
                mats = gr.Image(label="connectome vs ground truth", type="pil")
                timings = gr.Dataframe(headers=["stage", "seconds"], label="timings", interactive=False)
                tck = gr.File(label="tractogram (.tck, MRtrix)")
        go.click(run_pipeline, inputs=[preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, *shell_inputs],
                 outputs=[headline, dwi, tract, mats, timings, tck], concurrency_limit=1)
        demo.load(lambda: _load().get("error") and f"**the layout did not load:** {_load()['error']}" or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
