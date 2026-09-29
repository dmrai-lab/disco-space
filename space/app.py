"""The DiSCo Space's page: three tabs over :mod:`space.pipeline` -- the acquisition (a preset, DiSCo's own table, custom
shells on the layout's timing classes, or an uploaded bvals/bvecs table; SNR; the tracker's settings; the run button,
whose progress bar names the stage it is in), the ground truth (the strands in a rotatable 3-D view, the two matrices),
and the results (a DWI slice viewer with the FODs' principal directions over it, the tractogram in 3-D, the connectome
beside the ground truth, the timings, the downloads). Nothing scientific lives here: every number comes from the
pipeline, every figure from :mod:`space.viewers`."""
from __future__ import annotations

import os
import tempfile
import threading
import time

import numpy as np

from . import pipeline as P
from . import viewers as V

MAX_SHELLS = 4
CUSTOM = "custom shells"
UPLOADED = "uploaded table"
STAGE_TEXT = {"replay": "replaying the grid from the stored walk", "noise": "adding Rician noise", "csd": "fitting CSD (order 8)",
              "track": "tracking from the sixteen regions", "score": "scoring the connectome"}
_state = {"layout": None, "error": None}
_lock = threading.Lock()


def _load():
    """The layout and the ground truth, loaded once per process (the first call downloads the moments)."""
    with _lock:
        if _state["layout"] is None and _state["error"] is None:
            try:
                t0 = time.perf_counter()
                cfg = P.config()
                layout = P.Layout(cfg, local=os.environ.get("DISCO_MOMENTS"))
                layout.warm()
                from dmipy_sim.io.strands import read_tck, read_diameters
                _state["strands"] = read_tck(os.path.join(P.DATA_DIR, "DiSCo_Strands_Trajectories.tck"), coordinate_unit_m=1.0)
                _state["diameters"] = read_diameters(os.path.join(P.DATA_DIR, "DiSCo_Strands_Diameters.txt"), diameter_unit_m=1.0)
                _state["layout"] = layout; _state["cfg"] = cfg; _state["load_seconds"] = time.perf_counter() - t0
            except Exception as e:                       # shown on the page instead of a dead Space
                _state["error"] = repr(e)
        return _state


def _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs, table=None):
    """The acquisition the page asks for: DiSCo's own table, a config preset, the shell rows, or an uploaded table
    (``table`` = (bvals path, bvecs path, timing class))."""
    if preset == "DiSCo 364":
        return P.disco_protocol(cfg)[0]
    if preset == UPLOADED:
        if table is None or not table[0] or not table[1]:
            raise ValueError("upload a bvals and a bvecs file for the uploaded table")
        return P.protocol_from_table(table[0], table[1], table[2], cfg["shapes"])
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


def _result_state(res, layout, state):
    """What the results tab needs, kept per session: float32 volumes, the peaks, a streamline sample, the numbers."""
    peaks, amp = P.peaks(res.sh)
    return dict(dwi=res.dwi.astype(np.float32), meas=res.meas, peaks=peaks.astype(np.float32), peak_amp=amp.astype(np.float32),
                matrix=res.matrix, score=res.score, seconds=res.seconds, load_seconds=state.get("load_seconds", float("nan")),
                tractogram=res.tractogram, rois=layout.rois, shape=layout.mask.shape, name=res.protocol.name)


def run_pipeline(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, bvals_file, bvecs_file, table_shape, *shell_inputs,
                 progress=None):
    """The run button: ``progress(fraction, desc=)`` is Gradio's progress bar (a no-op when None)."""
    import gradio as gr
    progress = progress or (lambda *a, **k: None)
    progress(0.0, desc="loading the layout" if _state["layout"] is None else "starting")
    state = _load()
    if state["error"]:
        raise gr.Error(f"the layout did not load: {state['error']}")
    layout, cfg = state["layout"], state["cfg"]
    try:
        protocol = _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs, table=(bvals_file, bvecs_file, table_shape))
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    tracking = P.Tracking(density=int(density), step_mm=float(step_mm), max_angle=float(max_angle), key=int(key))

    def at(stage, k, n):
        progress((k + 0.5) / (n + 1), desc=f"{k + 1}/{n} {STAGE_TEXT[stage]}")
    res = P.run(layout, protocol, snr=(float(snr) if snr_on else None), tracking=tracking, progress=at)
    progress(0.92, desc="writing the downloads")
    out = tempfile.mkdtemp(); stem = f"disco_{protocol.name.replace(' ', '_')}"
    tck = os.path.join(out, f"{stem}.tck"); res.tractogram.to_tck(tck)
    vols = P.write_volumes(res, out, prefix=stem)
    progress(0.97, desc="drawing")
    rs = _result_state(res, layout, state)
    s = res.score
    headline = (f"**Pearson vs strand count {s['pearson_count']:.3f}, vs cross-sectional area {s['pearson_area']:.3f}** "
                f"({protocol.n_meas} measurements, {len(res.tractogram):,} streamlines, {res.seconds['total']:.1f} s in total; "
                f"replay floor median {np.nanmedian(res.floor[layout.mask]):.4f}). Results are in the third tab.")
    z0 = rs["dwi"].shape[2] // 2
    m0 = int(np.flatnonzero(~rs["meas"].b0)[0]) if (~rs["meas"].b0).any() else 0
    return (rs, headline, V.dwi_slice(rs["dwi"], rs["meas"], z0, m0, peaks=rs["peaks"], peak_amp=rs["peak_amp"], overlay=True),
            V.tractogram3d(rs["tractogram"], rs["rois"], rs["shape"]), V.matrices(rs["matrix"], rs["score"], layout.gt_count),
            V.timings_rows(rs["seconds"], rs["load_seconds"]), tck, [vols["dwi"], vols["bvals"], vols["bvecs"], vols["fod"]],
            _slider_update(z0, rs["dwi"].shape[2] - 1), _slider_update(m0, protocol.n_meas - 1))


def _slider_update(value, maximum):
    import gradio as gr
    return gr.update(value=int(value), maximum=int(maximum))


def redraw_slice(rs, z, m, overlay):
    if not rs:
        return None
    return V.dwi_slice(rs["dwi"], rs["meas"], int(z), int(m), peaks=rs["peaks"], peak_amp=rs["peak_amp"], overlay=bool(overlay))


def ground_truth_views():
    state = _load()
    if state["error"]:
        return None, None
    layout = state["layout"]
    return (V.strands3d(state["strands"], state["diameters"], layout.rois, layout.mask.shape),
            V.ground_truth_matrix(layout.gt_count, layout.gt_area))


def build():
    import gradio as gr
    cfg = P.config()
    shapes = cfg["shapes"]; presets = ["DiSCo 364"] + list(cfg["presets"]) + [CUSTOM, UPLOADED]
    shape_names = list(shapes); labels = {n: shapes[n]["label"] for n in shape_names}
    with gr.Blocks(title="DiSCo replay to tractogram") as demo:
        gr.Markdown(
            "# DiSCo: one Monte-Carlo walk, any acquisition, a connectome\n"
            "The DiSCo phantom's walkers were simulated once (SubstrateCommons/disco-replay). Choose an acquisition; the Space "
            "replays the whole 40³ grid from the stored walk, adds noise, fits constrained spherical deconvolution, tracks from "
            "the sixteen regions and scores the connectome against the ground truth. The progress bar names each stage.")
        result = gr.State(None)
        with gr.Tabs():
            with gr.Tab("1 · acquisition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        preset = gr.Dropdown(presets, value="DiSCo 364", label="acquisition")
                        n_b0 = gr.Slider(1, 10, value=1, step=1, label="b = 0 measurements (custom shells)")
                        shell_inputs = []
                        for k in range(MAX_SHELLS):
                            with gr.Row():
                                on = gr.Checkbox(value=(k == 0), label=f"shell {k + 1}")
                                shape = gr.Dropdown(shape_names, value=shape_names[0], label="timing class")
                                b = gr.Number(value=[1000, 2000, 3000, 6000][k], label="b (s/mm²)")
                                n = gr.Slider(6, 128, value=[30, 60, 90, 60][k], step=1, label="directions")
                            shell_inputs += [on, shape, b, n]
                        gr.Markdown("Timing classes: " + "; ".join(f"`{n}` = {labels[n]}" for n in shape_names))
                        with gr.Row():
                            bvals_file = gr.File(label="uploaded table: bvals (s/mm²)", file_count="single", type="filepath")
                            bvecs_file = gr.File(label="uploaded table: bvecs", file_count="single", type="filepath")
                            table_shape = gr.Dropdown(shape_names, value=shape_names[0], label="its timing class (every row)")
                    with gr.Column(scale=1):
                        with gr.Row():
                            snr_on = gr.Checkbox(value=True, label="add Rician noise")
                            snr = gr.Slider(5, 100, value=30, step=1, label="SNR at b = 0")
                        density = gr.Slider(1, 4, value=cfg["tracking"]["density"], step=1, label="seeds per region voxel (density³)")
                        max_angle = gr.Slider(10, 60, value=cfg["tracking"]["max_angle"], step=1, label="max angle (°)")
                        step_mm = gr.Slider(0.25, 1.0, value=cfg["tracking"]["step_mm"], step=0.05, label="step (voxels)")
                        key = gr.Number(value=0, precision=0, label="random key")
                        go = gr.Button("replay → CSD → track → score", variant="primary")
                        headline = gr.Markdown()
            with gr.Tab("2 · ground truth") as gt_tab:
                strands_view = gr.Plot(label="the strands")
                gt_matrix = gr.Image(label="the ground-truth matrices", type="pil")
            with gr.Tab("3 · results"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dwi_view = gr.Image(label="DWI slice", type="pil")
                        with gr.Row():
                            z_slider = gr.Slider(0, 39, value=20, step=1, label="axial slice z")
                            m_slider = gr.Slider(0, 363, value=0, step=1, label="measurement")
                            overlay = gr.Checkbox(value=True, label="FOD principal directions")
                    with gr.Column(scale=1):
                        tract_view = gr.Plot(label="tractogram")
                mats = gr.Image(label="connectome vs ground truth", type="pil")
                with gr.Row():
                    timings = gr.Dataframe(headers=["stage", "seconds"], label="timings", interactive=False)
                    with gr.Column():
                        tck = gr.File(label="tractogram (.tck, MRtrix)")
                        volumes = gr.File(label="DWI (.nii.gz) with bvals/bvecs, and the FOD SH field (.nii.gz, tournier07 order 8)", file_count="multiple")
        def run_with_progress(*args, progress=gr.Progress()):
            return run_pipeline(*args, progress=progress)
        go.click(run_with_progress, inputs=[preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, bvals_file, bvecs_file, table_shape, *shell_inputs],
                 outputs=[result, headline, dwi_view, tract_view, mats, timings, tck, volumes, z_slider, m_slider], concurrency_limit=1)
        for ctl in (z_slider, m_slider, overlay):
            ctl.change(redraw_slice, inputs=[result, z_slider, m_slider, overlay], outputs=dwi_view, show_progress="hidden")
        gt_tab.select(ground_truth_views, outputs=[strands_view, gt_matrix])
        demo.load(lambda: (_load().get("error") and f"**the layout did not load:** {_load()['error']}") or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
