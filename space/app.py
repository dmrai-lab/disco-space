"""The DiSCo Space's page: three tabs over :mod:`space.pipeline` -- the acquisition (a preset, DiSCo's own table, custom
shells; SNR; the tracker's settings; the run button, whose progress bar names the stage it is in), the ground truth
(the strands in a rotatable 3-D view, the two matrices), and the results (a DWI slice viewer with the FODs' principal
directions over it, the tractogram in 3-D, the connectome beside the ground truth, the timings, the downloads). In
demo mode (the hosted Spaces) a shell plays one of the layout's stored pulse timings; in full mode (``DISCO_MODE=full``,
the columnar pack at ``DISCO_COLUMNS``) a shell has its own delta / Delta / TE and a Camino scheme file can be uploaded.
Nothing scientific lives here: every number comes from the pipeline, every figure from :mod:`space.viewers`."""
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
UPLOADED = "uploaded scheme"
PER_SHELL = 7                                            # on, timing class, b, directions, delta, Delta, TE
N_PHYSICS = 17                                           # on, field, B0 preset, theta, phi, 3 T2, 3 T1, rho, chi_iso, chi_aniso, 3 tiers
FREE_B0 = "free (polar and azimuth angles below)"
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
                if P.mode(cfg) == "full":
                    layout = P.Columns(cfg)
                else:
                    layout = P.Layout(cfg, local=os.environ.get("DISCO_MOMENTS"))
                layout.warm()
                from dmipy_sim.io.strands import read_tck, read_diameters
                _state["strands"] = read_tck(os.path.join(P.DATA_DIR, "DiSCo_Strands_Trajectories.tck"), coordinate_unit_m=1.0)
                _state["diameters"] = read_diameters(os.path.join(P.DATA_DIR, "DiSCo_Strands_Diameters.txt"), diameter_unit_m=1.0)
                _state["layout"] = layout; _state["cfg"] = cfg; _state["load_seconds"] = time.perf_counter() - t0
            except Exception as e:                       # shown on the page instead of a dead Space
                _state["error"] = repr(e)
        return _state


def _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs, scheme=None, full=False):
    """The acquisition the page asks for: DiSCo's own table, a config preset, the shell rows (in full mode each with
    its own delta / Delta / TE in ms, else on a stored timing class), or in full mode an uploaded Camino scheme."""
    if preset == "DiSCo 364":
        return P.disco_protocol(cfg)[0]
    if preset == UPLOADED:
        if not full:
            raise ValueError("a scheme upload needs full mode (DISCO_MODE=full with the columnar pack)")
        if not scheme:
            raise ValueError("upload a Camino .scheme file")
        return P.protocol_from_scheme(scheme)
    if preset in cfg["presets"]:
        p = cfg["presets"][preset]
        return P.Protocol(tuple(P.Shell(s["shape"], float(s["b"]), int(s["n_dirs"])) for s in p["shells"]), n_b0=int(p["n_b0"]), name=preset)
    if preset != CUSTOM:
        raise ValueError(f"unknown acquisition {preset!r}")
    shells = []
    for k in range(MAX_SHELLS):
        on, shape, b, n, delta, Delta, TE = shell_inputs[PER_SHELL * k: PER_SHELL * (k + 1)]
        if on and full:
            shells.append(P.Shell(f"d{float(delta):g}-D{float(Delta):g}-TE{float(TE):g}", float(b), int(n),
                                  delta=float(delta) * 1e-3, Delta=float(Delta) * 1e-3, TE=float(TE) * 1e-3))
        elif on:
            shells.append(P.Shell(str(shape), float(b), int(n)))
    return P.Protocol(tuple(shells), n_b0=int(n_b0), name=CUSTOM)


def _physics_from_inputs(cfg, on, field_T, b0_mode, theta, phi, T2i, T2e, T2m, T1i, T1e, T1m, rho, chi_iso, chi_aniso, relax, contact, fld):
    """The page's tissue-and-scanner panel as a :class:`space.pipeline.Physics` (None when the panel is off): T2 and
    T1 in ms, rho in um/s, chi in ppm on the page; SI in the pipeline."""
    if not on:
        return None
    u = P.B0_PRESETS[b0_mode] if b0_mode in P.B0_PRESETS else P.b0_direction(theta, phi)
    return P.Physics(field_T=float(field_T), b0_direction=u,
                     T2={"intra": float(T2i) * 1e-3, "extra": float(T2e) * 1e-3, "myelin": float(T2m) * 1e-3},
                     T1={"intra": float(T1i) * 1e-3, "extra": float(T1e) * 1e-3, "myelin": float(T1m) * 1e-3},
                     rho=float(rho) * 1e-6, chi_iso=float(chi_iso) * 1e-6, chi_aniso=float(chi_aniso) * 1e-6,
                     relaxation=bool(relax), contact=bool(contact), field=bool(fld))


def catalogue_numbers(field_T):
    """The catalogue's white matter at ``field_T`` in the page's units (ms, um/s, ppm) plus the note that says which
    cited field it came from: what the reset button and the field presets fill in."""
    c = P.catalogue(float(field_T))
    note = (f"catalogue values at {c['catalogue_field']:g} T" if abs(c["catalogue_field"] - float(field_T)) < 1e-9
            else f"the catalogue has no cited relaxation at {float(field_T):g} T: nearest is {c['catalogue_field']:g} T, edit as you see fit")
    return ([c["T2"][q] * 1e3 for q in P.POOLS] + [c["T1"][q] * 1e3 for q in P.POOLS]
            + [c["rho"] * 1e6, c["chi_iso"] * 1e6, c["chi_aniso"] * 1e6, note])


def _result_state(res, layout, state):
    """What the results tab needs, kept per session: float32 volumes, the peaks, a streamline sample, the numbers."""
    peaks, amp = P.peaks(res.sh)
    return dict(dwi=res.dwi.astype(np.float32), meas=res.meas, peaks=peaks.astype(np.float32), peak_amp=amp.astype(np.float32),
                matrix=res.matrix, score=res.score, seconds=res.seconds, load_seconds=state.get("load_seconds", float("nan")),
                tractogram=res.tractogram, rois=layout.rois, shape=layout.mask.shape, name=res.protocol.name)


def run_pipeline(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, *rest, progress=None):
    """The run button, a generator: while a stage runs it yields the stage's name into the headline (the other
    outputs untouched), and last the results. A generator reaches the page from inside any worker, where a progress
    object does not."""
    import gradio as gr
    keep = gr.update()
    n_out = 9
    if progress:
        progress(0.0, desc="starting")
    state = _load()
    if state["error"]:
        raise gr.Error(f"the layout did not load: {state['error']}")
    layout, cfg = state["layout"], state["cfg"]
    full = isinstance(layout, P.Columns)
    physics_inputs, shell_inputs = rest[:N_PHYSICS], rest[N_PHYSICS:]
    try:
        protocol = _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs, scheme=scheme_file, full=full)
        physics = _physics_from_inputs(cfg, *physics_inputs)
        if full:                                             # the plan before any byte moves: what it reads, how long
            plan = layout.plan(P.measurements(protocol, cfg["shapes"]))
            yield (keep, f"**full replay: {plan['rows']:,} rows, {plan['bytes'] / 1e9:.1f} GB to read, about "
                         f"{plan['estimated_seconds'] / 60:.0f} min** (bands {plan['K']}, modes {plan['M']})") + (keep,) * (n_out - 1)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    tracking = P.Tracking(density=int(density), step_mm=float(step_mm), max_angle=float(max_angle), key=int(key))
    t0 = time.perf_counter()
    res = None
    for item in P.run_stages(layout, protocol, snr=(float(snr) if snr_on else None), tracking=tracking, physics=physics):
        if isinstance(item, P.Result):
            res = item
            break
        stage, k, n = item
        text = f"**{k + 1}/{n} {STAGE_TEXT[stage]}** … ({protocol.n_meas} measurements, {time.perf_counter() - t0:.0f} s so far)"
        if progress:
            progress((k + 0.5) / (n + 1), desc=f"{k + 1}/{n} {STAGE_TEXT[stage]}")
        yield (keep, text) + (keep,) * (n_out - 1)
    yield (keep, f"**writing the downloads and drawing** … ({time.perf_counter() - t0:.0f} s so far)") + (keep,) * (n_out - 1)
    out = tempfile.mkdtemp(); stem = f"disco_{protocol.name.replace(' ', '_')}"
    tck = os.path.join(out, f"{stem}.tck"); res.tractogram.to_tck(tck)
    vols = P.write_volumes(res, out, prefix=stem)
    rs = _result_state(res, layout, state)
    s = res.score
    headline = (f"**Pearson vs strand count {s['pearson_count']:.3f}, vs cross-sectional area {s['pearson_area']:.3f}** "
                f"({protocol.n_meas} measurements, {physics.label() if physics else 'bare diffusion'}, {len(res.tractogram):,} streamlines, "
                f"{res.seconds['total']:.1f} s in total; replay floor median {np.nanmedian(res.floor[layout.mask]):.4f}). Results are in the third tab.")
    z0 = rs["dwi"].shape[2] // 2
    m0 = int(np.flatnonzero(~rs["meas"].b0)[0]) if (~rs["meas"].b0).any() else 0
    yield (rs, headline, V.dwi_slice(rs["dwi"], rs["meas"], z0, m0, peaks=rs["peaks"], peak_amp=rs["peak_amp"], overlay=True),
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


def build(runner=None):
    """The Blocks. ``runner`` wraps :func:`run_pipeline` for the run button (the ZeroGPU entry passes
    ``spaces.GPU(...)``); the wrapper receives the same positional inputs and Gradio's progress."""
    import gradio as gr
    cfg = P.config()
    full = P.mode(cfg) == "full"
    shapes = cfg["shapes"]; presets = ["DiSCo 364"] + list(cfg["presets"]) + [CUSTOM] + ([UPLOADED] if full else [])
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
                                shape = gr.Dropdown(shape_names, value=shape_names[0], label="pulse timing δ / Δ", visible=not full)
                                b = gr.Number(value=[1000, 2000, 3000, 6000][k], label="b (s/mm²)")
                                n = gr.Slider(6, 128, value=[30, 60, 90, 60][k], step=1, label="directions")
                                delta = gr.Number(value=10.2, label="δ (ms)", visible=full)
                                Delta = gr.Number(value=16.7, label="Δ (ms)", visible=full)
                                TE = gr.Number(value=53.5, label="TE (ms)", visible=full)
                            shell_inputs += [on, shape, b, n, delta, Delta, TE]
                        if full:
                            gr.Markdown("**Full mode**: the columnar replay pack is read for every run, so a shell's δ / Δ / TE are free "
                                        "(one TE per run, square pulses) and a Camino `.scheme` file can be uploaded; a run takes minutes, "
                                        "the plan shown first says how many.")
                        else:
                            gr.Markdown("**Demo mode**: a shell's **pulse timing** (δ, Δ; TE 53.5 ms, square pulses) is one of the stored classes, "
                                        "because the layout holds each walker's response to that pulse shape; the b-value, the directions and their "
                                        "number, the SNR and the tracker are free. Stored: " + "; ".join(f"`{n}` = {labels[n]}" for n in shape_names)
                                        + ". The same image beside the columnar pack (`DISCO_MODE=full`) replays any timing, in minutes.")
                        scheme_file = gr.File(label="uploaded scheme: Camino STEJSKALTANNER (.scheme)", file_count="single", type="filepath", visible=full)
                    with gr.Column(scale=1):
                        phys = cfg["physics"]; f0 = float(phys["default_field"]); c0 = catalogue_numbers(f0)
                        field_presets = {f"{f:g} T": float(f) for f in phys["fields"]}
                        with gr.Accordion("tissue and scanner: the physics tiers", open=True):
                            physics_on = gr.Checkbox(value=bool(phys["default_on"]), label="evaluate the walk in tissue at a field (off: bare diffusion)")
                            with gr.Row():
                                field_preset = gr.Dropdown(list(field_presets), value=f"{f0:g} T", label="field preset")
                                field_T = gr.Slider(0.05, 12.0, value=f0, step=0.001, label="B0 (T)")
                            with gr.Row():
                                b0_mode = gr.Dropdown(list(P.B0_PRESETS) + [FREE_B0], value=list(P.B0_PRESETS)[0], label="B0 direction")
                                theta = gr.Slider(0, 180, value=0, step=1, label="polar angle from z (°)")
                                phi = gr.Slider(0, 360, value=0, step=1, label="azimuth from x (°)")
                            with gr.Row():
                                tier_relax = gr.Checkbox(value=True, label="relaxation (T2, T1)")
                                tier_contact = gr.Checkbox(value=True, label="contact (surface relaxivity ρ)")
                                tier_field = gr.Checkbox(value=True, label="field (myelin susceptibility)")
                            with gr.Row():
                                T2i = gr.Number(value=c0[0], label="T2 intra (ms)"); T2e = gr.Number(value=c0[1], label="T2 extra (ms)"); T2m = gr.Number(value=c0[2], label="T2 myelin (ms)")
                            with gr.Row():
                                T1i = gr.Number(value=c0[3], label="T1 intra (ms)"); T1e = gr.Number(value=c0[4], label="T1 extra (ms)"); T1m = gr.Number(value=c0[5], label="T1 myelin (ms)")
                            with gr.Row():
                                rho = gr.Number(value=c0[6], label="ρ (µm/s)"); chi_iso = gr.Number(value=c0[7], label="χ_iso myelin (ppm)"); chi_aniso = gr.Number(value=c0[8], label="Δχ_a myelin (ppm)")
                            with gr.Row():
                                catalogue_note = gr.Markdown(c0[9])
                                reset = gr.Button("reset to the catalogue at this field", size="sm")
                        physics_inputs = [physics_on, field_T, b0_mode, theta, phi, T2i, T2e, T2m, T1i, T1e, T1m, rho, chi_iso, chi_aniso,
                                          tier_relax, tier_contact, tier_field]
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
            yield from run_pipeline(*args, progress=progress)
        # a runner (the ZeroGPU entry) owns the call and its progress object: Gradio hands it the inputs only
        tissue_numbers = [T2i, T2e, T2m, T1i, T1e, T1m, rho, chi_iso, chi_aniso, catalogue_note]
        field_preset.change(lambda name: [field_presets[name]] + catalogue_numbers(field_presets[name]), inputs=field_preset,
                            outputs=[field_T] + tissue_numbers, show_progress="hidden")
        reset.click(catalogue_numbers, inputs=field_T, outputs=tissue_numbers, show_progress="hidden")
        go.click(run_with_progress if runner is None else runner(run_pipeline), inputs=[preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, *physics_inputs, *shell_inputs],
                 outputs=[result, headline, dwi_view, tract_view, mats, timings, tck, volumes, z_slider, m_slider], concurrency_limit=1,
                 api_name="run_pipeline")                                       # the endpoint tools/live_check.py drives
        for ctl in (z_slider, m_slider, overlay):
            ctl.change(redraw_slice, inputs=[result, z_slider, m_slider, overlay], outputs=dwi_view, show_progress="hidden")
        gt_tab.select(ground_truth_views, outputs=[strands_view, gt_matrix])
        demo.load(lambda: (_load().get("error") and f"**the layout did not load:** {_load()['error']}") or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
