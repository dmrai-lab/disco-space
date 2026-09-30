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
NO_KNOB = "none: run A only"
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


def knobs(cfg):
    """The one-knob changes B can make to A (disco-space#4 iteration 2): the field (with the catalogue's tissue at
    it), the field's direction, a tier off, the tissue off, the noise, or every shell's pulse timing."""
    out = {NO_KNOB: None}
    for f in cfg["physics"]["fields"]:
        out[f"field → {float(f):g} T (catalogue tissue at that field)"] = ("field", float(f))
    out["B0 direction → transverse (90° from z)"] = ("b0", list(P.B0_PRESETS)[1])
    out["B0 direction → along z"] = ("b0", list(P.B0_PRESETS)[0])
    for i, name in ((14, "relaxation"), (15, "contact"), (16, "field")):
        out[f"{name} tier → off"] = ("tier", i)
    out["tissue → off (bare diffusion)"] = ("bare", None)
    for v in (10, 100):
        out[f"SNR → {v}"] = ("snr", float(v))
    out["noise → off"] = ("snr", None)
    for name, sh in cfg["shapes"].items():
        out[f"every shell's pulse timing → {name} ({sh['label']})"] = ("shape", name)
    return out


def apply_knob(cfg, change, protocol, snr_on, snr, physics_inputs):
    """B's settings: A's with ``change`` (a value of :func:`knobs`) applied; ``(protocol, snr_on, snr, physics_inputs)``."""
    kind, value = change
    ph = list(physics_inputs)
    if kind == "field":
        ph[0] = True; ph[1] = value; ph[5:14] = catalogue_numbers(value)[:9]
    elif kind == "b0":
        ph[0] = True; ph[2] = value
    elif kind == "tier":
        ph[0] = True; ph[value] = False
    elif kind == "bare":
        ph[0] = False
    elif kind == "snr":
        snr_on = value is not None; snr = value if value is not None else snr
    elif kind == "shape":
        protocol = P.retime(protocol, value)
    else:
        raise ValueError(f"unknown knob {kind!r}")
    return protocol, snr_on, snr, ph


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


def _one_run(tag, layout, cfg, protocol, snr_on, snr, tracking, physics, t0, progress, keep, n_out):
    """One pipeline run as a generator of page updates, then ``(Result, files)`` last: ``tag`` is A or B in the
    stage text."""
    res = None
    for item in P.run_stages(layout, protocol, snr=(float(snr) if snr_on else None), tracking=tracking, physics=physics):
        if isinstance(item, P.Result):
            res = item
            break
        stage, k, n = item
        text = f"**{tag} · {k + 1}/{n} {STAGE_TEXT[stage]}** … ({protocol.n_meas} measurements, {time.perf_counter() - t0:.0f} s so far)"
        if progress:
            progress((k + 0.5) / (n + 1), desc=f"{tag} {k + 1}/{n} {STAGE_TEXT[stage]}")
        yield (keep, text) + (keep,) * (n_out - 1)
    out = tempfile.mkdtemp(); stem = f"disco_{tag}_{protocol.name.replace(' ', '_')}"
    tck = os.path.join(out, f"{stem}.tck"); res.tractogram.to_tck(tck)
    vols = P.write_volumes(res, out, prefix=stem)
    yield res, dict(tck=tck, volumes=[vols["dwi"], vols["bvals"], vols["bvecs"], vols["fod"]])


def _score_text(tag, res, layout):
    s = res.score; physics = res.physics
    return (f"**{tag}: Pearson vs strand count {s['pearson_count']:.3f}, vs area {s['pearson_area']:.3f}** "
            f"({res.protocol.n_meas} measurements, {physics.label() if physics else 'bare diffusion'}"
            f"{'' if res.snr is None else f', SNR {res.snr:g}'}, {len(res.tractogram):,} streamlines, {res.seconds['total']:.1f} s; "
            f"replay floor median {np.nanmedian(res.floor[layout.mask]):.4f})")


def run_pipeline(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, *rest, progress=None):
    """The run button, a generator: while a stage runs it yields the stage's name into the headline (the other
    outputs untouched), and last the results: A, and B when ``knob`` changes one thing (the same tracker key, so
    the difference is the knob's). A generator reaches the page from inside any worker, where a progress object
    does not."""
    import gradio as gr
    keep = gr.update()
    n_out = 12
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
        runs = [("A", protocol, snr_on, snr, _physics_from_inputs(cfg, *physics_inputs))]
        change = knobs(cfg).get(knob, "?")
        if change == "?":
            raise ValueError(f"unknown knob {knob!r}")
        if change is not None:
            pb, on_b, snr_b, ph_b = apply_knob(cfg, change, protocol, snr_on, snr, physics_inputs)
            runs.append(("B", pb, on_b, snr_b, _physics_from_inputs(cfg, *ph_b)))
        if full:                                             # the plan before any byte moves: what it reads, how long
            plan = layout.plan(P.measurements(protocol, cfg["shapes"]))
            yield (keep, f"**full replay: {plan['rows']:,} rows, {plan['bytes'] / 1e9:.1f} GB to read, about "
                         f"{plan['estimated_seconds'] / 60:.0f} min per run** (bands {plan['K']}, modes {plan['M']})") + (keep,) * (n_out - 1)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    tracking = P.Tracking(density=int(density), step_mm=float(step_mm), max_angle=float(max_angle), key=int(key))
    t0 = time.perf_counter()
    results = {}; files = {}
    for tag, prot, on, s_, ph in runs:
        for item in _one_run(tag, layout, cfg, prot, on, s_, tracking, ph, t0, progress, keep, n_out):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], P.Result):
                results[tag], files[tag] = item
            else:
                yield item
    yield (keep, f"**drawing** … ({time.perf_counter() - t0:.0f} s so far)") + (keep,) * (n_out - 1)
    rs = {tag: _result_state(r, layout, state) for tag, r in results.items()}
    ra = rs["A"]; rb = rs.get("B")
    headline = _score_text("A", results["A"], layout)
    if rb:
        c = P.compare(results["A"], results["B"])
        headline += ("<br>" + _score_text("B", results["B"], layout)
                     + f"<br>**A vs B: connectome Pearson {c['pearson_ab']:.3f}**, {c['only_a']} pairs in A only, {c['only_b']} in B only, "
                       f"B − A {c['delta_count']:+.3f} vs count, {c['delta_area']:+.3f} vs area; B = A with {knob}.")
    headline += " Results are in the third tab."
    z0 = ra["dwi"].shape[2] // 2
    m0 = int(np.flatnonzero(~ra["meas"].b0)[0]) if (~ra["meas"].b0).any() else 0
    timings = [[f"{tag} · {r}", t] for tag in rs for r, t in V.timings_rows(rs[tag]["seconds"], rs[tag]["load_seconds"])] if rb else V.timings_rows(ra["seconds"], ra["load_seconds"])
    yield (rs, headline, V.dwi_slice(ra["dwi"], ra["meas"], z0, m0, peaks=ra["peaks"], peak_amp=ra["peak_amp"], overlay=True),
           V.tractogram3d(ra["tractogram"], ra["rois"], ra["shape"]), V.matrices(ra["matrix"], ra["score"], layout.gt_count),
           timings, [f["tck"] for f in files.values()], [v for f in files.values() for v in f["volumes"]],
           _slider_update(z0, ra["dwi"].shape[2] - 1), _slider_update(m0, protocol.n_meas - 1),
           V.tractogram3d(rb["tractogram"], rb["rois"], rb["shape"]) if rb else None,
           V.matrices(rb["matrix"], rb["score"], layout.gt_count) if rb else None,
           gr.update(visible=rb is not None))


def _slider_update(value, maximum):
    import gradio as gr
    return gr.update(value=int(value), maximum=int(maximum))


def redraw_slice(rs, z, m, overlay, which):
    if not rs or which not in rs:
        return None
    r = rs[which]
    return V.dwi_slice(r["dwi"], r["meas"], int(z), int(m), peaks=r["peaks"], peak_amp=r["peak_amp"], overlay=bool(overlay))


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
                        knob = gr.Dropdown(list(knobs(cfg)), value=NO_KNOB, label="B: the same run with one knob changed")
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
                            which = gr.Radio(["A", "B"], value="A", label="run")
                    with gr.Column(scale=1):
                        tract_view = gr.Plot(label="tractogram A")
                mats = gr.Image(label="connectome A vs ground truth", type="pil")
                with gr.Row(visible=False) as b_row:
                    tract_view_b = gr.Plot(label="tractogram B")
                    mats_b = gr.Image(label="connectome B vs ground truth", type="pil")
                with gr.Row():
                    timings = gr.Dataframe(headers=["stage", "seconds"], label="timings", interactive=False)
                    with gr.Column():
                        tck = gr.File(label="tractograms (.tck, MRtrix)", file_count="multiple")
                        volumes = gr.File(label="DWI (.nii.gz) with bvals/bvecs, and the FOD SH field (.nii.gz, tournier07 order 8)", file_count="multiple")
        def run_with_progress(*args, progress=gr.Progress()):
            yield from run_pipeline(*args, progress=progress)
        # a runner (the ZeroGPU entry) owns the call and its progress object: Gradio hands it the inputs only
        tissue_numbers = [T2i, T2e, T2m, T1i, T1e, T1m, rho, chi_iso, chi_aniso, catalogue_note]
        field_preset.change(lambda name: [field_presets[name]] + catalogue_numbers(field_presets[name]), inputs=field_preset,
                            outputs=[field_T] + tissue_numbers, show_progress="hidden")
        reset.click(catalogue_numbers, inputs=field_T, outputs=tissue_numbers, show_progress="hidden")
        go.click(run_with_progress if runner is None else runner(run_pipeline), inputs=[preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, *physics_inputs, *shell_inputs],
                 outputs=[result, headline, dwi_view, tract_view, mats, timings, tck, volumes, z_slider, m_slider, tract_view_b, mats_b, b_row], concurrency_limit=1,
                 api_name="run_pipeline")                                       # the endpoint tools/live_check.py drives
        for ctl in (z_slider, m_slider, overlay, which):
            ctl.change(redraw_slice, inputs=[result, z_slider, m_slider, overlay, which], outputs=dwi_view, show_progress="hidden")
        gt_tab.select(ground_truth_views, outputs=[strands_view, gt_matrix])
        demo.load(lambda: (_load().get("error") and f"**the layout did not load:** {_load()['error']}") or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
