"""The DiSCo Space's page, three tabs over :mod:`space.pipeline`.

The acquisition tab: a preset, DiSCo's own table or custom shells (on a stored pulse timing in demo mode; at their
own δ / Δ / TE, or from an uploaded Camino scheme, in full mode); the tissue-and-scanner panel (the field and its
direction, T2 and T1 per pool, the surface relaxivity, myelin's susceptibility, each tier switchable); the noise;
the tracker; a scanner whose gradient limit refuses a shell it cannot play; B, the same run with one knob changed;
N tracker keys for the tractogram's own spread; the run button, whose progress names the stage it is in. The ground
truth tab: the strands in 3-D and the two matrices. The Replay DWI Explorer: the ingredient maps of the tiers, the
noise-free layers (the tier ladder, A, B) and their differences in the DWI or in MD / FA, per shell. The DiSCo
results tab: a DWI slice with the FODs' principal directions and the replay floor beside it, the tractograms and
connectomes of A and B, the spread over keys, the accuracy table, the timings, the downloads.

Nothing scientific lives here: every number comes from the pipeline, every figure from :mod:`space.viewers`."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import numpy as np
from dataclasses import replace

from . import pipeline as P
from . import viewers as V

MAX_SHELLS = 4
CUSTOM = "custom shells"
UPLOADED = "uploaded scheme"
PER_SHELL = 7                                            # on, timing class, b, directions, delta, Delta, TE
POOLS = tuple(P.config()["physics"]["pools"])           # the seeded pools the panel offers T2 and T1 for
PHYSICS_FIELDS = ("on", "field_T", "b0_mode", "theta", "phi", *[f"T2_{q}" for q in POOLS], *[f"T1_{q}" for q in POOLS],
                  "rho", "chi_iso", "chi_aniso", "relaxation", "contact", "field")     # the panel's inputs, in the run signature's order
TISSUE_NUMBERS = PHYSICS_FIELDS[5:5 + 2 * len(POOLS) + 3]   # the ones the catalogue fills in: ms, ms, µm/s, ppm on the page
B0_MODES = {"along z (the strands' frame)": P.B0_ALONG_Z, "transverse (x): 90° from z, as in a biplanar magnet like the Swoop": P.B0_TRANSVERSE}
FREE_B0 = "free (polar and azimuth angles below)"
NO_KNOB = "none: run A only"
NO_SCANNER = "none: any gradient amplitude"
SAMPLE = 10_000                                          # streamlines kept in the page's state and in the sample .tck
STAGE_TEXT = {"replay": "replaying the grid from the stored walk", "noise": "adding Rician noise", "csd": "fitting CSD (order 8)",
              "track": "tracking from the sixteen regions", "score": "scoring the connectome"}
OUTPUTS = ("result", "headline", "dwi_view", "tract_view", "mats", "timings", "tck", "volumes", "z_slider", "m_slider",
           "tract_view_b", "mats_b", "b_row", "floor_view", "accuracy", "spread_view",
           "explore_view", "metric_view", "ingredient_pool", "ingredient_contact", "ingredient_field", "layer_table", "layer_choice",
           "ez_slider", "em_slider")                                                     # the run button's outputs, in order
EXPLORE_MODES = ("signal", "minus the previous layer", "B − A", "(B − A) ÷ floor")
METRICS = ("DWI", "MD (µm²/ms)", "FA")
_state = {"source": None, "error": None}
_lock = threading.Lock()
_RUNS = tempfile.mkdtemp(prefix="disco-runs-")          # one directory per run tag, replaced on every run


def _load():
    """The source and the ground truth, loaded once per process (a failed load is retried on the next call)."""
    with _lock:
        if _state["source"] is None:
            try:
                t0 = time.perf_counter()
                cfg = P.config()
                src = P.source(cfg)
                src.warm()
                from dmipy_sim.io.strands import read_tck, read_diameters
                strands = read_tck(os.path.join(P.DATA_DIR, "DiSCo_Strands_Trajectories.tck"), coordinate_unit_m=P.VOXEL_M)
                _state["strands_vox"] = [s / P.VOXEL_M for s in strands]
                _state["diameters_m"] = read_diameters(os.path.join(P.DATA_DIR, "DiSCo_Strands_Diameters.txt"), diameter_unit_m=1e-3)
                _state["regions"] = V.region_markers(src.rois)
                _state["gt_views"] = None
                _state["source"] = src; _state["cfg"] = cfg; _state["load_seconds"] = time.perf_counter() - t0; _state["error"] = None
            except Exception as e:                       # shown on the page instead of a dead Space; the next call tries again
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
        return P.preset_protocol(cfg, preset)
    if preset != CUSTOM:
        raise ValueError(f"unknown acquisition {preset!r}")
    shells = []
    for k in range(MAX_SHELLS):
        on, shape, b, n, delta, Delta, TE = shell_inputs[PER_SHELL * k: PER_SHELL * (k + 1)]
        if on and full:
            shells.append(P.Shell.free(float(b), int(n), float(delta) * 1e-3, float(Delta) * 1e-3, float(TE) * 1e-3))
        elif on:
            shells.append(P.Shell(str(shape), float(b), int(n)))
    return P.Protocol(tuple(shells), n_b0=int(n_b0), name=CUSTOM)


def physics_values(*values):
    """The panel's inputs as a dict by :data:`PHYSICS_FIELDS`."""
    if len(values) != len(PHYSICS_FIELDS):
        raise ValueError(f"the physics panel has {len(PHYSICS_FIELDS)} inputs, got {len(values)}")
    return dict(zip(PHYSICS_FIELDS, values))


def physics_from(values):
    """The panel (a dict by :data:`PHYSICS_FIELDS`, page units: ms, µm/s, ppm) as a :class:`space.pipeline.Physics`
    in SI, or None when the panel is off."""
    if not values["on"]:
        return None
    u = B0_MODES[values["b0_mode"]] if values["b0_mode"] in B0_MODES else P.b0_direction(values["theta"], values["phi"])
    return P.Physics(field_T=float(values["field_T"]), b0_direction=u,
                     T2={q: float(values[f"T2_{q}"]) * 1e-3 for q in POOLS}, T1={q: float(values[f"T1_{q}"]) * 1e-3 for q in POOLS},
                     rho=float(values["rho"]) * 1e-6, chi_iso=float(values["chi_iso"]) * 1e-6, chi_aniso=float(values["chi_aniso"]) * 1e-6,
                     relaxation=bool(values["relaxation"]), contact=bool(values["contact"]), field=bool(values["field"]))


def catalogue_numbers(field_T):
    """The catalogue's white matter at ``field_T`` in the page's units (ms, µm/s, ppm), in :data:`TISSUE_NUMBERS`
    order, plus the note that says which cited field it came from: what the reset button and the field presets
    fill in."""
    c = P.catalogue(float(field_T), POOLS)
    note = (f"catalogue values at {c['catalogue_field']:g} T" if abs(c["catalogue_field"] - float(field_T)) < 1e-9
            else f"the catalogue has no cited relaxation at {float(field_T):g} T: nearest is {c['catalogue_field']:g} T, edit as you see fit")
    return ([c["T2"][q] * 1e3 for q in POOLS] + [c["T1"][q] * 1e3 for q in POOLS]
            + [c["rho"] * 1e6, c["chi_iso"] * 1e6, c["chi_aniso"] * 1e6, note])


def knobs(cfg):
    """The one-knob changes B can make to A: the field (with the catalogue's tissue at it), the field's direction,
    a tier off, the tissue off, the noise, or every shell's pulse timing; ``{label: (kind, value)}``."""
    out = {NO_KNOB: None}
    for f in cfg["physics"]["fields"]:
        out[f"field → {float(f):g} T (catalogue tissue at that field)"] = ("field", float(f))
    for label in B0_MODES:
        out[f"B0 direction → {label}"] = ("b0", label)
    for tier in ("relaxation", "contact", "field"):
        out[f"{tier} tier → off"] = ("tier", tier)
    out["tissue → off (bare diffusion)"] = ("bare", None)
    for v in (10, 100):
        out[f"SNR → {v}"] = ("snr", float(v))
    out["noise → off"] = ("snr", None)
    for name, sh in cfg["shapes"].items():
        out[f"every shell's pulse timing → {name} ({sh['label']})"] = ("shape", name)
    return out


def apply_knob(change, protocol, snr_on, snr, values):
    """B's settings: A's with ``change`` (a value of :func:`knobs`) applied; ``(protocol, snr_on, snr, values)``.
    A knob that changes the tissue panel needs A's panel on, so that B differs from A in that one thing."""
    kind, value = change
    v = dict(values)
    if kind in ("field", "b0", "tier") and not v["on"]:
        raise ValueError(f"the knob {kind!r} changes the tissue panel, which is off for A: switch it on, or choose another knob")
    if kind == "field":
        v["field_T"] = value; v.update(zip(TISSUE_NUMBERS, catalogue_numbers(value)))
    elif kind == "b0":
        v["b0_mode"] = value
    elif kind == "tier":
        v[value] = False
    elif kind == "bare":
        v["on"] = False
    elif kind == "snr":
        snr_on = value is not None; snr = value if value is not None else snr
    elif kind == "shape":
        protocol = P.retime(protocol, value)
    else:
        raise ValueError(f"unknown knob {kind!r}")
    return protocol, snr_on, snr, v


def gradient_text(protocol, shapes, scanner):
    """The per-shell gradient table as Markdown, and whether every shell is playable on ``scanner``."""
    rows = P.playable(protocol, shapes, scanner if scanner != NO_SCANNER else None)
    lines = ["| shell | b (s/mm²) | timing | needs | limit |", "|---|---|---|---|---|"]
    for name, b, delta, Delta, G, G_max, ok in rows:
        limit = "any" if G_max is None else f"{G_max * 1e3:.0f} mT/m {'✓' if ok else '✗ cannot play'}"
        lines.append(f"| {name} | {b:g} | δ {delta * 1e3:g} / Δ {Delta * 1e3:g} ms | {G * 1e3:.0f} mT/m | {limit} |")
    return "\n".join(lines), all(r[-1] for r in rows)


def plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs):
    """The runs the button asks for, validated before any work: ``[(tag, protocol, snr_on, snr, physics)]`` for A
    and, with a knob, B; refused with the reason when the scanner cannot play a shell or the source cannot do the
    run."""
    full = source.mode == "full"
    protocol = _protocol_from_inputs(cfg, preset, n_b0, *shell_inputs, scheme=scheme_file, full=full)
    runs = [("A", protocol, snr_on, snr, physics_from(values))]
    if knob not in knobs(cfg):
        raise ValueError(f"unknown knob {knob!r}")
    change = knobs(cfg)[knob]
    if change is not None:
        pb, on_b, snr_b, vb = apply_knob(change, protocol, snr_on, snr, values)
        runs.append(("B", pb, on_b, snr_b, physics_from(vb)))
    for tag, prot, _, _, physics in runs:
        table, ok = gradient_text(prot, cfg["shapes"], scanner)
        if not ok:
            raise ValueError(f"{scanner} cannot play run {tag}'s shells (square pulses):\n\n{table}")
        source.validate(prot, physics)
    return runs


def _result_state(res, sample, load_seconds):
    """What the results tab needs, kept per session: float32 volumes, the peaks, the streamline sample, the numbers."""
    pk, amp = P.peaks(res.sh)
    return dict(dwi=res.dwi.astype(np.float32), floor=res.floor.astype(np.float32), floor_median=P.floor_stats(res)["median"], meas=res.meas,
                peaks=pk.astype(np.float32), peak_amp=amp.astype(np.float32), matrix=res.matrix, score=res.score, seconds=res.seconds,
                load_seconds=load_seconds, tractogram=sample, n_streamlines=len(res.tractogram), shape=res.dwi.shape[:3], name=res.protocol.name)


def _layer_labels(physics, knob):
    """The explorer's names for A and B: A is the ladder's top rung, named by every tier it has on; B is A with the
    knob."""
    tiers = [q for q in ("relaxation", "contact", "field") if physics and getattr(physics, q)]
    a = "A = bare" + "".join(f" + {q}" for q in tiers) if tiers else "A = bare diffusion"
    return a, f"B = A with {knob}"


def _explorer_state(results, ladder, ingredients, knob):
    """What the Replay DWI Explorer keeps per session: the noise-free layers (the ladder, then A, then B) as float16
    volumes with their tensor maps, the replay floor, the ingredient maps, the per-shell layer differences."""
    ra = results["A"]; mask = np.isfinite(ra.clean[..., 0])
    name_a, name_b = _layer_labels(ra.physics, knob)
    layers = list(ladder) + [(name_a, ra.clean)] + ([(name_b, results["B"].clean)] if "B" in results else [])
    metrics = {}
    for label, vol in layers:
        try:
            md, fa = P.dti(vol, ra.meas, mask)
        except ValueError:                                  # too few low-b rows for a tensor: no metric maps
            md = fa = None
        metrics[label] = (None if md is None else md.astype(np.float16), None if fa is None else fa.astype(np.float16))
    return dict(layers=[(label, vol.astype(np.float16)) for label, vol in layers], metrics=metrics, floor=ra.floor.astype(np.float32),
                meas=ra.meas, mask=mask, ingredients=ingredients, differences=P.layer_differences(layers, ra.meas, mask),
                snr=ra.snr, physics=ra.physics)


def _status(text):
    import gradio as gr
    keep = gr.update()
    return (keep, text) + (keep,) * (len(OUTPUTS) - 2)


def _one_run(tag, source, protocol, snr_on, snr, tracking, physics, t0):
    """One pipeline run as a generator of ``(text, fraction)`` stage updates, then the :class:`Result` last: ``tag``
    is A or B in the stage text."""
    for item in P.run_stages(source, protocol, snr=(float(snr) if snr_on else None), tracking=tracking, physics=physics):
        if isinstance(item, P.Result):
            yield item
            return
        stage, k, n = item
        yield (f"**{tag} · {k + 1}/{n} {STAGE_TEXT[stage]}** … ({protocol.n_meas} measurements, {time.perf_counter() - t0:.0f} s so far)", (k + 0.5) / (n + 1))


def _write_files(tag, res):
    """The run's files under its tag (the previous run's under the same tag replaced): the full tractogram and a
    sample as .tck, the DWI and FOD volumes; ``(sample, {"tck": [...], "volumes": [...]})``."""
    out = os.path.join(_RUNS, tag); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    stem = f"disco_{tag}_{res.protocol.name.replace(' ', '_')}"
    tck = os.path.join(out, f"{stem}.tck"); res.tractogram.to_tck(tck)
    sample = P.sample_tractogram(res.tractogram, SAMPLE)
    sample_path = os.path.join(out, f"{stem}_sample{SAMPLE // 1000}k.tck"); sample.to_tck(sample_path)
    vols = P.write_volumes(res, out, prefix=stem)
    return sample, dict(tck=[tck, sample_path], volumes=[vols["dwi"], vols["bvals"], vols["bvecs"], vols["fod"]])


def _score_text(tag, res):
    s = res.score; physics = res.physics
    snr = P.b0_snr(res)
    noise = "" if snr is None else f", SNR {res.snr:g} at M0 = {snr['median']:.1f} at b = 0 in the median voxel"
    return (f"**{tag}: Pearson vs strand count {s['pearson_count']:.3f}, vs area {s['pearson_area']:.3f}** "
            f"({res.protocol.n_meas} measurements, {physics.label() if physics else 'bare diffusion'}"
            f"{noise}, {len(res.tractogram):,} streamlines, {res.seconds['total']:.1f} s; "
            f"replay floor median {P.floor_stats(res)['median']:.4f})")


def explore(ex, layer, mode, metric, z, m):
    """The explorer's map: ``layer`` of the state's layers, in ``mode`` (the signal, its difference to the previous
    layer, B minus A, or that over the replay floor), as the DWI at measurement ``m`` or a tensor metric, slice ``z``."""
    if not ex:
        return None
    labels = [l for l, _ in ex["layers"]]; vols = dict(ex["layers"])
    if layer not in vols:
        layer = labels[-1]
    z = int(z); m = min(int(m), len(ex["meas"].bvals) - 1)
    k = METRICS.index(metric) if metric in METRICS else 0

    def field(label):
        if k == 0:
            return np.asarray(vols[label], np.float32)[..., m]
        md, fa = ex["metrics"][label]
        v = md if k == 1 else fa
        return None if v is None else np.asarray(v, np.float32)
    what = f"{metric} of {layer}" if k else f"{layer}: b = {ex['meas'].bvals[m]:g} s/mm², measurement {m}"
    x = field(layer)
    if x is None:
        return None
    if mode == "signal":
        return V.map_slice(x, z, f"{what}, slice z = {z}", cmap="gray" if k == 0 else "viridis", vmin=0 if k == 0 else None, vmax=1 if k != 1 else None)
    if mode == "minus the previous layer":
        i = labels.index(layer)
        if i == 0:
            return V.map_slice(x, z, f"{layer} is the first layer: its signal, slice z = {z}", cmap="gray", vmin=0, vmax=1)
        prev = field(labels[i - 1])
        return V.map_slice(x - prev, z, f"{what} minus {labels[i - 1]}, slice z = {z}", symmetric=True)
    a_label = next((l for l in labels if l.startswith("A")), None); b_label = next((l for l in labels if l.startswith("B")), None)
    if a_label is None or b_label is None:
        return V.map_slice(x, z, f"no B in this run (choose a knob in the acquisition tab): {what}, slice z = {z}", cmap="gray" if k == 0 else "viridis")
    d = field(b_label) - field(a_label)
    if mode == "B − A":
        return V.map_slice(d, z, f"B − A, {metric} at measurement {m}" if k == 0 else f"B − A, {metric}", symmetric=True)
    return V.map_slice(d / np.where(ex["floor"] > 0, ex["floor"], np.nan), z, f"(B − A) ÷ replay floor, {metric}" + (f" at measurement {m}" if k == 0 else ""), symmetric=True)


def ingredient_views(ex, z):
    """The three ingredient maps of the run's first class at slice ``z``: the intra-axonal weight fraction, the
    walkers' wall contact (and the contact tier's survival at the run's rho), the field's dephasing spread."""
    ing = ex and ex.get("ingredients")
    if not ing:
        return None, None, None
    z = int(z)
    pool = V.map_slice(ing["intra_fraction"], z, f"intra-axonal weight fraction (relaxation tier re-weights it), z = {z}", vmin=0, vmax=1)
    if ing.get("contact_survival") is not None:
        contact = V.map_slice(ing["contact_survival"], z, f"contact survival exp(−ρ ℓ / D) at the run's ρ (ℓ the walkers' wall contact), z = {z}", cmap="magma", vmax=1)
    elif ing.get("wall_contact_um") is not None:
        contact = V.map_slice(ing["wall_contact_um"], z, f"walkers' wall contact ℓ (boundary local time, µm), z = {z}", cmap="magma", unit="µm")
    else:
        contact = None
    fld = V.map_slice(ing["field_rad"], z, f"spread over the voxel's walkers of the sheath field's dephasing phase at the echo (rad), z = {z}", cmap="inferno", unit="rad") if ing.get("field_rad") is not None else None
    return pool, contact, fld


def layer_table(ex):
    rows = [[a, b, f"{sh:g}", f"{med:.4f}", f"{p99:.4f}"] for a, b, sh, med, p99 in ex["differences"]] if ex else []
    if ex:
        f = ex["floor"][ex["mask"]]
        rows.append(["replay floor", "", "", f"{np.median(f):.4f}", f"{np.quantile(f, 0.99):.4f}"])
    return rows


def compute(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """Everything a run needs the device for, a generator: ``(text, fraction)`` stage updates while it works, then
    the payload last, a dict of plain data (:class:`P.Result` per tag, the explorer's ladder and ingredient maps, the
    tracker-key spread, the seconds of each step, the clock at the handoff). ``rest`` is the physics panel then the
    shell rows. On a shared GPU pool this runs in the forked worker and its yields cross to the page's process;
    nothing here draws or writes a file, so the device is held for the compute alone."""
    state = _load()
    if state["error"]:
        raise ValueError(f"the source did not load: {state['error']}")
    source, cfg = state["source"], state["cfg"]
    values = physics_values(*rest[:len(PHYSICS_FIELDS)]); shell_inputs = rest[len(PHYSICS_FIELDS):]
    runs = plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs)
    for tag, prot, _, _, physics in runs:                # full mode: what each run reads, before any byte moves
        plan = source.plan(P.measurements(prot, cfg["shapes"]), physics)
        if plan:
            yield (f"**{tag}: full replay of {plan['rows']:,} rows, {plan['bytes'] / 1e9:.1f} GB to read, about "
                   f"{plan['estimated_seconds'] / 60:.0f} min** (bands {plan['K']}, field modes {plan['modes']})", 0.0)
    tracking = P.Tracking(density=int(density), step_mm=float(step_mm), max_angle=float(max_angle), key=int(key))
    t0 = time.perf_counter()
    results = {}; seconds = {}
    try:
        for tag, prot, on, s_, ph in runs:
            for item in _one_run(tag, source, prot, on, s_, tracking, ph, t0):
                if isinstance(item, P.Result):
                    results[tag] = item
                else:
                    yield item
        meas_a = results["A"].meas; physics_a = runs[0][4]
        ladder = []
        if ladder_on and physics_a is not None and not physics_a.bare:
            yield (f"**Replay DWI Explorer · the tier ladder of A, noise-free** … ({time.perf_counter() - t0:.0f} s so far)", 0.8)
            t = time.perf_counter(); ladder = source.ladder(meas_a, physics_a); seconds["explorer · ladder"] = time.perf_counter() - t
        yield (f"**Replay DWI Explorer · the ingredient maps** … ({time.perf_counter() - t0:.0f} s so far)", 0.85)
        t = time.perf_counter(); ingredients = source.ingredients(meas_a, physics_a); seconds["explorer · ingredients"] = time.perf_counter() - t
        spread = None
        if int(n_keys) > 1:                                 # A's tracking repeated over further keys: the tractogram's own spread
            keys = [int(key) + 1 + i for i in range(int(n_keys) - 1)]
            mats = [results["A"].matrix]; scores = [results["A"].score]
            for i, (k, M, sc, secs) in enumerate(P.repeat_tracking(results["A"], source, tracking, keys)):
                mats.append(M); scores.append(sc)
                yield (f"**A · tracking again with key {k} ({i + 2}/{int(n_keys)})** … ({time.perf_counter() - t0:.0f} s so far)", 0.9)
            spread = P.pair_spread(mats, scores)
    finally:
        source.release()
    yield dict(results={tag: _light(r) for tag, r in results.items()}, ladder=[(label, vol.astype(np.float32)) for label, vol in ladder],
               ingredients=ingredients, spread=spread, knob=knob, seconds=seconds, compute_seconds=time.perf_counter() - t0, handed_off_at=time.time())


def _light(res):
    """``res`` with its DWI volumes in float32 for the handoff: what the page keeps, draws, fits and writes is float32
    or narrower, so the payload carries half the bytes and the compute's own arithmetic is untouched."""
    return replace(res, dwi=res.dwi.astype(np.float32), clean=None if res.clean is None else res.clean.astype(np.float32))


def present(payload, load_seconds, regions, gt_count, accuracy):
    """The page's outputs (:data:`OUTPUTS`, in order) from a compute payload: the files, the per-session states, the
    headline and the figures. Runs where the page runs, never on the device."""
    import gradio as gr
    received = time.time()
    results = payload["results"]; ladder = payload["ladder"]; ingredients = payload["ingredients"]; spread = payload["spread"]; knob = payload["knob"]
    post = dict(payload["seconds"])
    t = time.perf_counter()
    samples = {}; files = {}
    for tag, r in results.items():
        samples[tag], files[tag] = _write_files(tag, r)
    post["page · files"] = time.perf_counter() - t; t = time.perf_counter()
    rs = {tag: _result_state(r, samples[tag], load_seconds) for tag, r in results.items()}
    ex = _explorer_state(results, ladder, ingredients, knob)
    post["page · states"] = time.perf_counter() - t; t = time.perf_counter()
    labels = [l for l, _ in ex["layers"]]; name_a = labels[-2] if "B" in results else labels[-1]
    ra = rs["A"]; rb = rs.get("B")
    headline = _score_text("A", results["A"])
    if rb:
        c = P.compare(results["A"], results["B"])
        headline += ("<br>" + _score_text("B", results["B"])
                     + f"<br>**A vs B: connectome Pearson {c['pearson_ab']:.3f}**, {c['only_a']} pairs in A only, {c['only_b']} in B only, "
                       f"B − A {c['delta_count']:+.3f} vs count, {c['delta_area']:+.3f} vs area; B = A with {knob}.")
    if spread:
        headline += (f"<br>**A over {spread['n']} tracker keys: Pearson vs count {spread['pearson_mean']:.3f} ± {spread['pearson_std']:.3f}**, "
                     f"median pair count CV {spread['cv_median']:.2f}, {spread['pairs_always']} pairs in every run, {spread['pairs_any']} in any.")
    headline += " The Replay DWI Explorer is the third tab, the DiSCo results the fourth."
    z0 = ra["dwi"].shape[2] // 2
    m0 = int(np.flatnonzero(~ra["meas"].b0)[0]) if (~ra["meas"].b0).any() else 0
    timings = [[f"{tag} · {r}", t_] for tag in rs for r, t_ in V.timings_rows(rs[tag]["seconds"], rs[tag]["load_seconds"])] if rb else V.timings_rows(ra["seconds"], ra["load_seconds"])
    timings += [["device held (compute)", f"{payload['compute_seconds']:.2f}"], ["handoff to the page", f"{received - payload['handed_off_at']:.2f}"]]
    timings += [[k, f"{v:.2f}"] for k, v in post.items()]
    rs["explorer"] = ex                                  # the page state: A, B and the explorer's layers
    out = (rs, headline, V.dwi_slice(ra["dwi"], ra["meas"], z0, m0, peaks=ra["peaks"], peak_amp=ra["peak_amp"], overlay=True, label="A: "),
           V.tractogram3d(ra["tractogram"], regions, ra["shape"], total=ra["n_streamlines"]), V.matrices(ra["matrix"], ra["score"], gt_count),
           timings, [t_ for f in files.values() for t_ in f["tck"]], [v for f in files.values() for v in f["volumes"]],
           _slider_update(z0, ra["dwi"].shape[2] - 1), _slider_update(m0, results["A"].protocol.n_meas - 1),
           V.tractogram3d(rb["tractogram"], regions, rb["shape"], total=rb["n_streamlines"]) if rb else None,
           V.matrices(rb["matrix"], rb["score"], gt_count) if rb else None,
           gr.update(visible=rb is not None),
           V.floor_slice(ra["floor"], z0, ra["floor_median"], label="A: "), accuracy(results["A"]),
           V.spread_matrices(spread) if spread else None,
           explore(ex, name_a, "minus the previous layer" if len(labels) > 1 else "signal", METRICS[0], z0, m0),
           explore(ex, name_a, "signal", METRICS[2], z0, m0), *ingredient_views(ex, z0), layer_table(ex),
           gr.update(choices=labels, value=name_a), _slider_update(z0, ra["dwi"].shape[2] - 1), _slider_update(m0, results["A"].protocol.n_meas - 1))
    timings.append(["page · figures", f"{time.perf_counter() - t:.2f}"])
    return out


def run_pipeline(compute_fn, *args, progress=None):
    """The run button, a generator: the stage texts of ``compute_fn`` (:func:`compute`, or it wrapped for a GPU
    pool) into the headline as they arrive (the other outputs untouched, the progress bar following), then the
    page's outputs from its payload (:func:`present`)."""
    import gradio as gr
    state = _load()
    if state["error"]:
        raise gr.Error(f"the source did not load: {state['error']}")
    if progress:
        progress(0.0, desc="starting")
    payload = None
    try:
        for item in compute_fn(*args):
            if isinstance(item, dict):
                payload = item
            else:
                text, fraction = item
                if progress:
                    progress(fraction, desc=text.split("**")[1] if "**" in text else text)
                yield _status(text)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    yield _status(f"**drawing** … (device held {payload['compute_seconds']:.0f} s)")
    yield present(payload, state["load_seconds"], state["regions"], state["source"].gt_count, state["source"].accuracy)


GPU_TIERS = (("logged out", 120), ("free account", 300), ("PRO", 2400))    # ZeroGPU's daily quota per visitor tier, seconds


def estimated_seconds(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """The GPU seconds a run reserves on the shared pool, from its inputs (the same positional inputs as
    :func:`run_pipeline`). Measured on the pool at DiSCo 364 with every tier after the compute/page split (#7): the
    worker's start and the payload's handoff 8 s, noise to scoring 8 s, the first replay 0.065 s per measurement
    (the tiles uploaded inside the call), each further replay on the resident tiles 0.028 s per measurement (the
    ladder is three, B one plus its 8 s of stages), 10 s per extra tracker key; times 1.3, within 30 and 480 s. The
    pool refuses a request above the visitor's daily quota (:data:`GPU_TIERS`) and kills a run that outlives its
    reservation, so this is the measured cost with its margin, not a generous one."""
    try:
        cfg = P.config()
        protocol = _protocol_from_inputs(cfg, preset, n_b0, *rest[len(PHYSICS_FIELDS):], scheme=scheme_file, full=P.mode(cfg) == "full")
        n = protocol.n_meas
    except Exception:
        return 480
    secs = 16 + 0.065 * n + (3 * 0.028 * n if ladder_on else 0) + ((8 + 0.028 * n) if knob != NO_KNOB else 0) + 10 * (int(n_keys) - 1)
    return int(min(480, max(30, 1.3 * secs)))


def gpu_seconds_text(*args):
    """The readout under the run button on the pool: the seconds this run reserves against the tiers' quotas, and
    which tiers can run it (a request above a visitor's daily quota is refused by the pool before it starts)."""
    secs = estimated_seconds(*args)
    fits = [name for name, cap in GPU_TIERS if secs <= cap]
    who = ("a visitor " + ", ".join(fits)) if fits else "no tier: split the run (drop B, the ladder or the extra keys)"
    caps = ", ".join(f"{name} {cap // 60} min" for name, cap in GPU_TIERS)
    return (f"**This run reserves {secs} s of GPU.** ZeroGPU grants each visitor a daily quota ({caps}) and refuses a "
            f"single request above it (\"larger than the maximum allowed\"): this run can be started by {who}. "
            f"Log in to Hugging Face in this browser to use your own quota.")


def _slider_update(value, maximum):
    import gradio as gr
    return gr.update(value=int(value), maximum=int(maximum))


def redraw_explorer(rs, layer, mode, metric, z, m):
    ex = rs and rs.get("explorer")
    return (explore(ex, layer, mode, metric, z, m), explore(ex, layer, mode, METRICS[2] if metric == METRICS[0] else metric, z, m), *ingredient_views(ex, z))


def redraw_slice(rs, z, m, overlay, which):
    if not rs or which not in rs:
        return None, None
    r = rs[which]
    return (V.dwi_slice(r["dwi"], r["meas"], int(z), int(m), peaks=r["peaks"], peak_amp=r["peak_amp"], overlay=bool(overlay), label=f"{which}: "),
            V.floor_slice(r["floor"], int(z), r["floor_median"], label=f"{which}: "))


def ground_truth_views():
    """The strands figure and the ground-truth matrices, drawn once per process."""
    state = _load()
    if state["error"]:
        return None, None
    if state["gt_views"] is None:
        src = state["source"]
        state["gt_views"] = (V.strands3d(state["strands_vox"], state["diameters_m"], state["regions"], src.mask.shape),
                             V.ground_truth_matrix(src.gt_count, src.gt_area, P.connected_pairs(src.gt_count)))
    return state["gt_views"]


def build(runner=None):
    """The Blocks. ``runner`` wraps :func:`compute` for the run button (the ZeroGPU entry passes ``spaces.GPU(...)``):
    the device part of a run; the page draws from its payload in this process."""
    import gradio as gr
    cfg = P.config()
    full = P.mode(cfg) == "full"
    shapes = cfg["shapes"]; presets = ["DiSCo 364"] + list(cfg["presets"]) + [CUSTOM] + ([UPLOADED] if full else [])
    shape_names = list(shapes); labels = {n: shapes[n]["label"] for n in shape_names}
    with gr.Blocks(title="DiSCo replay to tractogram", delete_cache=(3600, 3600)) as demo:
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
                                        "(one TE and one pulse kind per run, square pulses, the field along z) and a Camino `.scheme` file "
                                        "can be uploaded; a run takes minutes, the plan shown first says how many.")
                        else:
                            gr.Markdown("**Demo mode**: a shell's **pulse timing** (δ, Δ; TE 53.5 ms, square pulses) is one of the stored classes, "
                                        "because the layout holds each walker's response to that pulse shape; the b-value, the directions and their "
                                        "number, the tissue, the field, the SNR and the tracker are free. Stored: " + "; ".join(f"`{n}` = {labels[n]}" for n in shape_names)
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
                                b0_mode = gr.Dropdown(list(B0_MODES) + [FREE_B0], value=list(B0_MODES)[0], label="B0 direction")
                                theta = gr.Slider(0, 180, value=0, step=1, label="polar angle from z (°)")
                                phi = gr.Slider(0, 360, value=0, step=1, label="azimuth from x (°)")
                            with gr.Row():
                                tier_relax = gr.Checkbox(value=True, label="relaxation (T2, T1)")
                                tier_contact = gr.Checkbox(value=True, label="contact (surface relaxivity ρ)")
                                tier_field = gr.Checkbox(value=True, label="field (myelin susceptibility)")
                            n = len(POOLS)
                            with gr.Row():
                                T2s = [gr.Number(value=c0[k], label=f"T2 {q} (ms)") for k, q in enumerate(POOLS)]
                            with gr.Row():
                                T1s = [gr.Number(value=c0[n + k], label=f"T1 {q} (ms)") for k, q in enumerate(POOLS)]
                            with gr.Row():
                                rho = gr.Number(value=c0[2 * n], label="ρ (µm/s)"); chi_iso = gr.Number(value=c0[2 * n + 1], label="χ_iso of the sheath, the field source (ppm)")
                                chi_aniso = gr.Number(value=c0[2 * n + 2], label="Δχ_a of the sheath (ppm)")
                            with gr.Row():
                                catalogue_note = gr.Markdown(c0[2 * n + 3])
                                reset = gr.Button("reset to the catalogue at this field", size="sm")
                            gr.Markdown("The pack's walkers live in the intra- and extra-axonal pools (its spec names a myelin pool nobody was seeded "
                                        "in). On this phantom the field's **direction** and the **stimulated echo** move the signal most; 3 T against "
                                        "7 T on a PGSE is small (the 180° refocuses the static dephasing), and at 7 T the catalogue's two T2 coincide.")
                        physics_inputs = [physics_on, field_T, b0_mode, theta, phi, *T2s, *T1s, rho, chi_iso, chi_aniso, tier_relax, tier_contact, tier_field]
                        assert len(physics_inputs) == len(PHYSICS_FIELDS)
                        with gr.Row():
                            snr_on = gr.Checkbox(value=True, label="add Rician noise")
                            snr = gr.Slider(5, 100, value=30, step=1, label="SNR at M0 (a full water voxel before relaxation; each voxel's b = 0 SNR follows its tissue)")
                        density = gr.Slider(1, 4, value=cfg["tracking"]["density"], step=1, label="seeds per region voxel (density³)")
                        max_angle = gr.Slider(10, 60, value=cfg["tracking"]["max_angle"], step=1, label="max angle (°)")
                        step_mm = gr.Slider(0.25, 1.0, value=cfg["tracking"]["step_mm"], step=0.05, label="step (voxels; the grid is the mm frame)")
                        key = gr.Number(value=0, precision=0, label="random key")
                        knob = gr.Dropdown(list(knobs(cfg)), value=NO_KNOB, label="B: the same run with one knob changed")
                        scanner = gr.Dropdown([NO_SCANNER] + list(P.scanner_classes()), value=NO_SCANNER,
                                              label="scanner gradient limit (the catalogue's classes): a shell it cannot play refuses the run")
                        gradients = gr.Markdown()
                        n_keys = gr.Slider(1, 8, value=1, step=1, label="repeat A's tracking over N keys (the tractogram's own spread)")
                        ladder_on = gr.Checkbox(value=not full, visible=not full, label="Replay DWI Explorer: replay A's tier ladder too (bare, +relaxation, +contact; noise-free; about 30 s more of GPU time at DiSCo 364)")
                        go = gr.Button("replay → CSD → track → score", variant="primary")
                        gpu_text = gr.Markdown(visible=runner is not None)
                        headline = gr.Markdown()
            with gr.Tab("2 · ground truth") as gt_tab:
                strands_view = gr.Plot(label="the strands")
                gt_matrix = gr.Image(label="the ground-truth matrices", type="pil")
            with gr.Tab("3 · Replay DWI Explorer"):
                gr.Markdown("What the replay made, before the noise and the tractography. **A** is the run you configured in the first tab "
                            "(its replay before the noise); **B** is the same run with the one knob you chose there changed, and exists only "
                            "when a knob is set. **Ingredients**: what each tier multiplies into every walker's term, reduced per voxel. "
                            "**Layers**: A's walk replayed with A's tiers switched on one at a time, ending at A itself, then B; look at a "
                            "layer, its difference to the previous one, or B minus A, in the DWI itself or in the tensor's MD and FA from the "
                            "b ≤ 1500 shells; divide by the replay floor to see where a difference means something.")
                with gr.Row():
                    ingredient_pool = gr.Image(label="relaxation tier: intra-axonal weight fraction", type="pil")
                    ingredient_contact = gr.Image(label="contact tier: the walkers' wall contact", type="pil")
                    ingredient_field = gr.Image(label="field tier: dephasing phase spread at the echo", type="pil")
                with gr.Row():
                    layer_choice = gr.Dropdown(["A"], value="A", label="layer")
                    mode = gr.Radio(list(EXPLORE_MODES), value=EXPLORE_MODES[0], label="show")
                    metric = gr.Radio(list(METRICS), value=METRICS[0], label="quantity")
                with gr.Row():
                    ez_slider = gr.Slider(0, 39, value=20, step=1, label="axial slice z")
                    em_slider = gr.Slider(0, 363, value=0, step=1, label="measurement (DWI only)")
                with gr.Row():
                    explore_view = gr.Image(label="the chosen layer and mode", type="pil")
                    metric_view = gr.Image(label="the same for FA (or the chosen metric)", type="pil")
                layer_tbl = gr.Dataframe(headers=["from", "to", "b (s/mm²)", "median |ΔS|", "99 % |ΔS|"], label="layer differences per shell, and the replay floor", interactive=False)
            with gr.Tab("4 · DiSCo results"):
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
                spread_view = gr.Image(label="A over N tracker keys: mean and spread per pair", type="pil")
                with gr.Accordion("accuracy: what this replay is an approximation of", open=False):
                    gr.Markdown("A replay is a measured approximation of the stored walk, not a rendering: the walk's two halves are replayed "
                                "separately and their disagreement per voxel is the **floor** below which a signal difference means nothing; "
                                "the layout keeps K temporal bands of each walker's path and the **band error** is what the dropped bands "
                                "would have added at the built gradient amplitude.")
                    with gr.Row():
                        floor_view = gr.Image(label="split-half floor (this run, the slice above)", type="pil")
                        accuracy = gr.Dataframe(headers=["what", "value"], label="the source and this run", interactive=False, wrap=True)
                with gr.Row():
                    timings = gr.Dataframe(headers=["stage", "seconds"], label="timings", interactive=False)
                    with gr.Column():
                        tck = gr.File(label="tractograms (.tck, MRtrix): every streamline, and a 10k sample", file_count="multiple")
                        volumes = gr.File(label="DWI (.nii.gz) with bvals/bvecs, and the FOD SH field (.nii.gz, tournier07 order 8)", file_count="multiple")
        compute_fn = compute if runner is None else runner(compute)      # the device part alone runs under a pool's GPU

        def run_with_progress(*args, progress=gr.Progress()):
            yield from run_pipeline(compute_fn, *args, progress=progress)
        tissue_numbers = [*T2s, *T1s, rho, chi_iso, chi_aniso, catalogue_note]
        field_preset.change(lambda name: [field_presets[name]] + catalogue_numbers(field_presets[name]), inputs=field_preset,
                            outputs=[field_T] + tissue_numbers, show_progress="hidden")
        reset.click(catalogue_numbers, inputs=field_T, outputs=tissue_numbers, show_progress="hidden")
        def show_gradients(preset_, n_b0_, scanner_, scheme_, *shells_):
            try:
                prot = _protocol_from_inputs(cfg, preset_, n_b0_, *shells_, scheme=scheme_, full=full)
            except (ValueError, KeyError) as e:
                return f"({e})"
            return gradient_text(prot, cfg["shapes"], scanner_)[0]
        for ctl in (preset, scanner, scheme_file, *shell_inputs):
            ctl.change(show_gradients, inputs=[preset, n_b0, scanner, scheme_file, *shell_inputs], outputs=gradients, show_progress="hidden")
        outputs = dict(result=result, headline=headline, dwi_view=dwi_view, tract_view=tract_view, mats=mats, timings=timings, tck=tck,
                       volumes=volumes, z_slider=z_slider, m_slider=m_slider, tract_view_b=tract_view_b, mats_b=mats_b, b_row=b_row,
                       floor_view=floor_view, accuracy=accuracy, spread_view=spread_view, explore_view=explore_view, metric_view=metric_view,
                       ingredient_pool=ingredient_pool, ingredient_contact=ingredient_contact, ingredient_field=ingredient_field,
                       layer_table=layer_tbl, layer_choice=layer_choice, ez_slider=ez_slider, em_slider=em_slider)
        for ctl in (layer_choice, mode, metric, ez_slider, em_slider):
            ctl.change(redraw_explorer, inputs=[result, layer_choice, mode, metric, ez_slider, em_slider],
                       outputs=[explore_view, metric_view, ingredient_pool, ingredient_contact, ingredient_field], show_progress="hidden")
        run_inputs = [preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *physics_inputs, *shell_inputs]
        if runner is not None:                                   # the pool: what the run reserves, live as the inputs change
            gr.on([c.change for c in run_inputs] + [demo.load], gpu_seconds_text, inputs=run_inputs, outputs=gpu_text, show_progress="hidden")
        go.click(run_with_progress, inputs=run_inputs,
                 outputs=[outputs[name] for name in OUTPUTS], concurrency_limit=1, api_name="run_pipeline")   # the endpoint tools/live.py drives
        for ctl in (z_slider, m_slider, overlay, which):
            ctl.change(redraw_slice, inputs=[result, z_slider, m_slider, overlay, which], outputs=[dwi_view, floor_view], show_progress="hidden")
        gt_tab.select(ground_truth_views, outputs=[strands_view, gt_matrix])
        demo.load(lambda: (_load().get("error") and f"**the source did not load:** {_load()['error']}") or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
