"""The Spaces' page, four tabs over :mod:`space.pipeline` and the configured source (:mod:`space.sources`).

The acquisition tab: a preset (the source's), custom shells (on a stored pulse timing; at their own δ / Δ / TE, or
from an uploaded Camino scheme, in DiSCo's full mode); the source's tissue-and-scanner panel; the noise; the
tracker; the scanner (ideal, or a catalogued machine whose every catalogued term the replay applies at the phantom's
place in the bore, and whose gradient limit refuses a shell it cannot play; docs/scanner.md); B, the same run with one
knob changed; N
tracker keys for the tractogram's own spread; the run button, whose progress names the stage it is in. The truth
tab: the source's input and ground truth. The Replay DWI Explorer: the ingredient maps of the tiers, the noise-free
layers (the tier ladder, A, B) and their differences in the DWI or in MD / FA, per shell, and the estimated against
the true responses where the source has them. The results tab: a DWI slice with the FODs' principal directions,
the tractograms and connectomes of A and B against the source's truth, the spread over keys, the accuracy and round
trip tables, the timings, the downloads.

The page is built from the source's class (:meth:`space.pipeline.Source.describe`, ``presets``, ``panel``,
``knobs``, ``estimated_seconds``) before any data loads; a run is :func:`prepare_runs` on the host (a lookup of the
source's share that needs no device, in the page's cache), :func:`compute` on the device (which computes the share
the page had not cached first, so the GPU call is entered as soon as the request arrives), :func:`present` back on
the host. Nothing scientific lives here: every number comes from the pipeline or the source, every figure from
:mod:`space.viewers`."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import numpy as np
from dataclasses import replace

from . import pipeline as P
from . import sources
from . import viewers as V

MAX_SHELLS = 4
CUSTOM = "custom shells"
UPLOADED = "uploaded scheme"
PER_SHELL = 7                                            # on, timing class, b, directions, delta, Delta, TE
NO_KNOB = P.NO_KNOB
IDEAL = P.IDEAL
RESPONSES_ROW = "worker · the packs' responses not cached on the page"     # the timings row tools/live.py prints
SAMPLE = 10_000                                          # streamlines kept in the page's state and in the sample .tck
OUTPUTS = ("result", "headline", "dwi_view", "tract_view", "mats", "timings", "tck", "volumes", "z_slider", "m_slider",
           "tract_view_b", "mats_b", "b_row", "floor_view", "accuracy", "spread_view",
           "explore_view", "metric_view", "ingredient_pool", "ingredient_contact", "ingredient_field", "layer_table", "layer_choice",
           "ez_slider", "em_slider", "truth_view", "fractions_view", "lobar_view", "roundtrip", "response_view")   # the run button's outputs, in order
EXPLORE_MODES = ("signal", "minus the previous layer", "B − A", "(B − A) ÷ floor")
METRICS = ("DWI", "MD (µm²/ms)", "FA")
_state = {"source": None, "error": None}
_lock = threading.Lock()
_RUNS = tempfile.mkdtemp(prefix="disco-runs-")          # one directory per run tag, replaced on every run


def _load():
    """The source, loaded once per process (a failed load is retried on the next call)."""
    with _lock:
        if _state["source"] is None:
            try:
                t0 = time.perf_counter()
                cfg = P.config()
                src = sources.source(cfg)
                _state["warm"] = src.warm_in_background()           # the brain's warm-up runs beside the serving page
                _state["regions"] = V.region_markers(src.regions)
                _state["gt_views"] = None
                _state["source"] = src; _state["cfg"] = cfg; _state["load_seconds"] = time.perf_counter() - t0; _state["error"] = None
            except Exception as e:                       # shown on the page instead of a dead Space; the next call tries again
                _state["error"] = repr(e)
        return _state


def _protocol_from_inputs(S, cfg, preset, n_b0, *shell_inputs, scheme=None, full=False):
    """The acquisition the page asks for: a preset of the source class ``S``, the shell rows (in full mode each with
    its own delta / Delta / TE in ms, else on a stored timing class), or in full mode an uploaded Camino scheme."""
    if preset == UPLOADED:
        if not full:
            raise ValueError("a scheme upload needs full mode (DISCO_MODE=full with the columnar pack)")
        if not scheme:
            raise ValueError("upload a Camino .scheme file")
        return P.protocol_from_scheme(scheme)
    if preset in S.presets(cfg):
        return S.protocol(cfg, preset)
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


def physics_values(S, cfg, *values):
    """The panel's inputs as a dict by the source class ``S``'s :attr:`~space.pipeline.Panel.fields`."""
    fields = S.panel(cfg).fields
    if len(values) != len(fields):
        raise ValueError(f"the physics panel has {len(fields)} inputs, got {len(values)}")
    return dict(zip(fields, values))


def gradient_text(cfg, protocol, shapes, scanner):
    """The per-shell gradient table as Markdown, and whether every shell is playable on the menu's ``scanner``."""
    rows = P.playable(protocol, shapes, P.machine(cfg, scanner))
    lines = ["| shell | b (s/mm²) | timing | needs | limit |", "|---|---|---|---|---|"]
    for name, b, delta, Delta, G, G_max, ok in rows:
        limit = "any" if G_max is None else f"{G_max * 1e3:.0f} mT/m {'✓' if ok else '✗ cannot play'}"
        lines.append(f"| {name} | {b:g} | δ {delta * 1e3:g} / Δ {Delta * 1e3:g} ms | {G * 1e3:.0f} mT/m | {limit} |")
    return "\n".join(lines), all(r[-1] for r in rows)


def plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs):
    """The runs the button asks for, validated before any work: ``[(tag, protocol, snr_on, snr, physics)]`` for A
    and, with a knob, B; refused with the reason when the scanner cannot play a shell or the source cannot do the
    run."""
    S = type(source)
    full = source.mode == "full"
    protocol = _protocol_from_inputs(S, cfg, preset, n_b0, *shell_inputs, scheme=scheme_file, full=full)
    key = P.machine(cfg, scanner)
    runs = [("A", protocol, snr_on, snr, S.physics_from(cfg, values, scanner=key))]
    knobs = S.knobs(cfg)
    if knob not in knobs:
        raise ValueError(f"unknown knob {knob!r}")
    change = knobs[knob]
    if change is not None:
        if key is not None and change[0] in ("field", "b0"):
            raise ValueError(f"the {scanner} fixes the field and its direction: the knob {knob!r} would make B the same run; "
                             "choose the ideal scanner to vary them")
        pb, on_b, snr_b, vb = S.apply_knob(cfg, change, protocol, snr_on, snr, values)
        runs.append(("B", pb, on_b, snr_b, S.physics_from(cfg, vb, scanner=key)))
    for tag, prot, _, _, physics in runs:
        table, ok = gradient_text(cfg, prot, source.shapes, scanner)
        if not ok:
            raise ValueError(f"{scanner} cannot play run {tag}'s shells (square pulses):\n\n{table}")
        source.validate(prot, physics)
    return runs


def _split(S, cfg, rest):
    """The run signature's tail: the physics panel's values (a dict) and the shell rows."""
    n = len(S.panel(cfg).fields)
    return physics_values(S, cfg, *rest[:n]), rest[n:]


def response_entries(source, runs, ladder_on):
    """The :meth:`~space.pipeline.Source.prepare` entries of :func:`plan_runs`'s ``runs``: ``[(slot, meas, physics)]``
    for A, B and, with the ladder, A's rungs; ``slot`` is ``("runs", tag)`` or ``("ladder", i)``."""
    out = [(("runs", tag), P.measurements(prot, source.shapes), ph) for tag, prot, _, _, ph in runs]
    if ladder_on:
        meas_a = out[0][1]
        out += [(("ladder", i), meas_a, ph) for i, (_, ph) in enumerate(source.ladder_steps(runs[0][4]))]
    return out


def prepare_runs(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """The source's share of the runs the inputs ask for as the page's process has it cached
    (:meth:`~space.pipeline.Source.cached`), looked up and never computed, so the GPU call follows the request at
    once: ``{"runs": {tag: ...}, "ladder": [...] or None}``, None for an entry not cached (or a source with nothing
    to prepare); :func:`compute` computes the missing ones."""
    state = _load()
    if state["error"]:
        raise ValueError(f"the source did not load: {state['error']}")
    source, cfg = state["source"], state["cfg"]
    values, shell_inputs = _split(type(source), cfg, rest)
    runs = plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs)
    out = dict(runs={}, ladder=[] if ladder_on else None)
    for (where, k), meas, ph in response_entries(source, runs, ladder_on):
        hit = source.cached(meas, ph)
        if where == "runs":
            out["runs"][k] = hit
        else:
            out["ladder"].append(hit)
    return out


def _fill(source, runs, ladder_on, prepared):
    """:func:`prepare_runs`'s entries the page had not cached, computed here (inside the GPU call), a generator: a
    stage text when there are any, then returns ``(prepared, {response_key: result}, seconds)``: the entries
    complete, the ones computed here by key, the time (None for a source with nothing to prepare)."""
    entries = [(slot, meas, ph) for slot, meas, ph in response_entries(source, runs, ladder_on) if source.response_key(meas, ph) is not None]
    if not entries:
        return prepared, {}, None
    out = dict(runs=dict(prepared["runs"]), ladder=None if prepared["ladder"] is None else list(prepared["ladder"]))
    missing = [(slot, meas, ph) for slot, meas, ph in entries if out[slot[0]][slot[1]] is None]
    if missing:
        yield (f"**computing the packs' responses on the worker: not cached on the page** ({len(missing)} of {len(entries)}) …", 0.0)
    t0 = time.perf_counter()
    computed = {}
    for (where, k), meas, ph in missing:
        r = source.cached(meas, ph)                     # an earlier entry of this run with the same key (B = A but its M0)
        if r is None:
            r = source.prepare(meas, ph)
            computed[source.response_key(meas, ph)] = r
        out[where][k] = r
    return out, computed, time.perf_counter() - t0


def _result_state(res, sample, load_seconds):
    """What the results tab needs, kept per session: float32 volumes, the peaks, the streamline sample, the numbers."""
    pk, amp = P.peaks(res.sh)
    fr = res.extras.get("fractions")
    return dict(dwi=res.dwi.astype(np.float32), floor=res.floor.astype(np.float32), floor_median=P.floor_stats(res)["median"], meas=res.meas,
                peaks=pk.astype(np.float32), peak_amp=amp.astype(np.float32), matrix=res.matrix, score=res.score, seconds=res.seconds,
                load_seconds=load_seconds, tractogram=sample, n_streamlines=len(res.tractogram), shape=res.dwi.shape[:3], name=res.protocol.name,
                extras=dict(fractions=None if fr is None else fr.astype(np.float16)))


def _layer_labels(physics, knob):
    """The explorer's names for A and B: A is the ladder's top rung, named by every tier it has on; B is A with the
    knob."""
    tiers = [q for q in ("relaxation", "contact", "field") if physics and getattr(physics, q)]
    a = "A = bare" + "".join(f" + {q}" for q in tiers) if tiers else "A = bare diffusion"
    return a, f"B = A with {knob}"


def _explorer_state(results, ladder, ingredients, knob, source):
    """What the Replay DWI Explorer keeps per session: the noise-free layers (the ladder, then A, then B) as float16
    volumes with their tensor maps, the replay floor, the ingredient images' layers, the per-shell layer differences."""
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
                meas=ra.meas, mask=mask, ingredients=source.ingredient_layers(ingredients), differences=P.layer_differences(layers, ra.meas, mask),
                snr=ra.snr, physics=ra.physics)


def _status(text):
    import gradio as gr
    keep = gr.update()
    return (keep, text) + (keep,) * (len(OUTPUTS) - 2)


def _one_run(tag, source, protocol, snr_on, snr, tracking, physics, prepared, t0):
    """One pipeline run as a generator of ``(text, fraction)`` stage updates, then the :class:`Result` last: ``tag``
    is A or B in the stage text."""
    texts = source.describe(source.cfg)["stages"]
    for item in P.run_stages(source, protocol, snr=(float(snr) if snr_on else None), tracking=tracking, physics=physics, prepared=prepared):
        if isinstance(item, P.Result):
            yield item
            return
        stage, k, n = item
        yield (f"**{tag} · {k + 1}/{n} {texts[stage]}** … ({protocol.n_meas} measurements, {time.perf_counter() - t0:.0f} s so far)", (k + 0.5) / (n + 1))


def _write_files(tag, res, source):
    """The run's files under its tag (the previous run's under the same tag replaced): the full tractogram and a
    sample as .tck, the DWI and FOD volumes, the source's own files; ``(sample, {"tck": [...], "volumes": [...]})``."""
    out = os.path.join(_RUNS, tag); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    stem = f"{source.describe(source.cfg)['files']}_{tag}_{res.protocol.name.replace(' ', '_')}"
    tck = os.path.join(out, f"{stem}.tck"); res.tractogram.to_tck(tck)
    sample = P.sample_tractogram(res.tractogram, SAMPLE)
    sample_path = os.path.join(out, f"{stem}_sample{SAMPLE // 1000}k.tck"); sample.to_tck(sample_path)
    vols = P.write_volumes(res, out, prefix=stem, affine=source.affine)
    return sample, dict(tck=[tck, sample_path], volumes=[vols["dwi"], vols["bvals"], vols["bvecs"], vols["fod"]] + source.extra_files(res, out, stem))


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
    if not np.any(ex["floor"] > 0):
        return V.map_slice(d, z, "this source has no per-voxel replay floor: B − A, " + (f"{metric} at measurement {m}" if k == 0 else metric), symmetric=True)
    return V.map_slice(d / np.where(ex["floor"] > 0, ex["floor"], np.nan), z, f"(B − A) ÷ replay floor, {metric}" + (f" at measurement {m}" if k == 0 else ""), symmetric=True)


def ingredient_views(ex, z):
    """The explorer's three ingredient images at slice ``z``, from the source's layers of the run's ingredients."""
    layers = (ex or {}).get("ingredients") or [None, None, None]
    z = int(z)
    return tuple(None if item is None or item[1] is None else V.map_slice(item[1], z, f"{item[0]}, z = {z}", **item[2]) for item in layers)


def layer_table(ex):
    rows = [[a, b, f"{sh:g}", f"{med:.4f}", f"{p99:.4f}"] for a, b, sh, med, p99 in ex["differences"]] if ex else []
    if ex and np.any(ex["floor"] > 0):
        f = ex["floor"][ex["mask"]]
        rows.append(["replay floor", "", "", f"{np.median(f):.4f}", f"{np.quantile(f, 0.99):.4f}"])
    return rows


def compute(prepared, preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """Everything a run needs the device for, a generator: ``(text, fraction)`` stage updates while it works, then
    the payload last, a dict of plain data (:class:`P.Result` per tag, the explorer's ladder and ingredient maps, the
    tracker-key spread, the seconds of each step, the source's share the page had not cached as computed here by key
    and its seconds, the clock at the handoff). ``prepared`` is :func:`prepare_runs`'s, ``rest`` the physics panel
    then the shell rows. On a shared GPU pool this runs in the forked worker and its yields cross to the page's
    process; nothing here draws or writes a file, so the device is held for the compute alone."""
    state = _load()
    if state["error"]:
        raise ValueError(f"the source did not load: {state['error']}")
    source, cfg = state["source"], state["cfg"]
    S = type(source)
    values, shell_inputs = _split(S, cfg, rest)
    runs = plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs)
    for tag, prot, _, _, physics in runs:                # full mode: what each run reads, before any byte moves
        plan = source.plan(P.measurements(prot, source.shapes), physics)
        if plan:
            yield (f"**{tag}: full replay of {plan['rows']:,} rows, {plan['bytes'] / 1e9:.1f} GB to read, about "
                   f"{plan['estimated_seconds'] / 60:.0f} min** (bands {plan['K']}, field modes {plan['modes']})", 0.0)
    t0 = time.perf_counter()
    prepared, kernels, response_seconds = yield from _fill(source, runs, ladder_on, prepared)
    tracking = S.tracking(cfg, density=density, max_angle=max_angle, step=step_mm, key=key)
    results = {}; seconds = {}
    try:
        for tag, prot, on, s_, ph in runs:
            for item in _one_run(tag, source, prot, on, s_, tracking, ph, prepared["runs"][tag], t0):
                if isinstance(item, P.Result):
                    results[tag] = item
                else:
                    yield item
        meas_a = results["A"].meas; physics_a = runs[0][4]
        ladder = []
        if ladder_on and physics_a is not None and not physics_a.bare:
            yield (f"**Replay DWI Explorer · the tier ladder of A, noise-free** … ({time.perf_counter() - t0:.0f} s so far)", 0.8)
            t = time.perf_counter(); ladder = source.ladder(meas_a, physics_a, prepared["ladder"]); seconds["explorer · ladder"] = time.perf_counter() - t
        yield (f"**Replay DWI Explorer · the ingredient maps** … ({time.perf_counter() - t0:.0f} s so far)", 0.85)
        t = time.perf_counter(); ingredients = source.ingredients(meas_a, physics_a, prepared["runs"]["A"]); seconds["explorer · ingredients"] = time.perf_counter() - t
        spread = None
        if int(n_keys) > 1:                                 # A's tracking repeated over further keys: the tractogram's own spread
            keys = [int(key) + 1 + i for i in range(int(n_keys) - 1)]
            mats = [results["A"].matrix]; scores = [results["A"].score]
            for i, (k, M, sc, secs) in enumerate(P.repeat_tracking(results["A"], source, tracking, keys)):
                mats.append(M); scores.append(sc)
                yield (f"**A · tracking again with key {k} ({i + 2}/{int(n_keys)})** … ({time.perf_counter() - t0:.0f} s so far)", 0.9)
            spread = P.pair_spread(mats, scores, key=source.score_key())
    finally:
        source.release()
    yield dict(results={tag: _light(r) for tag, r in results.items()}, ladder=[(label, vol.astype(np.float32)) for label, vol in ladder],
               ingredients=ingredients, spread=spread, knob=knob, seconds=seconds, kernels=kernels, response_seconds=response_seconds,
               compute_seconds=time.perf_counter() - t0, handed_off_at=time.time())


def _light(res):
    """``res`` with its DWI volumes in float32 for the handoff: what the page keeps, draws, fits and writes is float32
    or narrower, so the payload carries half the bytes and the compute's own arithmetic is untouched."""
    return replace(res, dwi=res.dwi.astype(np.float32), clean=None if res.clean is None else res.clean.astype(np.float32))


def present(payload, state):
    """The page's outputs (:data:`OUTPUTS`, in order) from a compute payload: the files, the per-session states, the
    headline and the figures (A's source share from the page's cache, which holds it after :func:`run_pipeline`).
    Runs where the page runs, never on the device."""
    import gradio as gr
    received = time.time()
    source = state["source"]; load_seconds = state["load_seconds"]; regions = state["regions"]
    results = payload["results"]; ladder = payload["ladder"]; ingredients = payload["ingredients"]; spread = payload["spread"]; knob = payload["knob"]
    post = {}
    if payload.get("response_seconds") is not None:
        post[RESPONSES_ROW] = payload["response_seconds"]
    post.update(payload["seconds"])
    t = time.perf_counter()
    samples = {}; files = {}
    for tag, r in results.items():
        samples[tag], files[tag] = _write_files(tag, r, source)
    post["page · files"] = time.perf_counter() - t; t = time.perf_counter()
    rs = {tag: _result_state(r, samples[tag], load_seconds) for tag, r in results.items()}
    ex = _explorer_state(results, ladder, ingredients, knob, source)
    post["page · states"] = time.perf_counter() - t; t = time.perf_counter()
    labels = [l for l, _ in ex["layers"]]; name_a = labels[-2] if "B" in results else labels[-1]
    ra = rs["A"]; rb = rs.get("B")
    headline = source.score_text("A", results["A"])
    if rb:
        headline += "<br>" + source.score_text("B", results["B"]) + "<br>" + source.compare_text(source.compare(results["A"], results["B"]), knob)
    if spread:
        headline += (f"<br>**A over {spread['n']} tracker keys: {spread['key']} {spread['pearson_mean']:.3f} ± {spread['pearson_std']:.3f}**, "
                     f"median pair count CV {spread['cv_median']:.2f}, {spread['pairs_always']} pairs in every run, {spread['pairs_any']} in any.")
    d = source.describe(source.cfg)
    headline += " The Replay DWI Explorer is the third tab, the results the fourth."
    z0 = ra["dwi"].shape[2] // 2
    m0 = int(np.flatnonzero(~ra["meas"].b0)[0]) if (~ra["meas"].b0).any() else 0
    timings = [[f"{tag} · {r}", t_] for tag in rs for r, t_ in V.timings_rows(rs[tag]["seconds"], rs[tag]["load_seconds"])] if rb else V.timings_rows(ra["seconds"], ra["load_seconds"])
    timings += [["device held (compute)", f"{payload['compute_seconds']:.2f}"], ["handoff to the page", f"{received - payload['handed_off_at']:.2f}"]]
    timings += [[k, f"{v:.2f}"] for k, v in post.items()]
    rs["explorer"] = ex                                  # the page state: A, B and the explorer's layers
    box = V.grid_box(ra["shape"], source.affine) if d["views"]["truth"] else ra["shape"]
    pa = source.cached(results["A"].meas, results["A"].physics)
    out = (rs, headline, V.dwi_slice(ra["dwi"], ra["meas"], z0, m0, peaks=ra["peaks"], peak_amp=ra["peak_amp"], overlay=True, label="A: "),
           V.tractogram3d(ra["tractogram"], regions, box, total=ra["n_streamlines"]),
           source.matrices(results["A"].matrix, results["A"].score, results["A"].reference),
           timings, [t_ for f in files.values() for t_ in f["tck"]], [v for f in files.values() for v in f["volumes"]],
           _slider_update(z0, ra["dwi"].shape[2] - 1), _slider_update(m0, results["A"].protocol.n_meas - 1),
           V.tractogram3d(rb["tractogram"], regions, box, total=rb["n_streamlines"]) if rb else None,
           source.matrices(results["B"].matrix, results["B"].score, results["B"].reference) if rb else None,
           gr.update(visible=rb is not None),
           V.floor_slice(ra["floor"], z0, ra["floor_median"], label="A: ") if d["views"]["floor"] else None, source.accuracy(results["A"]),
           V.spread_matrices(spread) if spread else None,
           explore(ex, name_a, "minus the previous layer" if len(labels) > 1 else "signal", METRICS[0], z0, m0),
           explore(ex, name_a, "signal", METRICS[2], z0, m0), *ingredient_views(ex, z0), layer_table(ex),
           gr.update(choices=labels, value=name_a), _slider_update(z0, ra["dwi"].shape[2] - 1), _slider_update(m0, results["A"].protocol.n_meas - 1),
           source.truth_view(ra["dwi"], ra["meas"], z0, m0), source.fractions_view(ra["extras"], z0),
           source.lobar(results["A"].matrix, results["A"].score, results["A"].reference), source.roundtrip_rows(results["A"]),
           source.response_view(pa, results["A"].extras))
    timings.append(["page · figures", f"{time.perf_counter() - t:.2f}"])
    return out


def run_pipeline(compute_fn, *args, progress=None):
    """The run button, a generator: the source's share of the runs the page has cached (:func:`prepare_runs`, a
    lookup), then the stage texts of ``compute_fn`` (:func:`compute`, or it wrapped for a GPU pool) into the headline
    as they arrive (the other outputs untouched, the progress bar following), then the share the call computed kept
    in the page's cache and the page's outputs from its payload (:func:`present`)."""
    import gradio as gr
    state = _load()
    if state["error"]:
        raise gr.Error(f"the source did not load: {state['error']}")
    if progress:
        progress(0.0, desc="starting")
    payload = None
    try:
        yield _status("**starting the run** (the packs' responses looked up on the page) …")
        prepared = prepare_runs(*args)
        for item in compute_fn(prepared, *args):
            if isinstance(item, dict):
                payload = item
            else:
                text, fraction = item
                if progress:
                    progress(fraction, desc=text.split("**")[1] if "**" in text else text)
                yield _status(text)
    except (ValueError, KeyError) as e:
        raise gr.Error(str(e))
    if payload.get("kernels"):
        state["source"].keep(payload["kernels"])
    yield _status(f"**drawing** … (device held {payload['compute_seconds']:.0f} s)")
    yield present(payload, state)


GPU_TIERS = (("logged out", 120), ("free account", 300), ("PRO", 2400))    # ZeroGPU's daily quota per visitor tier, seconds


def estimated_seconds(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """The GPU seconds a run reserves on the shared pool, from its inputs (the same positional inputs as
    :func:`run_pipeline`): the configured source's measured cost model
    (:meth:`~space.pipeline.Source.estimated_seconds`) with the source's share the call computes because the page has
    it not cached (:func:`uncached_responses`). The pool refuses a request above the visitor's daily quota
    (:data:`GPU_TIERS`) and kills a run that outlives its reservation, so this is the measured cost with its margin,
    not a generous one; 480 when the inputs make no protocol."""
    try:
        cfg = P.config()
        S = sources.source_class(cfg)
        n = len(S.panel(cfg).fields)
        protocol = _protocol_from_inputs(S, cfg, preset, n_b0, *rest[n:], scheme=scheme_file, full=S.mode == "full")
    except Exception:
        return 480
    responses = uncached_responses(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest)
    try:
        machine = P.machine(cfg, scanner)
    except ValueError:
        return 480
    return S.estimated_seconds(cfg, protocol, density=density, knob=knob, n_keys=n_keys, ladder=ladder_on, responses=responses, scanner=machine)


def uncached_responses(preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *rest):
    """``(state, n_meas, saves)`` per entry of the source's share of the run the inputs ask for that the page's
    process has not cached (:meth:`~space.pipeline.Source.responses`): what the GPU call computes before its replay. Empty when
    the source is not loaded in this process (the pool's entry loads it before the page serves) or refuses the run
    (it is refused before the GPU call)."""
    source = _state["source"]
    if source is None:
        return []
    cfg = _state["cfg"]
    try:
        values, shell_inputs = _split(type(source), cfg, rest)
        runs = plan_runs(cfg, source, preset, n_b0, snr_on, snr, scheme_file, knob, scanner, values, shell_inputs)
    except (ValueError, KeyError):
        return []
    return source.responses([(meas, ph) for _, meas, ph in response_entries(source, runs, ladder_on)])


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
        return None, None, None, None
    r = rs[which]; source = _load()["source"]
    views = source.describe(source.cfg)["views"]
    return (V.dwi_slice(r["dwi"], r["meas"], int(z), int(m), peaks=r["peaks"], peak_amp=r["peak_amp"], overlay=bool(overlay), label=f"{which}: "),
            V.floor_slice(r["floor"], int(z), r["floor_median"], label=f"{which}: ") if views["floor"] else None,
            source.truth_view(r["dwi"], r["meas"], int(z), int(m)), source.fractions_view(r["extras"], int(z)))


def ground_truth_views():
    """The truth tab's views, drawn once per process."""
    state = _load()
    if state["error"]:
        return None, None
    if state["gt_views"] is None:
        state["gt_views"] = state["source"].truth_views()
    return state["gt_views"]


def _widget(c):
    """The Gradio component of a panel :class:`~space.pipeline.Control` (an input kind)."""
    import gradio as gr
    common = dict(label=c.label, visible=c.visible, interactive=c.interactive)
    if c.kind == "checkbox":
        return gr.Checkbox(value=bool(c.value), **common)
    if c.kind == "number":
        return gr.Number(value=c.value, **common)
    if c.kind == "slider":
        return gr.Slider(c.minimum, c.maximum, value=c.value, step=c.step, **common)
    if c.kind == "dropdown":
        return gr.Dropdown(list(c.choices), value=c.value, **common)
    raise ValueError(f"unknown control kind {c.kind!r}")


def build(runner=None, cfg=None):
    """The Blocks of the configured source (``cfg``, else :func:`space.pipeline.config`). ``runner`` wraps
    :func:`compute` for the run button (the ZeroGPU entry passes ``spaces.GPU(...)``): the device part of a run; the
    page looks up its cache before it and draws from its payload after it, in this process."""
    import gradio as gr
    cfg = cfg or P.config()
    S = sources.source_class(cfg)
    d = S.describe(cfg)
    full = S.mode == "full"
    shapes = S.shapes_of(cfg); presets = S.presets(cfg) + [CUSTOM] + ([UPLOADED] if full else [])
    shape_names = [n for n in shapes]
    panel = S.panel(cfg); tc = S.tracking_controls(cfg)
    with gr.Blocks(title=d["title"], delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            d["heading"]
            + ("\n\n**Before you press run:** the GPU time of a run is charged to *your* Hugging Face quota, not the Space's: "
               "2 minutes a day logged out, 5 with a free account, 40 with PRO. The line under the run button says how many "
               "seconds the configured run reserves and which of those can start it; a request above your quota is refused "
               "before it starts, so log in to Hugging Face in this browser if you want more than one run a day." if runner is not None else "")
            + "\n\n**Fixed here:** " + "; ".join(d["fixed"]) + ".")
        result = gr.State(None)
        with gr.Tabs():
            with gr.Tab("1 · acquisition, tissue and scanner"):
                with gr.Row():
                    with gr.Column(scale=1):
                        preset = gr.Dropdown(presets, value=presets[0], label="acquisition")
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
                        gr.Markdown(d["acquisition"])
                        scheme_file = gr.File(label="uploaded scheme: Camino STEJSKALTANNER (.scheme)", file_count="single", type="filepath", visible=full)
                    with gr.Column(scale=1):
                        field_presets = {f"{f:g} T": float(f) for f in panel.field_presets}
                        widgets = {}; catalogue_note = reset = field_preset = None
                        with gr.Accordion("tissue and scanner: the physics tiers", open=True):
                            for row in panel.rows:
                                with gr.Row():
                                    for c in row:
                                        if c.kind == "field_preset":
                                            field_preset = gr.Dropdown(list(c.choices), value=c.value, label=c.label)
                                        elif c.kind == "catalogue_note":
                                            catalogue_note = gr.Markdown(c.value)
                                        elif c.kind == "reset":
                                            reset = gr.Button(c.label, size="sm")
                                        elif c.kind == "markdown":
                                            gr.Markdown(c.value)
                                        else:
                                            widgets[c.name] = _widget(c)
                            gr.Markdown(d["tissue"])
                        physics_inputs = [widgets[name] for name in panel.fields]
                        with gr.Row():
                            snr_on = gr.Checkbox(value=True, label="add Rician noise")
                            snr = gr.Slider(5, 100, value=30, step=1, label="SNR at M0 (a full water voxel before relaxation; each voxel's b = 0 SNR follows its tissue)")
                        density = gr.Slider(tc["density"][0], tc["density"][1], value=tc["density"][2], step=tc["density"][3], label=tc["density"][4])
                        max_angle = gr.Slider(tc["max_angle"][0], tc["max_angle"][1], value=tc["max_angle"][2], step=tc["max_angle"][3], label=tc["max_angle"][4])
                        step_mm = gr.Slider(tc["step"][0], tc["step"][1], value=tc["step"][2], step=tc["step"][3], label=tc["step"][4])
                        key = gr.Number(value=0, precision=0, label="random key")
                        knob = gr.Dropdown(list(S.knobs(cfg)), value=NO_KNOB, label="B: the same run with one knob changed")
                        scanner = gr.Dropdown([IDEAL] + list(P.machines(cfg)), value=IDEAL,
                                              label="scanner: a catalogued machine sets the field and its direction, plays every term "
                                                    "its catalogue entry carries, and refuses a shell beyond its gradient limit")
                        scanner_terms = gr.Markdown(P.scanner_text(cfg, IDEAL, S.refused_machines(cfg)))
                        gradients = gr.Markdown()
                        n_keys = gr.Slider(1, 8, value=1, step=1, label="repeat A's tracking over N keys (the tractogram's own spread)")
                        ladder_on = gr.Checkbox(value=not full, visible=not full, label=d["ladder"])
                        go = gr.Button(d["run_label"], variant="primary")
                        gpu_text = gr.Markdown(visible=runner is not None)
                        headline = gr.Markdown()
            with gr.Tab(d["truth_tab"]) as gt_tab:
                strands_view = gr.Plot(label=d["truth_labels"][0])
                gt_matrix = gr.Image(label=d["truth_labels"][1], type="pil")
                gr.Markdown(f"**The truth:** {d['truth']}.")
            with gr.Tab("3 · Replay DWI Explorer"):
                gr.Markdown(d["explorer"])
                with gr.Row():
                    ingredient_pool = gr.Image(label=d["ingredients"][0], type="pil")
                    ingredient_contact = gr.Image(label=d["ingredients"][1], type="pil")
                    ingredient_field = gr.Image(label=d["ingredients"][2], type="pil")
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
                response_view = gr.Image(label="estimated vs true response per tissue", type="pil", visible=d["views"]["response"])
            with gr.Tab(d["results_tab"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        dwi_view = gr.Image(label="DWI slice", type="pil")
                        with gr.Row():
                            z_slider = gr.Slider(0, 39, value=20, step=1, label="axial slice z")
                            m_slider = gr.Slider(0, 363, value=0, step=1, label="measurement")
                            overlay = gr.Checkbox(value=True, label="FOD principal directions")
                            which = gr.Radio(["A", "B"], value="A", label="run")
                    with gr.Column(scale=1, visible=d["views"]["truth"]):
                        truth_view = gr.Image(label="the same slice with the input FOD's principal directions", type="pil")
                    with gr.Column(scale=1):
                        tract_view = gr.Plot(label="tractogram A")
                fractions_view = gr.Image(label="recovered vs input fractions", type="pil", visible=d["views"]["fractions"])
                mats = gr.Image(label="connectome A vs its truth", type="pil")
                lobar_view = gr.Image(label="connectome A vs its truth over the lobar groups", type="pil", visible=d["views"]["lobar"])
                with gr.Row(visible=False) as b_row:
                    tract_view_b = gr.Plot(label="tractogram B")
                    mats_b = gr.Image(label="connectome B vs its truth", type="pil")
                spread_view = gr.Image(label="A over N tracker keys: mean and spread per pair", type="pil")
                roundtrip = gr.Dataframe(headers=["what", "value"], label="the round trip: the reconstruction and the connectome against the input",
                                         interactive=False, wrap=True, visible=d["views"]["roundtrip"])
                with gr.Accordion("accuracy: what this replay is an approximation of", open=False):
                    gr.Markdown(d["accuracy"])
                    with gr.Row():
                        floor_view = gr.Image(label="split-half floor (this run, the slice above)", type="pil", visible=d["views"]["floor"])
                        accuracy = gr.Dataframe(headers=["what", "value"], label="the source and this run", interactive=False, wrap=True)
                with gr.Row():
                    timings = gr.Dataframe(headers=["stage", "seconds"], label="timings", interactive=False)
                    with gr.Column():
                        tck = gr.File(label="tractograms (.tck, MRtrix): every streamline, and a 10k sample", file_count="multiple")
                        volumes = gr.File(label=d["volumes_label"], file_count="multiple")
        compute_fn = compute if runner is None else runner(compute)      # the device part alone runs under a pool's GPU

        def run_with_progress(*args, progress=gr.Progress()):
            yield from run_pipeline(compute_fn, *args, progress=progress)
        tissue_numbers = [widgets[name] for name in panel.catalogue] + [catalogue_note]
        field_preset.change(lambda name: [field_presets[name]] + S.catalogue_numbers(cfg, field_presets[name]), inputs=field_preset,
                            outputs=[widgets["field_T"]] + tissue_numbers, show_progress="hidden")
        reset.click(lambda f: S.catalogue_numbers(cfg, f), inputs=widgets["field_T"], outputs=tissue_numbers, show_progress="hidden")

        def show_gradients(preset_, n_b0_, scanner_, scheme_, *shells_):
            try:
                prot = _protocol_from_inputs(S, cfg, preset_, n_b0_, *shells_, scheme=scheme_, full=full)
            except (ValueError, KeyError) as e:
                return f"({e})"
            return gradient_text(cfg, prot, shapes, scanner_)[0]

        def choose_scanner(label, field_now):
            """The menu's machine: its term table, its field on the slider and the catalogue's tissue at it (the ideal
            scanner keeps the field as set)."""
            key = P.machine(cfg, label)
            f = float(field_now) if key is None else float(P.limits(key).field_T)
            return [P.scanner_text(cfg, label, S.refused_machines(cfg)), f] + (S.catalogue_numbers(cfg, f) if key is not None else [gr.update()] * len(tissue_numbers))
        scanner.change(choose_scanner, inputs=[scanner, widgets["field_T"]], outputs=[scanner_terms, widgets["field_T"]] + tissue_numbers,
                       show_progress="hidden")
        for ctl in (preset, scanner, scheme_file, *shell_inputs):
            ctl.change(show_gradients, inputs=[preset, n_b0, scanner, scheme_file, *shell_inputs], outputs=gradients, show_progress="hidden")
        outputs = dict(result=result, headline=headline, dwi_view=dwi_view, tract_view=tract_view, mats=mats, timings=timings, tck=tck,
                       volumes=volumes, z_slider=z_slider, m_slider=m_slider, tract_view_b=tract_view_b, mats_b=mats_b, b_row=b_row,
                       floor_view=floor_view, accuracy=accuracy, spread_view=spread_view, explore_view=explore_view, metric_view=metric_view,
                       ingredient_pool=ingredient_pool, ingredient_contact=ingredient_contact, ingredient_field=ingredient_field,
                       layer_table=layer_tbl, layer_choice=layer_choice, ez_slider=ez_slider, em_slider=em_slider, truth_view=truth_view,
                       fractions_view=fractions_view, lobar_view=lobar_view, roundtrip=roundtrip, response_view=response_view)
        for ctl in (layer_choice, mode, metric, ez_slider, em_slider):
            ctl.change(redraw_explorer, inputs=[result, layer_choice, mode, metric, ez_slider, em_slider],
                       outputs=[explore_view, metric_view, ingredient_pool, ingredient_contact, ingredient_field], show_progress="hidden")
        run_inputs = [preset, n_b0, snr_on, snr, density, max_angle, step_mm, key, scheme_file, knob, scanner, n_keys, ladder_on, *physics_inputs, *shell_inputs]
        if runner is not None:                                   # the pool: what the run reserves, live as the inputs change
            gr.on([c.change for c in run_inputs] + [demo.load], gpu_seconds_text, inputs=run_inputs, outputs=gpu_text, show_progress="hidden")
        run_event = go.click(run_with_progress, inputs=run_inputs,
                             outputs=[outputs[name] for name in OUTPUTS], concurrency_limit=1, api_name="run_pipeline")   # the endpoint tools/live.py drives
        if runner is not None:                                   # the run's responses are now cached on the page: the reservation falls
            run_event.then(gpu_seconds_text, inputs=run_inputs, outputs=gpu_text, show_progress="hidden")
        for ctl in (z_slider, m_slider, overlay, which):
            ctl.change(redraw_slice, inputs=[result, z_slider, m_slider, overlay, which], outputs=[dwi_view, floor_view, truth_view, fractions_view], show_progress="hidden")
        gt_tab.select(ground_truth_views, outputs=[strands_view, gt_matrix])
        demo.load(lambda: (_load().get("error") and f"**the source did not load:** {_load()['error']}") or "", outputs=headline)
    return demo


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()          # download and warm while the page comes up
    build().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
